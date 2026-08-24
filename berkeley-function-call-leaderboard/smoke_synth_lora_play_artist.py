#!/usr/bin/env python3
"""Smoke: attach-from-start vs post-prefill attach for the trained play_artist LoRA."""

from __future__ import annotations

import json
import time
from pathlib import Path

from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.model_handler.local_inference.bfcl_tool_schema import tools_hash
from bfcl_eval.model_handler.local_inference.doc_to_lora import (
    TOOL_CALL_SYSTEM_MSG,
    _D2LWorkerProxy,
    _parse_tool_calls,
)
from bfcl_eval.utils import add_language_specific_hint_to_function_doc

ROOT = Path(__file__).resolve().parent
CASE_ID = "live_simple_93-54-0"
ADAPTER_HASH = "031442ba1fbf4022"
ADAPTER_DIR = Path(
    "/home/jah649/tool-lora/doc-to-lora/train_outputs/"
    "live_simple_synth_toolcall_r8_down"
)
D2L_ROOT = Path("/home/jah649/tool-lora/doc-to-lora")
D2L_PYTHON = D2L_ROOT / ".venv/bin/python"
PEFT_WORKER = ROOT / "bfcl_eval/model_handler/local_inference/peft_worker.py"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OUT_DIR = ROOT / "result_d2l" / "smoke_synth_play_artist_r8_down"
MAX_NEW_TOKENS = 1024


def load_case() -> tuple[dict, list]:
    raw_rows = [
        json.loads(line)
        for line in (
            ROOT / "bfcl_eval/data/BFCL_v4_live_simple.json"
        ).read_text().splitlines()
        if line.strip()
    ]
    raw = next(e for e in raw_rows if e["id"] == CASE_ID)
    # Training keyed adapters without the BFCL Python syntax hint; eval scoring
    # still uses the hinted schema (same as the leaderboard checker).
    assert tools_hash(raw["function"]) == ADAPTER_HASH, tools_hash(raw["function"])
    entry = add_language_specific_hint_to_function_doc(
        [json.loads(json.dumps(raw))]
    )[0]
    gt = {
        json.loads(line)["id"]: json.loads(line)["ground_truth"]
        for line in (
            ROOT / "bfcl_eval/data/possible_answer/BFCL_v4_live_simple.json"
        ).read_text().splitlines()
        if line.strip()
    }
    return entry, gt[CASE_ID]


def score(text: str, entry: dict, possible_answer: list) -> dict:
    calls = _parse_tool_calls(text)
    decoded = [
        {c["name"]: c.get("arguments", {})} for c in calls if c.get("name")
    ]
    try:
        checker = ast_checker(
            entry["function"],
            decoded,
            possible_answer,
            Language.PYTHON,
            "live_simple",
            "doc-to-lora/qwen3-4b-peft",
        )
    except Exception as exc:
        checker = {
            "valid": False,
            "error": [f"{type(exc).__name__}: {exc}"],
            "error_type": "smoke:checker_exception",
        }
    return {
        "valid": bool(checker.get("valid")),
        "checker": checker,
        "parsed_calls": calls,
        "malformed_call": len(calls) != 1,
    }


def run_one(worker, *, condition: str, kwargs: dict, entry, possible_answer) -> dict:
    messages = [
        {"role": "system", "content": TOOL_CALL_SYSTEM_MSG},
        *entry["question"][0],
    ]
    t0 = time.perf_counter()
    result = worker.send(
        "generate_with_switch",
        {
            "messages": messages,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
            "enable_thinking": False,
            "restrict_toolgen": False,
            **kwargs,
        },
    )
    latency = time.perf_counter() - t0
    scored = score(result["text"], entry, possible_answer)
    record = {
        "id": entry["id"],
        "condition": condition,
        "latency_sec": latency,
        "raw_output": result["text"],
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "prefill_latency": result.get("prefill_latency"),
        "decode_latency": result.get("decode_latency"),
        "replay_prompt_tokens": result.get("replay_prompt_tokens"),
        "activation_events": result.get("activation_events"),
        **scored,
    }
    return record


def main() -> int:
    entry, possible_answer = load_case()
    adapter_path = ADAPTER_DIR / ADAPTER_HASH
    assert (adapter_path / "adapter_model.safetensors").is_file()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Case: {CASE_ID}")
    print(f"User: {entry['question'][0][0]['content']}")
    print(f"Gold: {possible_answer}")
    print(f"Adapter: {adapter_path}")

    worker = _D2LWorkerProxy(
        str(D2L_PYTHON),
        str(D2L_ROOT),
        gpu_device="0",
        worker_script=str(PEFT_WORKER),
        worker_args=[str(D2L_ROOT)],
    )
    worker.send("load_base", {"base_model": BASE_MODEL, "d2l_root": str(D2L_ROOT)})
    worker.send(
        "preload_adapters",
        {"adapters": {ADAPTER_HASH: str(adapter_path)}},
    )

    conditions = [
        (
            "correct_from_start",
            {
                "start_adapter": ADAPTER_HASH,
                "end_adapter": ADAPTER_HASH,
                "switch_at": None,
                "cache_policy": "preserve",
                "prefill_adapter": None,
                "replay_prompt_tokens": 0,
            },
        ),
        (
            "base_only",
            {
                "start_adapter": None,
                "end_adapter": None,
                "switch_at": None,
                "cache_policy": "preserve",
                "prefill_adapter": None,
                "replay_prompt_tokens": 0,
            },
        ),
        (
            "base_prefill_replay_1_to_correct",
            {
                "start_adapter": ADAPTER_HASH,
                "end_adapter": ADAPTER_HASH,
                "switch_at": None,
                "cache_policy": "preserve",
                "prefill_adapter": None,
                "replay_last_prompt_token": True,
                "replay_prompt_tokens": 1,
            },
        ),
    ]

    records = []
    for name, kwargs in conditions:
        print(f"\n===== {name} =====", flush=True)
        rec = run_one(
            worker, condition=name, kwargs=kwargs, entry=entry, possible_answer=possible_answer
        )
        records.append(rec)
        print(f"valid={rec['valid']} malformed={rec['malformed_call']} "
              f"latency={rec['latency_sec']:.2f}s out_tok={rec['output_tokens']}")
        print(f"raw:\n{rec['raw_output']}")
        print(f"parsed: {json.dumps(rec['parsed_calls'], ensure_ascii=False)}")
        if not rec["valid"]:
            print(f"checker: {json.dumps(rec['checker'], ensure_ascii=False)[:500]}")

    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    summary = {
        r["condition"]: {
            "valid": r["valid"],
            "malformed_call": r["malformed_call"],
            "latency_sec": r["latency_sec"],
            "output_tokens": r["output_tokens"],
            "parsed_calls": r["parsed_calls"],
        }
        for r in records
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")
    worker._stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
