#!/usr/bin/env python3
"""Full live_simple attach eval: LoRA from-start vs post-prefill (+ ICL lookup)."""

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
ADAPTER_DIR = Path(
    "/home/jah649/tool-lora/doc-to-lora/train_outputs/"
    "live_simple_synth_toolcall_r8_down"
)
D2L_ROOT = Path("/home/jah649/tool-lora/doc-to-lora")
D2L_PYTHON = D2L_ROOT / ".venv/bin/python"
PEFT_WORKER = ROOT / "bfcl_eval/model_handler/local_inference/peft_worker.py"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ICL_SCORE = (
    ROOT
    / "score/Qwen_Qwen3-4B-Instruct-2507-FC/live/BFCL_v4_live_simple_score.json"
)
ICL_RESULT = (
    ROOT
    / "result/Qwen_Qwen3-4B-Instruct-2507-FC/live/BFCL_v4_live_simple_result.json"
)
OUT_DIR = ROOT / "result_d2l" / "synth_toolcall_r8_down_attach_full"
MAX_NEW_TOKENS = 1024


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score(text, entry, possible_answer):
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
            "error_type": "attach_eval:checker_exception",
        }
    return {
        "valid": bool(checker.get("valid")),
        "checker": checker,
        "parsed_calls": calls,
        "malformed_call": len(calls) != 1,
    }


def run_one(worker, *, adapter_hash, condition, kwargs, entry, possible_answer):
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
    scored = score(result["text"], entry, possible_answer)
    return {
        "id": entry["id"],
        "adapter_hash": adapter_hash,
        "tool": entry["function"][0]["name"],
        "condition": condition,
        "latency_sec": time.perf_counter() - t0,
        "raw_output": result["text"],
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        **scored,
    }


def main() -> int:
    raw = [
        json.loads(line)
        for line in (ROOT / "bfcl_eval/data/BFCL_v4_live_simple.json")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    hinted = add_language_specific_hint_to_function_doc(json.loads(json.dumps(raw)))
    gt = {
        json.loads(line)["id"]: json.loads(line)["ground_truth"]
        for line in (
            ROOT / "bfcl_eval/data/possible_answer/BFCL_v4_live_simple.json"
        )
        .read_text()
        .splitlines()
        if line.strip()
    }

    adapters = {}
    for entry in hinted:
        h = tools_hash(entry["function"])
        path = ADAPTER_DIR / h
        if not (path / "adapter_model.safetensors").is_file():
            raise SystemExit(f"Missing adapter for {entry['id']}: {h}")
        adapters[h] = str(path)

    icl_fails = {
        r["id"]: r for r in load_jsonl(ICL_SCORE)[1:] if "id" in r
    }
    icl_results = {r["id"]: r for r in load_jsonl(ICL_RESULT)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUT_DIR / "results.jsonl"
    done = set()
    if out_jsonl.is_file():
        for row in load_jsonl(out_jsonl):
            done.add((row["id"], row["condition"]))

    print(f"Cases: {len(hinted)} unique adapters: {len(adapters)} done_pairs={len(done)}")

    worker = _D2LWorkerProxy(
        str(D2L_PYTHON),
        str(D2L_ROOT),
        gpu_device="0",
        worker_script=str(PEFT_WORKER),
        worker_args=[str(D2L_ROOT)],
    )
    worker.send("load_base", {"base_model": BASE_MODEL, "d2l_root": str(D2L_ROOT)})
    print(f"Preloading {len(adapters)} adapters ...", flush=True)
    worker.send("preload_adapters", {"adapters": adapters})
    print("Preload complete.", flush=True)

    with out_jsonl.open("a") as fout:
        for i, entry in enumerate(hinted, 1):
            cid = entry["id"]
            h = tools_hash(entry["function"])
            print(
                f"\n[{i}/{len(hinted)}] {cid} {entry['function'][0]['name']}",
                flush=True,
            )

            if (cid, "icl_fc") not in done:
                icl_valid = cid not in icl_fails
                rec = {
                    "id": cid,
                    "adapter_hash": h,
                    "tool": entry["function"][0]["name"],
                    "condition": "icl_fc",
                    "valid": icl_valid,
                    "raw_output": icl_results.get(cid, {}).get("result", ""),
                    "checker": None
                    if icl_valid
                    else {
                        "valid": False,
                        "error": icl_fails[cid].get("error"),
                        "error_type": icl_fails[cid].get("error_type"),
                    },
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                done.add((cid, "icl_fc"))
                print(f"  [{'PASS' if icl_valid else 'FAIL'}] icl_fc")

            conds = [
                (
                    "correct_from_start",
                    {
                        "start_adapter": h,
                        "end_adapter": h,
                        "switch_at": None,
                        "cache_policy": "preserve",
                        "replay_prompt_tokens": 0,
                    },
                ),
                (
                    "base_prefill_replay_1_to_correct",
                    {
                        "start_adapter": h,
                        "end_adapter": h,
                        "switch_at": None,
                        "cache_policy": "preserve",
                        "prefill_adapter": None,
                        "replay_last_prompt_token": True,
                        "replay_prompt_tokens": 1,
                    },
                ),
            ]
            for name, kwargs in conds:
                if (cid, name) in done:
                    print(f"  skip {name}")
                    continue
                rec = run_one(
                    worker,
                    adapter_hash=h,
                    condition=name,
                    kwargs=kwargs,
                    entry=entry,
                    possible_answer=gt[cid],
                )
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                done.add((cid, name))
                print(
                    f"  [{'PASS' if rec['valid'] else 'FAIL'}] {name:36s} "
                    f"lat={rec['latency_sec']:.2f}s"
                )

    final = {
        "icl_fc": 0,
        "correct_from_start": 0,
        "base_prefill_replay_1_to_correct": 0,
    }
    by_case = {}
    for row in load_jsonl(out_jsonl):
        by_case.setdefault(row["id"], {})[row["condition"]] = bool(row.get("valid"))
        if row.get("valid"):
            final[row["condition"]] = final.get(row["condition"], 0) + 1
    n_cases = len(by_case)
    summary = {
        "n_cases": n_cases,
        "icl_fc": final.get("icl_fc", 0),
        "correct_from_start": final.get("correct_from_start", 0),
        "post_prefill": final.get("base_prefill_replay_1_to_correct", 0),
        "icl_accuracy": final.get("icl_fc", 0) / max(n_cases, 1),
        "from_start_accuracy": final.get("correct_from_start", 0) / max(n_cases, 1),
        "post_prefill_accuracy": final.get("base_prefill_replay_1_to_correct", 0)
        / max(n_cases, 1),
        "agreement_icl_from_start": sum(
            1
            for m in by_case.values()
            if m.get("icl_fc") == m.get("correct_from_start")
        ),
        "agreement_from_start_post": sum(
            1
            for m in by_case.values()
            if m.get("correct_from_start")
            == m.get("base_prefill_replay_1_to_correct")
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FINAL ===")
    print(json.dumps(summary, indent=2))
    worker._stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
