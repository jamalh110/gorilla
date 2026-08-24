#!/usr/bin/env python3
"""Smoke finished synth tool-call LoRAs: from-start vs post-prefill vs base."""

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
OUT_DIR = ROOT / "result_d2l" / "smoke_synth_finished_r8_down"
MAX_NEW_TOKENS = 1024
# One BFCL case per finished adapter (first id).
ONE_CASE_PER_ADAPTER = True


def load_rows():
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
    return hinted, gt


def finished_jobs(hinted):
    by_hash = {}
    for row in hinted:
        h = tools_hash(row["function"])
        by_hash.setdefault(h, []).append(row)

    jobs = []
    for path in sorted(ADAPTER_DIR.glob("*/adapter_model.safetensors")):
        h = path.parent.name
        cases = by_hash.get(h, [])
        if not cases:
            print(f"skip orphan adapter {h} (no hinted BFCL cases)")
            continue
        meta_path = path.parent / "train_meta.json"
        tool = (
            json.loads(meta_path.read_text()).get("tool_name")
            if meta_path.is_file()
            else cases[0]["function"][0]["name"]
        )
        selected = cases[:1] if ONE_CASE_PER_ADAPTER else cases
        jobs.append(
            {
                "hash": h,
                "path": str(path.parent),
                "tool": tool,
                "cases": selected,
            }
        )
    return jobs


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
            "error_type": "smoke:checker_exception",
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
        "user": entry["question"][0][0]["content"],
        **scored,
    }


def main() -> int:
    hinted, gt = load_rows()
    jobs = finished_jobs(hinted)
    if not jobs:
        raise SystemExit("No finished eval-compatible adapters found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Smoking {len(jobs)} adapters")

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
        {"adapters": {j["hash"]: j["path"] for j in jobs}},
    )

    records = []
    for job in jobs:
        h = job["hash"]
        for entry in job["cases"]:
            print(f"\n##### {job['tool']} {h} {entry['id']} #####", flush=True)
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
                    "base_only",
                    {
                        "start_adapter": None,
                        "end_adapter": None,
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
                rec = run_one(
                    worker,
                    adapter_hash=h,
                    condition=name,
                    kwargs=kwargs,
                    entry=entry,
                    possible_answer=gt[entry["id"]],
                )
                records.append(rec)
                mark = "PASS" if rec["valid"] else "FAIL"
                print(
                    f"  [{mark}] {name:36s} "
                    f"lat={rec['latency_sec']:.2f}s "
                    f"out={rec['raw_output'][:120]!r}"
                )

    (OUT_DIR / "results.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False)
    )

    # Compact table
    by_tool = {}
    for r in records:
        by_tool.setdefault((r["tool"], r["id"]), {})[r["condition"]] = r["valid"]
    print("\n=== SUMMARY ===")
    print(f"{'tool':30s} {'case':22s} {'from_start':10s} {'post_prefill':12s} {'base':6s}")
    n_fs = n_pp = n_base = 0
    for (tool, cid), m in by_tool.items():
        fs = m.get("correct_from_start", False)
        pp = m.get("base_prefill_replay_1_to_correct", False)
        b = m.get("base_only", False)
        n_fs += fs
        n_pp += pp
        n_base += b
        print(
            f"{tool:30s} {cid:22s} "
            f"{'Y' if fs else 'N':10s} {'Y' if pp else 'N':12s} {'Y' if b else 'N':6s}"
        )
    n = len(by_tool)
    print(f"\nTotals: from_start {n_fs}/{n}  post_prefill {n_pp}/{n}  base {n_base}/{n}")
    summary = {
        "n_adapters": len(jobs),
        "n_cases": n,
        "from_start": n_fs,
        "post_prefill": n_pp,
        "base_only": n_base,
        "per_case": {
            f"{tool}|{cid}": m for (tool, cid), m in by_tool.items()
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_DIR}")
    worker._stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
