#!/usr/bin/env python3
"""Batch smoke: LoRA from-start, post-prefill, and ICL-FC baseline for finished adapters."""

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
OUT_DIR = ROOT / "result_d2l" / "smoke_synth_finished_latest_r8_down"
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
            "error_type": "smoke:checker_exception",
        }
    return {
        "valid": bool(checker.get("valid")),
        "checker": checker,
        "parsed_calls": calls,
        "malformed_call": len(calls) != 1,
    }


def discover_jobs(hinted):
    by_hash = {}
    for row in hinted:
        by_hash.setdefault(tools_hash(row["function"]), []).append(row)
    jobs, orphans = [], []
    for path in sorted(ADAPTER_DIR.glob("*/adapter_model.safetensors")):
        h = path.parent.name
        cases = by_hash.get(h, [])
        if not cases:
            orphans.append(h)
            continue
        meta = path.parent / "train_meta.json"
        tool = (
            json.loads(meta.read_text()).get("tool_name")
            if meta.is_file()
            else cases[0]["function"][0]["name"]
        )
        jobs.append(
            {
                "hash": h,
                "path": str(path.parent),
                "tool": tool,
                "entry": cases[0],  # one case per adapter
            }
        )
    return jobs, orphans


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
    jobs, orphans = discover_jobs(hinted)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Finished adapters: {len(jobs) + len(orphans)} "
        f"(eval-compatible={len(jobs)}, orphans={orphans})"
    )

    # ICL baseline from existing FC run (score file lists failures only after summary).
    icl_rows = load_jsonl(ICL_SCORE)
    icl_fails = {r["id"]: r for r in icl_rows[1:] if "id" in r}
    icl_results = {r["id"]: r for r in load_jsonl(ICL_RESULT)}

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
    table = []
    for job in jobs:
        h = job["hash"]
        entry = job["entry"]
        cid = entry["id"]
        print(f"\n##### {job['tool']} {h} {cid} #####", flush=True)

        icl_valid = cid not in icl_fails
        icl_raw = icl_results.get(cid, {}).get("result", "")
        icl_rec = {
            "id": cid,
            "adapter_hash": h,
            "tool": job["tool"],
            "condition": "icl_fc",
            "valid": icl_valid,
            "raw_output": icl_raw,
            "parsed_calls": None,
            "checker": (
                None
                if icl_valid
                else {
                    "valid": False,
                    "error": icl_fails[cid].get("error"),
                    "error_type": icl_fails[cid].get("error_type"),
                    "model_result_decoded": icl_fails[cid].get(
                        "model_result_decoded"
                    ),
                }
            ),
            "user": entry["question"][0][0]["content"],
            "source": str(ICL_RESULT),
        }
        if not icl_valid:
            icl_rec["parsed_calls"] = [
                {"name": k, "arguments": v}
                for d in (icl_fails[cid].get("model_result_decoded") or [])
                for k, v in d.items()
            ]
        records.append(icl_rec)
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
        row = {
            "tool": job["tool"],
            "id": cid,
            "hash": h,
            "icl_fc": icl_valid,
        }
        for name, kwargs in conds:
            rec = run_one(
                worker,
                adapter_hash=h,
                condition=name,
                kwargs=kwargs,
                entry=entry,
                possible_answer=gt[cid],
            )
            records.append(rec)
            row[name] = rec["valid"]
            print(
                f"  [{'PASS' if rec['valid'] else 'FAIL'}] {name:36s} "
                f"lat={rec['latency_sec']:.2f}s "
                f"out={rec['raw_output'][:100]!r}"
            )
        table.append(row)

    (OUT_DIR / "results.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False)
    )

    n = len(table)
    n_icl = sum(1 for r in table if r["icl_fc"])
    n_fs = sum(1 for r in table if r["correct_from_start"])
    n_pp = sum(1 for r in table if r["base_prefill_replay_1_to_correct"])
    print("\n=== SUMMARY ===")
    print(
        f"{'tool':32s} {'case':22s} {'ICL':5s} {'from_start':10s} {'post_prefill':12s}"
    )
    for r in table:
        print(
            f"{r['tool'][:32]:32s} {r['id']:22s} "
            f"{'Y' if r['icl_fc'] else 'N':5s} "
            f"{'Y' if r['correct_from_start'] else 'N':10s} "
            f"{'Y' if r['base_prefill_replay_1_to_correct'] else 'N':12s}"
        )
    print(f"\nTotals over {n} cases: ICL {n_icl}/{n}  from_start {n_fs}/{n}  post_prefill {n_pp}/{n}")
    # agreement
    both_fs_icl = sum(1 for r in table if r["icl_fc"] == r["correct_from_start"])
    both_pp_fs = sum(
        1
        for r in table
        if r["correct_from_start"] == r["base_prefill_replay_1_to_correct"]
    )
    print(f"Agreement ICL↔from_start: {both_fs_icl}/{n}")
    print(f"Agreement from_start↔post_prefill: {both_pp_fs}/{n}")

    summary = {
        "n_finished_adapters": len(jobs) + len(orphans),
        "n_eval_compatible": n,
        "orphans": orphans,
        "icl_fc": n_icl,
        "correct_from_start": n_fs,
        "post_prefill": n_pp,
        "agreement_icl_from_start": both_fs_icl,
        "agreement_from_start_post": both_pp_fs,
        "per_case": table,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_DIR}")
    worker._stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
