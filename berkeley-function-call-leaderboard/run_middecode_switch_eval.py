#!/usr/bin/env python3
"""Run deterministic mid-decode D2L LoRA switching experiments.

The script talks directly to the existing PEFT and raw-D2L subprocess workers,
uses the same prompt contract as the BFCL handlers, and scores each generation
with BFCL's AST checker. Results are append-only JSONL so interrupted matrices
can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.model_handler.local_inference.bfcl_tool_schema import (
    build_tools_json,
    normalize_functions,
    tools_hash,
)
from bfcl_eval.model_handler.local_inference.doc_to_lora import (
    TOOL_CALL_SYSTEM_MSG,
    _D2LWorkerProxy,
    _parse_tool_calls,
)
from bfcl_eval.utils import add_language_specific_hint_to_function_doc

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "bfcl_eval/data/BFCL_v4_live_simple.json"
GROUND_TRUTH_PATH = (
    ROOT / "bfcl_eval/data/possible_answer/BFCL_v4_live_simple.json"
)
DEFAULT_ADAPTER_DIR = Path(
    "/home/jah649/tool-lora/doc-to-lora/train_outputs/"
    "live_simple_d2l_adapters"
)
DEFAULT_CHECKPOINT = Path(
    "/home/jah649/tool-lora/doc-to-lora/train_outputs/"
    "runs/nemotron_best/pytorch_model.bin"
)
DEFAULT_D2L_ROOT = Path("/home/jah649/tool-lora/doc-to-lora")
DEFAULT_D2L_PYTHON = DEFAULT_D2L_ROOT / ".venv/bin/python"
PEFT_WORKER = (
    ROOT
    / "bfcl_eval/model_handler/local_inference/peft_worker.py"
)
RAW_WORKER = (
    ROOT
    / "bfcl_eval/model_handler/local_inference/d2l_worker.py"
)
FAILURE_SCORE_PATH = (
    ROOT
    / "score_d2l/simple_live_schema_sft_noconstrained/"
    "doc-to-lora_qwen3-4b-peft/live/BFCL_v4_live_simple_score.json"
)

DIRECTIONS = {
    "base_to_correct": (None, "correct"),
    "correct_to_base": ("correct", None),
    "wrong_to_correct": ("alternate", "correct"),
    "correct_to_wrong": ("correct", "alternate"),
}
FIXED_BOUNDARIES = {
    "token_0": 0,
    "token_8": 8,
    "token_16": 16,
    "token_32": 32,
}
SEMANTIC_BOUNDARIES = {
    "after_think",
    "after_tool_call",
    "after_arguments_key",
    "eos",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_experiment_entries(path: Path) -> list[dict]:
    """Load BFCL rows with the same language hints used during generation."""
    return add_language_specific_hint_to_function_doc(load_jsonl(path))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def schema_features(functions: list[dict]) -> set[str]:
    features: set[str] = set()

    def visit(schema: dict, prefix: str) -> None:
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            features.add(f"type:{prefix}:{schema_type.casefold()}")
        for name, child in sorted(schema.get("properties", {}).items()):
            folded = name.casefold()
            features.add(f"param:{folded}")
            visit(child, f"{prefix}.{folded}")
        items = schema.get("items")
        if isinstance(items, dict):
            visit(items, f"{prefix}[]")

    for function in normalize_functions(functions):
        for token in re.findall(r"[a-z0-9]+", function.get("name", "").casefold()):
            features.add(f"name_token:{token}")
        visit(function.get("parameters", {}), "root")
    return features


def choose_alternates(
    entries: list[dict], available_hashes: set[str]
) -> dict[str, tuple[str, str]]:
    hash_to_functions: dict[str, list[dict]] = {}
    name_to_hashes: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        adapter_hash = tools_hash(entry["function"])
        if adapter_hash not in available_hashes:
            continue
        hash_to_functions.setdefault(adapter_hash, entry["function"])
        for function in entry["function"]:
            name_to_hashes[function["name"]].add(adapter_hash)

    feature_map = {
        adapter_hash: schema_features(functions)
        for adapter_hash, functions in hash_to_functions.items()
    }
    alternates: dict[str, tuple[str, str]] = {}
    for correct_hash, functions in sorted(hash_to_functions.items()):
        same_name = sorted(
            {
                candidate
                for function in functions
                for candidate in name_to_hashes[function["name"]]
                if candidate != correct_hash
            }
        )
        if same_name:
            alternates[correct_hash] = (same_name[0], "same_name_different_schema")
            continue

        correct_features = feature_map[correct_hash]
        candidates = []
        for candidate_hash, candidate_features in feature_map.items():
            if candidate_hash == correct_hash:
                continue
            union = correct_features | candidate_features
            similarity = (
                len(correct_features & candidate_features) / len(union)
                if union
                else 0.0
            )
            candidates.append((-similarity, candidate_hash))
        if not candidates:
            raise RuntimeError("at least two adapters are required")
        _, alternate_hash = min(candidates)
        alternates[correct_hash] = (alternate_hash, "schema_similar")
    return alternates


def load_failure_ids() -> list[str]:
    if not FAILURE_SCORE_PATH.is_file():
        return []
    rows = load_jsonl(FAILURE_SCORE_PATH)
    return [row["id"] for row in rows[1:] if row.get("id")]


def select_subsets(
    records: list[dict], seed: int, smoke_count: int, pilot_count: int
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    same_name = sorted(
        record["id"]
        for record in records
        if record["alternate_rule"] == "same_name_different_schema"
    )
    failures = sorted(
        set(load_failure_ids()) & {record["id"] for record in records}
    )
    remaining = sorted(record["id"] for record in records)
    rng.shuffle(same_name)
    rng.shuffle(failures)
    rng.shuffle(remaining)

    smoke: list[str] = []
    for pool in (same_name, failures, remaining):
        for entry_id in pool:
            if entry_id not in smoke:
                smoke.append(entry_id)
            if len(smoke) >= smoke_count:
                break
        if len(smoke) >= smoke_count:
            break

    pilot = list(smoke)
    target_same_name = min(20, pilot_count)
    for entry_id in same_name:
        if entry_id not in pilot:
            pilot.append(entry_id)
        if sum(item in set(same_name) for item in pilot) >= target_same_name:
            break
    for pool in (failures, remaining):
        for entry_id in pool:
            if entry_id not in pilot:
                pilot.append(entry_id)
            if len(pilot) >= pilot_count:
                break
        if len(pilot) >= pilot_count:
            break

    return {
        "smoke": smoke[:smoke_count],
        "pilot": pilot[:pilot_count],
        "full": sorted(record["id"] for record in records),
    }


def build_manifest(args) -> dict:
    entries = load_experiment_entries(args.data_path)
    available_hashes = {
        path.name
        for path in args.adapter_dir.iterdir()
        if path.is_dir() and (path / "adapter_model.safetensors").is_file()
    }
    alternates = choose_alternates(entries, available_hashes)
    manifest_entries = []
    for entry in entries:
        correct_hash = tools_hash(entry["function"])
        if correct_hash not in available_hashes:
            raise FileNotFoundError(
                f"missing exported adapter for {entry['id']}: {correct_hash}"
            )
        alternate_hash, alternate_rule = alternates[correct_hash]
        manifest_entries.append(
            {
                "id": entry["id"],
                "function_name": entry["function"][0]["name"],
                "correct_adapter": correct_hash,
                "alternate_adapter": alternate_hash,
                "alternate_rule": alternate_rule,
                "boundaries": {},
                "references": {},
            }
        )

    subsets = select_subsets(
        manifest_entries,
        args.seed,
        args.smoke_count,
        args.pilot_count,
    )
    manifest = {
        "version": 1,
        "seed": args.seed,
        "base_model": args.base_model,
        "checkpoint": str(args.checkpoint),
        "adapter_dir": str(args.adapter_dir),
        "data_path": str(args.data_path),
        "ground_truth_path": str(args.ground_truth_path),
        "num_entries": len(manifest_entries),
        "num_adapters": len(available_hashes),
        "cache_policies": ["preserve", "recompute"],
        "fixed_boundaries": FIXED_BOUNDARIES,
        "subsets": subsets,
        "entries": manifest_entries,
    }

    if args.manifest_path.is_file():
        old_manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
        old_by_id = {
            record["id"]: record for record in old_manifest.get("entries", [])
        }
        for record in manifest["entries"]:
            old = old_by_id.get(record["id"], {})
            if (
                old.get("correct_adapter") == record["correct_adapter"]
                and old.get("alternate_adapter") == record["alternate_adapter"]
            ):
                record["boundaries"] = old.get("boundaries", {})
                record["references"] = old.get("references", {})
    atomic_write_json(args.manifest_path, manifest)
    return manifest


def parse_csv_option(value: str, allowed: set[str], option: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(parsed) - allowed
    if unknown:
        raise ValueError(f"unknown {option}: {', '.join(sorted(unknown))}")
    return parsed


def messages_for_entry(entry: dict) -> list[dict]:
    return [
        {"role": "system", "content": TOOL_CALL_SYSTEM_MSG},
        *entry["question"][0],
    ]


def resolve_state(
    symbolic_state: str | None, manifest_record: dict
) -> str | None:
    if symbolic_state is None:
        return None
    if symbolic_state == "correct":
        return manifest_record["correct_adapter"]
    if symbolic_state == "alternate":
        return manifest_record["alternate_adapter"]
    raise ValueError(f"unknown symbolic adapter state: {symbolic_state}")


def score_text(
    *,
    text: str,
    entry: dict,
    possible_answer: list[dict],
    alternate_functions: list[dict],
) -> dict:
    calls = _parse_tool_calls(text)
    decoded = [
        {call["name"]: call.get("arguments", {})}
        for call in calls
        if call.get("name")
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
            "error_type": "middecode:checker_exception",
        }

    correct_function = entry["function"][0]
    alternate_function = alternate_functions[0]
    alternate_name = alternate_function["name"]
    correct_keys = set(
        correct_function.get("parameters", {}).get("properties", {})
    )
    alternate_keys = set(
        alternate_function.get("parameters", {}).get("properties", {})
    )
    adapter_leak = False
    if calls:
        first = calls[0]
        argument_keys = set((first.get("arguments") or {}).keys())
        adapter_leak = (
            first.get("name") == alternate_name
            and alternate_name != correct_function["name"]
        ) or bool((argument_keys & (alternate_keys - correct_keys)))
    return {
        "valid": bool(checker.get("valid")),
        "checker": checker,
        "parsed_calls": calls,
        "malformed_call": len(calls) != 1,
        "adapter_leak": adapter_leak,
    }


def record_key(record: dict) -> tuple:
    return (
        record.get("backend"),
        record.get("id"),
        record.get("condition"),
        record.get("boundary"),
        record.get("cache_policy"),
    )


def existing_records(path: Path) -> dict[tuple, dict]:
    if not path.is_file():
        return {}
    return {
        record_key(record): record
        for record in load_jsonl(path)
        if record.get("record_type") == "generation"
    }


def start_worker(args, backend: str):
    gpu = args.peft_gpu if backend == "peft" else args.raw_gpu
    worker_script = PEFT_WORKER if backend == "peft" else RAW_WORKER
    worker_args = [str(args.d2l_root)] if backend == "peft" else [
        str(args.d2l_root / "src")
    ]
    worker = _D2LWorkerProxy(
        str(args.d2l_python),
        str(args.d2l_root),
        gpu_device=gpu,
        worker_script=str(worker_script),
        worker_args=worker_args,
    )
    if backend == "peft":
        worker.send(
            "load_base",
            {
                "base_model": args.base_model,
                "d2l_root": str(args.d2l_root),
            },
        )
    else:
        worker.send("load_model", {"checkpoint_path": str(args.checkpoint)})
    return worker


def preload_worker(
    worker,
    backend: str,
    selected_manifest: list[dict],
    entries_by_id: dict[str, dict],
    manifest_by_hash: dict[str, list[dict]],
    args,
) -> None:
    needed_hashes = {
        adapter_hash
        for record in selected_manifest
        for adapter_hash in (
            record["correct_adapter"],
            record["alternate_adapter"],
        )
    }
    if backend == "peft":
        adapters = {
            adapter_hash: str(args.adapter_dir / adapter_hash)
            for adapter_hash in sorted(needed_hashes)
        }
        worker.send("preload_adapters", {"adapters": adapters})
    else:
        adapters = {
            adapter_hash: build_tools_json(manifest_by_hash[adapter_hash])
            for adapter_hash in sorted(needed_hashes)
        }
        worker.send(
            "preload_loras",
            {"adapters": adapters, "chunk_size": args.chunk_size},
        )


def run_generation(
    *,
    worker,
    backend: str,
    entry: dict,
    manifest_record: dict,
    alternate_functions: list[dict],
    possible_answer: list[dict],
    condition: str,
    boundary: str | None,
    cache_policy: str,
    start_adapter: str | None,
    end_adapter: str | None,
    switch_at: int | None,
    args,
    return_first_logits: bool = False,
    prefill_adapter: str | None = None,
    replay_last_prompt_token: bool = False,
    replay_prompt_tokens: int = 0,
) -> dict:
    started = time.perf_counter()
    base_record = {
        "record_type": "generation",
        "backend": backend,
        "subset": args.subset,
        "id": entry["id"],
        "function_name": entry["function"][0]["name"],
        "correct_adapter": manifest_record["correct_adapter"],
        "alternate_adapter": manifest_record["alternate_adapter"],
        "alternate_rule": manifest_record["alternate_rule"],
        "condition": condition,
        "boundary": boundary,
        "cache_policy": cache_policy,
        "start_adapter": start_adapter,
        "end_adapter": end_adapter,
        "switch_at": switch_at,
        "prefill_adapter": prefill_adapter,
        "replay_last_prompt_token": replay_last_prompt_token,
        "replay_prompt_tokens": replay_prompt_tokens,
    }
    try:
        result = worker.send(
            "generate_with_switch",
            {
                "messages": messages_for_entry(entry),
                "start_adapter": start_adapter,
                "end_adapter": end_adapter,
                "switch_at": switch_at,
                "cache_policy": cache_policy,
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0,
                "enable_thinking": False,
                "return_first_logits": return_first_logits,
                "restrict_toolgen": args.constrained,
                "constraint_tools": normalize_functions(entry["function"]),
                "strict_json_schema": True,
                "prefill_adapter": prefill_adapter,
                "replay_last_prompt_token": replay_last_prompt_token,
                "replay_prompt_tokens": replay_prompt_tokens,
            },
        )
        score = score_text(
            text=result["text"],
            entry=entry,
            possible_answer=possible_answer,
            alternate_functions=alternate_functions,
        )
        return {
            **base_record,
            "status": "ok",
            "wall_latency": time.perf_counter() - started,
            "raw_output": result.pop("text"),
            **score,
            "result": result,
        }
    except Exception as exc:
        return {
            **base_record,
            "status": "error",
            "wall_latency": time.perf_counter() - started,
            "valid": False,
            "malformed_call": True,
            "adapter_leak": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_reference_generate(
    *,
    worker,
    backend: str,
    entry: dict,
    manifest_record: dict,
    alternate_functions: list[dict],
    possible_answer: list[dict],
    args,
) -> dict:
    correct = manifest_record["correct_adapter"]
    if backend == "peft":
        worker.send("activate_adapter", {"adapter_name": correct})
    else:
        worker.send(
            "internalize",
            {
                "tool_defs": build_tools_json(entry["function"]),
                "chunk_size": args.chunk_size,
            },
        )
    started = time.perf_counter()
    try:
        result = worker.send(
            "generate",
            {
                "messages": messages_for_entry(entry),
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0,
                "enable_thinking": False,
            },
        )
        score = score_text(
            text=result["text"],
            entry=entry,
            possible_answer=possible_answer,
            alternate_functions=alternate_functions,
        )
        return {
            "record_type": "generation",
            "backend": backend,
            "subset": args.subset,
            "id": entry["id"],
            "function_name": entry["function"][0]["name"],
            "correct_adapter": correct,
            "alternate_adapter": manifest_record["alternate_adapter"],
            "alternate_rule": manifest_record["alternate_rule"],
            "condition": "model_generate_correct",
            "boundary": None,
            "cache_policy": "generate",
            "start_adapter": correct,
            "end_adapter": correct,
            "switch_at": None,
            "status": "ok",
            "wall_latency": time.perf_counter() - started,
            "raw_output": result.pop("text"),
            **score,
            "result": result,
        }
    except Exception as exc:
        return {
            "record_type": "generation",
            "backend": backend,
            "subset": args.subset,
            "id": entry["id"],
            "condition": "model_generate_correct",
            "boundary": None,
            "cache_policy": "generate",
            "status": "error",
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def boundary_value(name: str, reference: dict) -> int | None:
    if name in FIXED_BOUNDARIES:
        return FIXED_BOUNDARIES[name]
    if name in SEMANTIC_BOUNDARIES:
        return reference.get("semantic_boundaries", {}).get(name)
    raise ValueError(f"unknown boundary: {name}")


def skipped_boundary_record(
    *,
    backend: str,
    subset: str,
    entry_id: str,
    direction: str,
    boundary: str,
    policy: str,
) -> dict:
    return {
        "record_type": "generation",
        "backend": backend,
        "subset": subset,
        "id": entry_id,
        "condition": direction,
        "boundary": boundary,
        "cache_policy": policy,
        "status": "boundary_missing",
        "valid": False,
    }


def run_backend(args, manifest: dict, backend: str) -> None:
    entries = load_experiment_entries(args.data_path)
    entries_by_id = {entry["id"]: entry for entry in entries}
    ground_truth_by_id = {
        record["id"]: record["ground_truth"]
        for record in load_jsonl(args.ground_truth_path)
    }
    manifest_by_id = {
        record["id"]: record for record in manifest["entries"]
    }
    functions_by_hash = {}
    for entry in entries:
        functions_by_hash.setdefault(tools_hash(entry["function"]), entry["function"])

    selected_ids = manifest["subsets"][args.subset]
    if args.case_offset:
        selected_ids = selected_ids[args.case_offset :]
    if args.case_limit is not None:
        selected_ids = selected_ids[: args.case_limit]
    selected_manifest = [manifest_by_id[entry_id] for entry_id in selected_ids]
    shard_suffix = f"_{args.result_shard}" if args.result_shard else ""
    output_path = (
        args.output_dir / backend / f"{args.subset}{shard_suffix}.jsonl"
    )
    completed = existing_records(output_path)

    worker = start_worker(args, backend)
    try:
        preload_worker(
            worker,
            backend,
            selected_manifest,
            entries_by_id,
            functions_by_hash,
            args,
        )
        for index, manifest_record in enumerate(selected_manifest):
            entry = entries_by_id[manifest_record["id"]]
            alternate_functions = functions_by_hash[
                manifest_record["alternate_adapter"]
            ]
            possible_answer = ground_truth_by_id[entry["id"]]
            print(
                f"[{backend}] {index + 1}/{len(selected_manifest)} "
                f"{entry['id']}",
                flush=True,
            )

            reference_key = (
                backend,
                entry["id"],
                "model_generate_correct",
                None,
                "generate",
            )
            if (
                "model_generate" in args.controls
                and index < args.reference_parity_count
                and reference_key not in completed
            ):
                reference_record = run_reference_generate(
                    worker=worker,
                    backend=backend,
                    entry=entry,
                    manifest_record=manifest_record,
                    alternate_functions=alternate_functions,
                    possible_answer=possible_answer,
                    args=args,
                )
                append_jsonl(output_path, reference_record)
                completed[reference_key] = reference_record

            controls = [
                ("correct_only", "correct", "correct", None),
                ("base_only", None, None, None),
                ("wrong_only", "alternate", "alternate", None),
            ]
            for condition, symbolic_start, symbolic_end, switch_at in controls:
                control_name = condition.removesuffix("_only")
                if control_name not in args.controls:
                    continue
                key = (backend, entry["id"], condition, None, "preserve")
                if key in completed:
                    continue
                record = run_generation(
                    worker=worker,
                    backend=backend,
                    entry=entry,
                    manifest_record=manifest_record,
                    alternate_functions=alternate_functions,
                    possible_answer=possible_answer,
                    condition=condition,
                    boundary=None,
                    cache_policy="preserve",
                    start_adapter=resolve_state(symbolic_start, manifest_record),
                    end_adapter=resolve_state(symbolic_end, manifest_record),
                    switch_at=switch_at,
                    args=args,
                    return_first_logits=(
                        condition == "correct_only"
                        and index < args.logit_parity_count
                    ),
                )
                append_jsonl(output_path, record)
                completed[key] = record

            if args.prefill_replay:
                replay_condition = (
                    f"base_prefill_replay_{args.prefill_replay_tokens}_to_correct"
                )
                replay_key = (
                    backend,
                    entry["id"],
                    replay_condition,
                    None,
                    "preserve",
                )
                if replay_key not in completed:
                    replay_record = run_generation(
                        worker=worker,
                        backend=backend,
                        entry=entry,
                        manifest_record=manifest_record,
                        alternate_functions=alternate_functions,
                        possible_answer=possible_answer,
                        condition=replay_condition,
                        boundary=None,
                        cache_policy="preserve",
                        start_adapter=manifest_record["correct_adapter"],
                        end_adapter=manifest_record["correct_adapter"],
                        switch_at=None,
                        args=args,
                        prefill_adapter=None,
                        replay_last_prompt_token=args.prefill_replay_tokens == 1,
                        replay_prompt_tokens=args.prefill_replay_tokens,
                    )
                    append_jsonl(output_path, replay_record)
                    completed[replay_key] = replay_record

            correct_record = completed.get(
                (backend, entry["id"], "correct_only", None, "preserve")
            )
            if not correct_record or correct_record.get("status") != "ok":
                continue
            result = correct_record["result"]
            manifest_record["boundaries"][backend] = {
                **FIXED_BOUNDARIES,
                **result.get("semantic_boundaries", {}),
            }
            manifest_record["references"][backend] = {
                "condition": "correct_only",
                "token_ids": result.get("token_ids", []),
                "semantic_boundaries": result.get("semantic_boundaries", {}),
                "raw_output": correct_record.get("raw_output", ""),
            }
            if not args.no_manifest_updates:
                atomic_write_json(args.manifest_path, manifest)

            no_op_at = boundary_value("token_16", result)
            no_op_key = (
                backend,
                entry["id"],
                "correct_to_correct",
                "token_16",
                "preserve",
            )
            if "noop" in args.controls and no_op_key not in completed:
                no_op_record = run_generation(
                    worker=worker,
                    backend=backend,
                    entry=entry,
                    manifest_record=manifest_record,
                    alternate_functions=alternate_functions,
                    possible_answer=possible_answer,
                    condition="correct_to_correct",
                    boundary="token_16",
                    cache_policy="preserve",
                    start_adapter=manifest_record["correct_adapter"],
                    end_adapter=manifest_record["correct_adapter"],
                    switch_at=no_op_at,
                    args=args,
                )
                append_jsonl(output_path, no_op_record)
                completed[no_op_key] = no_op_record

            eos_at = boundary_value("eos", result)
            eos_key = (
                backend,
                entry["id"],
                "correct_to_wrong",
                "eos",
                "preserve",
            )
            if "eos" in args.controls and eos_key not in completed:
                eos_record = run_generation(
                    worker=worker,
                    backend=backend,
                    entry=entry,
                    manifest_record=manifest_record,
                    alternate_functions=alternate_functions,
                    possible_answer=possible_answer,
                    condition="correct_to_wrong",
                    boundary="eos",
                    cache_policy="preserve",
                    start_adapter=manifest_record["correct_adapter"],
                    end_adapter=manifest_record["alternate_adapter"],
                    switch_at=eos_at,
                    args=args,
                )
                append_jsonl(output_path, eos_record)
                completed[eos_key] = eos_record

            for direction, boundary, policy in args.schedules:
                symbolic_start, symbolic_end = DIRECTIONS[direction]
                switch_at = boundary_value(boundary, result)
                key = (
                    backend,
                    entry["id"],
                    direction,
                    boundary,
                    policy,
                )
                if key in completed:
                    continue
                if switch_at is None:
                    if args.missing_boundary_policy == "keep_start":
                        record = run_generation(
                            worker=worker,
                            backend=backend,
                            entry=entry,
                            manifest_record=manifest_record,
                            alternate_functions=alternate_functions,
                            possible_answer=possible_answer,
                            condition=direction,
                            boundary=boundary,
                            cache_policy=policy,
                            start_adapter=resolve_state(
                                symbolic_start, manifest_record
                            ),
                            end_adapter=resolve_state(
                                symbolic_end, manifest_record
                            ),
                            switch_at=None,
                            args=args,
                        )
                        record["boundary_fallback"] = True
                    else:
                        record = skipped_boundary_record(
                            backend=backend,
                            subset=args.subset,
                            entry_id=entry["id"],
                            direction=direction,
                            boundary=boundary,
                            policy=policy,
                        )
                else:
                    record = run_generation(
                        worker=worker,
                        backend=backend,
                        entry=entry,
                        manifest_record=manifest_record,
                        alternate_functions=alternate_functions,
                        possible_answer=possible_answer,
                        condition=direction,
                        boundary=boundary,
                        cache_policy=policy,
                        start_adapter=resolve_state(
                            symbolic_start, manifest_record
                        ),
                        end_adapter=resolve_state(
                            symbolic_end, manifest_record
                        ),
                        switch_at=switch_at,
                        args=args,
                    )
                append_jsonl(output_path, record)
                completed[key] = record
    finally:
        worker._stop()


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def paired_interval(differences: list[int]) -> tuple[float, float]:
    if not differences:
        return (math.nan, math.nan)
    rng = random.Random(42)
    means = []
    for _ in range(4000):
        sample = [rng.choice(differences) for _ in differences]
        means.append(sum(sample) / len(sample))
    means.sort()
    return (means[int(0.025 * len(means))], means[int(0.975 * len(means))])


def mcnemar_exact_p(treatment_wins: int, control_wins: int) -> float:
    discordant = treatment_wins + control_wins
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        math.comb(discordant, value)
        for value in range(min(treatment_wins, control_wins) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def analyze_results(output_dir: Path) -> dict:
    records = []
    for path in sorted(output_dir.glob("*/*.jsonl")):
        records.extend(
            record
            for record in load_jsonl(path)
            if record.get("record_type") == "generation"
        )

    boundary_missing = [
        record for record in records if record.get("status") == "boundary_missing"
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    by_case: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    for record in records:
        if record.get("status") == "boundary_missing":
            continue
        group_key = (
            record.get("backend"),
            record.get("subset"),
            record.get("condition"),
            record.get("boundary"),
            record.get("cache_policy"),
        )
        groups[group_key].append(record)
        by_case[(record.get("backend"), record.get("subset"), record.get("id"))][
            (
                record.get("condition"),
                record.get("boundary"),
                record.get("cache_policy"),
            )
        ] = record

    group_summaries = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        backend, subset, condition, boundary, policy = key
        ok = [row for row in rows if row.get("status") == "ok"]
        switch_latencies = [
            event["latency"]
            for row in ok
            for event in row.get("result", {}).get("activation_events", [])
            if event.get("kind") in {"switch", "post_prefill_switch"}
        ]
        switched_rows = [
            row
            for row in ok
            if (
                row.get("boundary") is not None
                or row.get("replay_last_prompt_token")
                or int(row.get("replay_prompt_tokens") or 0) > 0
            )
            and row.get("condition") not in {"correct_only", "base_only", "wrong_only"}
        ]
        peak_memory = [
            row.get("result", {}).get("peak_memory_bytes")
            for row in ok
            if row.get("result", {}).get("peak_memory_bytes") is not None
        ]
        logprob_jumps = []
        for row in switched_rows:
            result = row.get("result", {})
            switch_at = row.get("switch_at")
            logprobs = result.get("selected_logprobs", [])
            if (
                result.get("switch_applied")
                and isinstance(switch_at, int)
                and 0 < switch_at < len(logprobs)
            ):
                logprob_jumps.append(
                    abs(logprobs[switch_at] - logprobs[switch_at - 1])
                )
        error_types: dict[str, int] = defaultdict(int)
        for row in ok:
            error_type = row.get("checker", {}).get("error_type")
            if error_type:
                error_types[error_type] += 1
        group_summaries.append(
            {
                "backend": backend,
                "subset": subset,
                "condition": condition,
                "boundary": boundary,
                "cache_policy": policy,
                "total": len(rows),
                "completed": len(ok),
                "accuracy": (
                    sum(bool(row.get("valid")) for row in ok) / len(ok)
                    if ok
                    else None
                ),
                "malformed_rate": (
                    sum(bool(row.get("malformed_call")) for row in ok) / len(ok)
                    if ok
                    else None
                ),
                "adapter_leak_rate": (
                    sum(bool(row.get("adapter_leak")) for row in ok) / len(ok)
                    if ok
                    else None
                ),
                "mean_wall_latency": (
                    statistics.fmean(row["wall_latency"] for row in ok)
                    if ok
                    else None
                ),
                "mean_switch_latency": (
                    statistics.fmean(switch_latencies)
                    if switch_latencies
                    else None
                ),
                "switch_applied_rate": (
                    sum(
                        bool(row.get("result", {}).get("switch_applied"))
                        for row in switched_rows
                    )
                    / len(switched_rows)
                    if switched_rows
                    else None
                ),
                "mean_peak_memory_gb": (
                    statistics.fmean(peak_memory) / (1024**3)
                    if peak_memory
                    else None
                ),
                "mean_boundary_logprob_jump": (
                    statistics.fmean(logprob_jumps)
                    if logprob_jumps
                    else None
                ),
                "error_types": dict(sorted(error_types.items())),
                "errors": len(rows) - len(ok),
            }
        )

    parity = []
    no_op_parity = []
    token_zero_parity = []
    paired = []
    for (backend, subset, entry_id), case_records in by_case.items():
        correct = case_records.get(("correct_only", None, "preserve"))
        generated = case_records.get(("model_generate_correct", None, "generate"))
        no_op = case_records.get(
            ("correct_to_correct", "token_16", "preserve")
        )
        if correct and generated and correct.get("status") == generated.get("status") == "ok":
            left = correct["result"].get("token_ids", [])
            right = generated["result"].get("token_ids", [])
            parity.append(
                {
                    "backend": backend,
                    "subset": subset,
                    "id": entry_id,
                    "exact": left == right,
                    "first_divergence": first_divergence(left, right),
                }
            )
        if correct and no_op and correct.get("status") == no_op.get("status") == "ok":
            left = correct["result"].get("token_ids", [])
            right = no_op["result"].get("token_ids", [])
            no_op_parity.append(
                {
                    "backend": backend,
                    "subset": subset,
                    "id": entry_id,
                    "exact": left == right,
                    "first_divergence": first_divergence(left, right),
                }
            )
        expected_zero_controls = {
            "base_to_correct": correct,
            "wrong_to_correct": correct,
            "correct_to_base": case_records.get(("base_only", None, "preserve")),
            "correct_to_wrong": case_records.get(("wrong_only", None, "preserve")),
        }
        for condition, expected in expected_zero_controls.items():
            switched = case_records.get((condition, "token_0", "preserve"))
            if not (
                expected
                and switched
                and expected.get("status") == switched.get("status") == "ok"
            ):
                continue
            expected_ids = expected["result"].get("token_ids", [])
            switched_ids = switched["result"].get("token_ids", [])
            token_zero_parity.append(
                {
                    "backend": backend,
                    "subset": subset,
                    "id": entry_id,
                    "condition": condition,
                    "exact": expected_ids == switched_ids,
                    "first_divergence": first_divergence(
                        expected_ids, switched_ids
                    ),
                }
            )
        if not correct or correct.get("status") != "ok":
            continue
        for treatment_key, treatment in case_records.items():
            if treatment_key[0] in {
                "correct_only",
                "base_only",
                "wrong_only",
                "model_generate_correct",
                "correct_to_correct",
            }:
                continue
            if treatment.get("status") != "ok":
                continue
            paired.append(
                {
                    "backend": backend,
                    "subset": subset,
                    "condition": treatment_key[0],
                    "boundary": treatment_key[1],
                    "cache_policy": treatment_key[2],
                    "difference": int(bool(treatment.get("valid")))
                    - int(bool(correct.get("valid"))),
                    "first_divergence": first_divergence(
                        correct["result"].get("token_ids", []),
                        treatment["result"].get("token_ids", []),
                    ),
                    "switch_applied": bool(
                        treatment.get("result", {}).get("switch_applied")
                    ),
                }
            )

    paired_groups: dict[tuple, list[int]] = defaultdict(list)
    for row in paired:
        paired_groups[
            (
                row["backend"],
                row["subset"],
                row["condition"],
                row["boundary"],
                row["cache_policy"],
            )
        ].append(row["difference"])
    paired_summaries = []
    for key, differences in sorted(
        paired_groups.items(), key=lambda item: str(item[0])
    ):
        low, high = paired_interval(differences)
        paired_summaries.append(
            {
                "backend": key[0],
                "subset": key[1],
                "condition": key[2],
                "boundary": key[3],
                "cache_policy": key[4],
                "n": len(differences),
                "accuracy_delta": statistics.fmean(differences),
                "bootstrap_95_ci": [low, high],
                "treatment_wins": sum(value > 0 for value in differences),
                "control_wins": sum(value < 0 for value in differences),
                "mcnemar_exact_p": mcnemar_exact_p(
                    sum(value > 0 for value in differences),
                    sum(value < 0 for value in differences),
                ),
                "mean_first_divergence": (
                    statistics.fmean(
                        row["first_divergence"]
                        for row in paired
                        if (
                            row["backend"],
                            row["subset"],
                            row["condition"],
                            row["boundary"],
                            row["cache_policy"],
                        )
                        == key
                        and row["first_divergence"] is not None
                    )
                    if any(
                        (
                            row["backend"],
                            row["subset"],
                            row["condition"],
                            row["boundary"],
                            row["cache_policy"],
                        )
                        == key
                        and row["first_divergence"] is not None
                        for row in paired
                    )
                    else None
                ),
            }
        )

    applied_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in paired:
        if not row["switch_applied"]:
            continue
        applied_groups[
            (
                row["backend"],
                row["subset"],
                row["condition"],
                row["boundary"],
                row["cache_policy"],
            )
        ].append(row)
    applied_summaries = []
    for key, rows in sorted(applied_groups.items(), key=lambda item: str(item[0])):
        differences = [row["difference"] for row in rows]
        low, high = paired_interval(differences)
        divergences = [
            row["first_divergence"]
            for row in rows
            if row["first_divergence"] is not None
        ]
        treatment_wins = sum(value > 0 for value in differences)
        control_wins = sum(value < 0 for value in differences)
        applied_summaries.append(
            {
                "backend": key[0],
                "subset": key[1],
                "condition": key[2],
                "boundary": key[3],
                "cache_policy": key[4],
                "n": len(rows),
                "accuracy_delta": statistics.fmean(differences),
                "bootstrap_95_ci": [low, high],
                "mcnemar_exact_p": mcnemar_exact_p(
                    treatment_wins, control_wins
                ),
                "mean_first_divergence": (
                    statistics.fmean(divergences) if divergences else None
                ),
            }
        )

    cross_backend_logits = []
    cross_backend_tokens = []
    peft_correct = {
        (record.get("subset"), record["id"]): record
        for record in records
        if record.get("backend") == "peft"
        and record.get("condition") == "correct_only"
        and record.get("status") == "ok"
        and "first_logits" in record.get("result", {})
    }
    for record in records:
        if not (
            record.get("backend") == "raw"
            and record.get("condition") == "correct_only"
            and record.get("status") == "ok"
            and "first_logits" in record.get("result", {})
        ):
            continue
        peer = peft_correct.get((record.get("subset"), record["id"]))
        if peer is None:
            continue
        peer_tokens = peer["result"].get("token_ids", [])
        raw_tokens = record["result"].get("token_ids", [])
        cross_backend_tokens.append(
            {
                "subset": record.get("subset"),
                "id": record["id"],
                "exact": peer_tokens == raw_tokens,
                "first_divergence": first_divergence(peer_tokens, raw_tokens),
            }
        )
        left = peer["result"]["first_logits"]
        right = record["result"]["first_logits"]
        if len(left) != len(right):
            continue
        absolute = [abs(a - b) for a, b in zip(left, right)]
        cross_backend_logits.append(
            {
                "subset": record.get("subset"),
                "id": record["id"],
                "max_abs_logit_difference": max(absolute),
                "mean_abs_logit_difference": statistics.fmean(absolute),
                "top1_same": max(range(len(left)), key=left.__getitem__)
                == max(range(len(right)), key=right.__getitem__),
            }
        )

    summary = {
        "num_records": len(records),
        "groups": group_summaries,
        "model_generate_parity": {
            "n": len(parity),
            "exact": sum(row["exact"] for row in parity),
            "details": parity,
        },
        "no_op_parity": {
            "n": len(no_op_parity),
            "exact": sum(row["exact"] for row in no_op_parity),
            "details": no_op_parity,
        },
        "token_zero_parity": {
            "n": len(token_zero_parity),
            "exact": sum(row["exact"] for row in token_zero_parity),
            "details": token_zero_parity,
        },
        "paired_comparisons": paired_summaries,
        "paired_switch_applied_comparisons": applied_summaries,
        "boundary_missing": {
            "count": len(boundary_missing),
            "by_backend_subset_boundary": dict(
                sorted(
                    {
                        "|".join(
                            str(value)
                            for value in (
                                row.get("backend"),
                                row.get("subset"),
                                row.get("boundary"),
                            )
                        ): sum(
                            1
                            for candidate in boundary_missing
                            if (
                                candidate.get("backend"),
                                candidate.get("subset"),
                                candidate.get("boundary"),
                            )
                            == (
                                row.get("backend"),
                                row.get("subset"),
                                row.get("boundary"),
                            )
                        )
                        for row in boundary_missing
                    }.items()
                )
            ),
        },
        "cross_backend_token_parity": cross_backend_tokens,
        "cross_backend_first_logits": cross_backend_logits,
    }
    execution_errors = [
        record for record in records if record.get("status") == "error"
    ]
    full_candidates = [
        row
        for row in paired_summaries
        if row["subset"] == "full"
        and row["backend"] == "peft"
        and row["n"] >= 200
    ]
    best_full = (
        max(full_candidates, key=lambda row: row["accuracy_delta"])
        if full_candidates
        else None
    )
    def find_comparison(
        *, subset: str, condition: str, boundary: str, cache_policy: str
    ):
        return next(
            (
                row
                for row in paired_summaries
                if row["backend"] == "peft"
                and row["subset"] == subset
                and row["condition"] == condition
                and row["boundary"] == boundary
                and row["cache_policy"] == cache_policy
            ),
            None,
        )

    early_activation = find_comparison(
        subset="pilot",
        condition="base_to_correct",
        boundary="token_8",
        cache_policy="preserve",
    )
    phase_switch = find_comparison(
        subset="full",
        condition="correct_to_base",
        boundary="after_tool_call",
        cache_policy="preserve",
    )
    expert_handoff = find_comparison(
        subset="full",
        condition="wrong_to_correct",
        boundary="token_16",
        cache_policy="recompute",
    )
    switched_groups = [
        group
        for group in group_summaries
        if group["mean_switch_latency"] is not None
        and group["mean_wall_latency"] not in (None, 0)
    ]
    max_switch_fraction = max(
        (
            group["mean_switch_latency"] / group["mean_wall_latency"]
            for group in switched_groups
        ),
        default=None,
    )
    summary["decision"] = {
        "mechanically_feasible": (
            not execution_errors
            and summary["model_generate_parity"]["n"] > 0
            and summary["model_generate_parity"]["exact"]
            == summary["model_generate_parity"]["n"]
            and summary["no_op_parity"]["n"] > 0
            and summary["no_op_parity"]["exact"]
            == summary["no_op_parity"]["n"]
        ),
        "execution_error_count": len(execution_errors),
        "best_full_schedule": best_full,
        "behaviorally_viable_point_estimate": (
            best_full is not None and best_full["accuracy_delta"] >= -0.05
        ),
        "behaviorally_equivalent_95_ci": (
            best_full is not None
            and best_full["bootstrap_95_ci"][0] >= -0.05
        ),
        "max_switch_time_fraction": max_switch_fraction,
        "operationally_useful": (
            max_switch_fraction is not None and max_switch_fraction < 0.05
        ),
        "early_activation_signal": early_activation,
        "phase_switch_signal": phase_switch,
        "expert_handoff_signal": expert_handoff,
        "rapid_token_moe_supported": (
            early_activation is not None
            and early_activation["bootstrap_95_ci"][0] >= -0.05
        ),
        "phase_switching_supported": (
            phase_switch is not None
            and phase_switch["accuracy_delta"] >= -0.05
        ),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", summary)
    write_plots(output_dir, group_summaries)
    return summary


def percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def write_plots(output_dir: Path, groups: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    treatment_names = set(DIRECTIONS)
    available_subsets = {
        group["subset"]
        for group in groups
        if group["condition"] in treatment_names and group["subset"]
    }
    if not available_subsets:
        return
    subset = (
        "full"
        if "full" in available_subsets
        else "pilot"
        if "pilot" in available_subsets
        else "smoke"
    )
    selected = [
        group
        for group in groups
        if group["subset"] == subset
        and group["condition"] in treatment_names
        and group["completed"]
    ]
    if not selected:
        return

    labels = [
        f"{group['condition']}\n{group['boundary']}\n{group['cache_policy']}"
        for group in selected
    ]
    accuracies = [100 * group["accuracy"] for group in selected]
    figure_width = max(10, len(selected) * 0.7)
    fig, axis = plt.subplots(figsize=(figure_width, 6))
    axis.bar(range(len(selected)), accuracies)
    axis.set_ylabel("BFCL AST accuracy (%)")
    axis.set_title(f"Mid-decode switching accuracy ({subset})")
    axis.set_xticks(range(len(selected)), labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_boundary.png", dpi=160)
    plt.close(fig)

    jump_groups = [
        group
        for group in selected
        if group["mean_boundary_logprob_jump"] is not None
    ]
    if jump_groups:
        jump_labels = [
            f"{group['condition']}\n{group['boundary']}\n{group['cache_policy']}"
            for group in jump_groups
        ]
        jumps = [
            group["mean_boundary_logprob_jump"] for group in jump_groups
        ]
        fig, axis = plt.subplots(
            figsize=(max(10, len(jump_groups) * 0.7), 6)
        )
        axis.bar(range(len(jump_groups)), jumps)
        axis.set_ylabel("Absolute selected-token logprob jump")
        axis.set_title(f"Logprob discontinuity at switch ({subset})")
        axis.set_xticks(
            range(len(jump_groups)), jump_labels, rotation=45, ha="right"
        )
        fig.tight_layout()
        fig.savefig(output_dir / "logprob_jump_by_boundary.png", dpi=160)
        plt.close(fig)


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Mid-Decode D2L LoRA Switching Results",
        "",
        f"Total generation records: {summary['num_records']}",
        "",
        "## Mechanical gates",
        "",
        (
            "- Custom-loop vs `model.generate()` exact parity: "
            f"{summary['model_generate_parity']['exact']}/"
            f"{summary['model_generate_parity']['n']}"
        ),
        (
            "- Correct→correct no-op exact parity: "
            f"{summary['no_op_parity']['exact']}/"
            f"{summary['no_op_parity']['n']}"
        ),
        (
            "- Token-zero switches match their always-on destination: "
            f"{summary['token_zero_parity']['exact']}/"
            f"{summary['token_zero_parity']['n']}"
        ),
        (
            "- Execution errors: "
            f"{summary['decision']['execution_error_count']}"
        ),
        "",
        "## Accuracy and runtime",
        "",
        "| Backend | Subset | Condition | Boundary | Cache | N | Accuracy | "
        "Malformed | Switch applied | Mean latency | Switch latency |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        lines.append(
            "| {backend} | {subset} | {condition} | {boundary} | {cache} | "
            "{completed}/{total} | {accuracy} | {malformed} | {applied} | {latency} | "
            "{switch_latency} |".format(
                backend=group["backend"],
                subset=group["subset"],
                condition=group["condition"],
                boundary=group["boundary"] or "-",
                cache=group["cache_policy"],
                completed=group["completed"],
                total=group["total"],
                accuracy=percentage(group["accuracy"]),
                malformed=percentage(group["malformed_rate"]),
                applied=percentage(group["switch_applied_rate"]),
                latency=(
                    f"{group['mean_wall_latency']:.3f}s"
                    if group["mean_wall_latency"] is not None
                    else "N/A"
                ),
                switch_latency=(
                    f"{1000 * group['mean_switch_latency']:.2f}ms"
                    if group["mean_switch_latency"] is not None
                    else "N/A"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Paired deltas from always-correct LoRA",
            "",
            "| Backend | Subset | Condition | Boundary | Cache | N | Delta | "
            "95% CI | McNemar p | First divergence |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["paired_comparisons"]:
        lines.append(
            "| {backend} | {subset} | {condition} | {boundary} | {cache} | "
            "{n} | {delta} | [{low}, {high}] | {pvalue} | {divergence} |".format(
                backend=row["backend"],
                subset=row["subset"],
                condition=row["condition"],
                boundary=row["boundary"],
                cache=row["cache_policy"],
                n=row["n"],
                delta=percentage(row["accuracy_delta"]),
                low=percentage(row["bootstrap_95_ci"][0]),
                high=percentage(row["bootstrap_95_ci"][1]),
                pvalue=f"{row['mcnemar_exact_p']:.4f}",
                divergence=(
                    f"{row['mean_first_divergence']:.1f}"
                    if row["mean_first_divergence"] is not None
                    else "N/A"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Paired deltas where the switch actually fired",
            "",
            "| Backend | Subset | Condition | Boundary | Cache | N | Delta | "
            "95% CI | McNemar p |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["paired_switch_applied_comparisons"]:
        lines.append(
            "| {backend} | {subset} | {condition} | {boundary} | {cache} | "
            "{n} | {delta} | [{low}, {high}] | {pvalue} |".format(
                backend=row["backend"],
                subset=row["subset"],
                condition=row["condition"],
                boundary=row["boundary"],
                cache=row["cache_policy"],
                n=row["n"],
                delta=percentage(row["accuracy_delta"]),
                low=percentage(row["bootstrap_95_ci"][0]),
                high=percentage(row["bootstrap_95_ci"][1]),
                pvalue=f"{row['mcnemar_exact_p']:.4f}",
            )
        )
    decision = summary["decision"]
    best = decision["best_full_schedule"]
    lines.extend(
        [
            "",
            "## Decision gates",
            "",
            (
                "- Mechanically feasible: "
                + ("PASS" if decision["mechanically_feasible"] else "FAIL")
            ),
            (
                "- Operationally useful: "
                + ("PASS" if decision["operationally_useful"] else "NOT YET")
            ),
        ]
    )
    if best is None:
        lines.append("- Behavioral viability: pending a full-set result")
    else:
        lines.extend(
            [
                (
                    "- Best full schedule: "
                    f"{best['condition']} at {best['boundary']} "
                    f"({best['cache_policy']})"
                ),
                (
                    "- Behavioral 5-point margin, point estimate: "
                    + (
                        "PASS"
                        if decision["behaviorally_viable_point_estimate"]
                        else "FAIL"
                    )
                ),
                (
                    "- Behavioral 5-point margin, 95% CI: "
                    + (
                        "PASS"
                        if decision["behaviorally_equivalent_95_ci"]
                        else "INCONCLUSIVE"
                    )
                ),
            ]
        )
    lines.extend(["", "## Architecture recommendation", ""])
    if decision["phase_switch_signal"] is None:
        lines.append("- Awaiting the full phase-switch result.")
    elif decision["phase_switching_supported"]:
        lines.append(
            "- Use pre-decode expert selection, with optional switch-off or "
            "handoff only after the tool call has been semantically committed."
        )
    else:
        lines.append(
            "- Keep one adapter active for the complete decode; phase switching "
            "did not meet the 5-point margin."
        )
    if decision["rapid_token_moe_supported"]:
        lines.append(
            "- The tested early token-level transition met the practical margin."
        )
    else:
        lines.append(
            "- Do not treat the current D2L adapters as freely interchangeable "
            "token-level experts without joint switch-aware training."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("peft", "raw", "both"),
        default="both",
    )
    parser.add_argument(
        "--subset",
        choices=("smoke", "pilot", "full"),
        default="smoke",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "result_d2l/middecode_switch",
    )
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--d2l-root", type=Path, default=DEFAULT_D2L_ROOT)
    parser.add_argument("--d2l-python", type=Path, default=DEFAULT_D2L_PYTHON)
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=GROUND_TRUTH_PATH,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-count", type=int, default=10)
    parser.add_argument("--pilot-count", type=int, default=40)
    parser.add_argument(
        "--case-offset",
        type=int,
        default=0,
        help="Skip the first N cases from the frozen subset",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        help="Run only the first N cases from the frozen subset",
    )
    parser.add_argument(
        "--result-shard",
        help="Optional suffix for disjoint result shards",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--constrained",
        action="store_true",
        help="Apply the single-tool XGrammar processor during custom decoding",
    )
    parser.add_argument(
        "--prefill-replay",
        action="store_true",
        help=(
            "Prefill prompt[:-1] on base, activate the correct adapter, and "
            "replay the final prompt token before selecting output token zero"
        ),
    )
    parser.add_argument(
        "--prefill-replay-tokens",
        type=int,
        default=0,
        help="Number of final prompt tokens to replay under the correct adapter",
    )
    parser.add_argument(
        "--missing-boundary-policy",
        choices=("skip", "keep_start"),
        default="skip",
        help="Skip missing semantic markers or continue with the start adapter",
    )
    parser.add_argument("--peft-gpu", default="0")
    parser.add_argument("--raw-gpu", default="1")
    parser.add_argument("--reference-parity-count", type=int, default=10)
    parser.add_argument("--logit-parity-count", type=int, default=3)
    parser.add_argument(
        "--controls",
        default="model_generate,correct,base,wrong,noop,eos",
        help=(
            "Comma-separated controls: model_generate, correct, base, wrong, "
            "noop, eos. Correct is required when running schedules."
        ),
    )
    parser.add_argument(
        "--directions",
        default=",".join(DIRECTIONS),
        help="Comma-separated switch directions",
    )
    parser.add_argument(
        "--boundaries",
        default="token_8,token_16,token_32,after_think,after_tool_call",
        help="Comma-separated boundary names",
    )
    parser.add_argument(
        "--cache-policies",
        default="preserve,recompute",
        help="Comma-separated cache policies",
    )
    parser.add_argument(
        "--schedules",
        help=(
            "Optional comma-separated direction:boundary:cache triples. "
            "When set, overrides the cross-product options."
        ),
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Create/update the frozen manifest without loading models",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Analyze existing JSONL results without loading models",
    )
    parser.add_argument(
        "--no-manifest-updates",
        action="store_true",
        help="Do not persist per-case references while generations run",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Do not regenerate shared summary/report files after a run",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.manifest_path = args.output_dir / "manifest.json"
    args.directions = parse_csv_option(
        args.directions, set(DIRECTIONS), "directions"
    )
    args.boundaries = parse_csv_option(
        args.boundaries,
        set(FIXED_BOUNDARIES) | SEMANTIC_BOUNDARIES,
        "boundaries",
    )
    args.cache_policies = parse_csv_option(
        args.cache_policies,
        {"preserve", "recompute"},
        "cache policies",
    )
    args.controls = parse_csv_option(
        args.controls,
        {"model_generate", "correct", "base", "wrong", "noop", "eos"},
        "controls",
    )
    if args.schedules:
        schedules = []
        for value in args.schedules.split(","):
            parts = tuple(item.strip() for item in value.split(":"))
            if len(parts) != 3:
                raise ValueError(f"invalid schedule triple: {value}")
            direction, boundary, policy = parts
            if direction not in DIRECTIONS:
                raise ValueError(f"unknown schedule direction: {direction}")
            if boundary not in set(FIXED_BOUNDARIES) | SEMANTIC_BOUNDARIES:
                raise ValueError(f"unknown schedule boundary: {boundary}")
            if policy not in {"preserve", "recompute"}:
                raise ValueError(f"unknown schedule cache policy: {policy}")
            schedules.append((direction, boundary, policy))
        args.schedules = schedules
    else:
        args.schedules = [
            (direction, boundary, policy)
            for direction in args.directions
            for boundary in args.boundaries
            for policy in args.cache_policies
        ]
    if (
        not args.analyze_only
        and not args.manifest_only
        and args.schedules
        and "correct" not in args.controls
    ):
        raise ValueError("the correct control is required for switch schedules")
    if args.prefill_replay and args.prefill_replay_tokens == 0:
        args.prefill_replay_tokens = 1
    if args.prefill_replay_tokens < 0:
        raise ValueError("--prefill-replay-tokens must be non-negative")
    args.prefill_replay = args.prefill_replay_tokens > 0
    if args.prefill_replay and args.backend not in {"peft", "both"}:
        raise ValueError("prompt replay is currently implemented for PEFT only")
    return args


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest_path),
                "entries": manifest["num_entries"],
                "adapters": manifest["num_adapters"],
                "subset": args.subset,
                "subset_count": len(manifest["subsets"][args.subset]),
            },
            indent=2,
        )
    )
    if args.manifest_only:
        return 0
    if not args.analyze_only:
        backends = ("peft", "raw") if args.backend == "both" else (args.backend,)
        for backend in backends:
            run_backend(args, manifest, backend)
    if args.skip_analysis:
        return 0
    summary = analyze_results(args.output_dir)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "report": str(args.output_dir / "report.md"),
                "records": summary["num_records"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
