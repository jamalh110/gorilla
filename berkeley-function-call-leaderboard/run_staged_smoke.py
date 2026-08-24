#!/usr/bin/env python3
"""Run a deterministic 10-case staged BFCL ``multiple`` smoke evaluation."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.model_handler.local_inference.doc_to_lora_staged import (
    DocToLoraMetaIntentBinderHandler,
    DocToLoraMetaSelectBindHandler,
    binder_prompt_contains_original_query,
    parse_exactly_one_tool_call,
)
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry


def select_seeded_multiple_cases(
    entries: list[dict], seed: int = 42, count: int = 10
) -> list[dict]:
    """Select fixed cases while cycling through 2/3/4-tool buckets."""
    buckets = {tool_count: [] for tool_count in (2, 3, 4)}
    for entry in entries:
        tool_count = len(entry.get("function", []))
        if tool_count in buckets:
            buckets[tool_count].append(entry)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    if any(not bucket for bucket in buckets.values()):
        raise RuntimeError("multiple smoke set must contain 2-, 3-, and 4-tool cases")

    selected = []
    indices = {tool_count: 0 for tool_count in buckets}
    while len(selected) < count:
        made_progress = False
        for tool_count in (2, 3, 4):
            index = indices[tool_count]
            if index < len(buckets[tool_count]) and len(selected) < count:
                selected.append(deepcopy(buckets[tool_count][index]))
                indices[tool_count] += 1
                made_progress = True
        if not made_progress:
            break
    if len(selected) != count:
        raise RuntimeError(f"could select only {len(selected)} of {count} cases")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("a", "b"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--binder-checkpoint")
    parser.add_argument("--main-gpu")
    parser.add_argument("--binder-gpu")
    parser.add_argument("--binder-python")
    parser.add_argument("--d2l-python")
    parser.add_argument("--d2l-source-path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--oracle-route", action="store_true")
    parser.add_argument("--skip-internalize", action="store_true")
    parser.add_argument("--stateful-continuation", action="store_true")
    parser.add_argument("--same-meta-prompts", action="store_true")
    parser.add_argument("--plain-schema-bind", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("staged_smoke"))
    args = parser.parse_args()
    if args.variant == "a" and not args.binder_checkpoint:
        parser.error("--binder-checkpoint is required for Variant A")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_log = args.output_dir / f"variant_{args.variant}_raw.jsonl"
    handler_class = (
        DocToLoraMetaIntentBinderHandler
        if args.variant == "a"
        else DocToLoraMetaSelectBindHandler
    )
    registry_name = (
        "doc-to-lora/qwen3-4b-meta-intent-0.6b"
        if args.variant == "a"
        else "doc-to-lora/qwen3-4b-meta-select-bind"
    )
    kwargs = {
        "checkpoint_path": args.checkpoint,
        "binder_checkpoint_path": args.binder_checkpoint,
        "raw_log_path": str(raw_log),
        "oracle_route": args.oracle_route,
        "skip_internalize": args.skip_internalize,
        "stateful_continuation": args.stateful_continuation,
        "plain_schema_bind": args.plain_schema_bind,
        "use_baseline_prompts": (
            False
            if (
                args.oracle_route
                or args.same_meta_prompts
                or args.plain_schema_bind
            )
            else args.skip_internalize
        ),
    }
    for name in (
        "main_gpu",
        "binder_gpu",
        "binder_python",
        "d2l_python",
        "d2l_source_path",
    ):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    handler = handler_class(
        model_name=registry_name,
        temperature=0,
        registry_name=registry_name,
        is_fc_model=False,
        **kwargs,
    )

    prompts = load_dataset_entry(
        "multiple",
        include_prereq=False,
        include_language_specific_hint=False,
    )
    ground_truth = {
        item["id"]: item["ground_truth"]
        for item in load_ground_truth_entry("multiple")
    }
    selected = select_seeded_multiple_cases(prompts, args.seed, args.count)
    handler.prepare(selected)
    records = []
    failures = []
    for entry in selected:
        record = {
            "id": entry["id"],
            "num_tools": len(entry["function"]),
            "valid": False,
        }
        try:
            output, metadata = handler.inference(
                deepcopy(entry), include_input_log=False, exclude_state_log=True
            )
            call = parse_exactly_one_tool_call(output)
            decoded = handler.decode_ast(
                output, language=ReturnFormat.PYTHON, has_tool_call_tag=False
            )
            checker = ast_checker(
                entry["function"],
                decoded,
                ground_truth[entry["id"]],
                Language.PYTHON,
                "multiple",
                registry_name,
            )
            trace = handler.last_trace
            assert trace["stage1"].get("parsed_call")
            assert trace.get("selected_schema")
            assert trace["stage2"].get("parsed_call")
            assert not trace["validation_errors"]
            for stage_name in ("stage1", "stage2"):
                attempts = trace[stage_name].get("attempts", [])
                assert attempts
                if args.variant == "b":
                    assert len(attempts) == 1
                    assert attempts[0].get("constraint_mode") == "none"
                else:
                    assert attempts[0].get("constraint_mode") == "xgrammar"
                    assert all(
                        attempt.get("constraint_mode") in {"xgrammar", "lexical"}
                        for attempt in attempts
                    )
            if args.variant == "a":
                assert not binder_prompt_contains_original_query(
                    trace["stage2"]["messages"], entry["question"][0]
                )
                assert trace["stage2"]["enable_thinking"] is False
            elif args.plain_schema_bind:
                stage2_messages = trace["stage2"]["messages"]
                assert [message["role"] for message in stage2_messages] == [
                    "system",
                    "user",
                    "user",
                ]
                assert "<tool_response>" not in stage2_messages[-1]["content"]
                assert trace["stage1"]["raw_output"] not in json.dumps(
                    stage2_messages
                )
                original_user_content = next(
                    message["content"]
                    for message in entry["question"][0]
                    if message.get("role") == "user"
                )
                assert (
                    sum(
                        message.get("content") == original_user_content
                        for message in stage2_messages
                    )
                    == 1
                )
            record.update(
                {
                    "valid": checker["valid"],
                    "checker": checker,
                    "final_call": call,
                    "input_token_count": metadata["input_token_count"],
                    "output_token_count": metadata["output_token_count"],
                    "latency": metadata["latency"],
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(entry["id"])
        records.append(record)

    score_path = args.output_dir / f"variant_{args.variant}_score.jsonl"
    header = {
        "seed": args.seed,
        "total_count": len(records),
        "correct_count": sum(record["valid"] for record in records),
        "accuracy": sum(record["valid"] for record in records) / len(records),
        "sample_ids": [record["id"] for record in records],
    }
    with score_path.open("w") as output:
        output.write(json.dumps(header) + "\n")
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"score_path": str(score_path), **header}, indent=2))
    handler.shutdown()
    if failures:
        raise RuntimeError(f"staged smoke failed for: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
