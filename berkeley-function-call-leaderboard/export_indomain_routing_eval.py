"""Export the held-out in-domain routing split as BFCL routing categories.

The router scores ~0.5 route_at_1 on its own Nemotron/Toucan validation split
but only 0.10-0.21 on the BFCL anon-10 gate, even though the gate is the easier
measure (ten candidates rather than the whole vocabulary). Two explanations
survive that: the schemas differ, or the alias ordering differs. Running the
*same* candidate-scoring harness over the in-domain split separates them.

Two categories are written from the same rows:
  ``_sorted``   aliases in tool_a..tool_j order, exactly as training presents them
  ``_shuffled`` same catalogues with the list order permuted, as BFCL presents them

Comparing the two isolates alias order; comparing either against the BFCL gate
isolates the schema distribution.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path(__file__).parent / "bfcl_eval" / "data"
DEFAULT_SOURCE = (
    "/home/jah649/tool-lora/doc-to-lora/data/raw_datasets/"
    "schema_router_distill_v1/b_joint_late_schema/validation/ds.parquet"
)


def _flatten(tool: dict) -> dict:
    """OpenAI-style {"type","function"} to the flat shape BFCL entries use."""
    function = tool["function"]
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {}),
    }


def _write(category: str, rows: list[dict]) -> None:
    (DATA / f"BFCL_v4_{category}.json").write_text(
        "".join(json.dumps(row["entry"]) + "\n" for row in rows)
    )
    (DATA / "possible_answer" / f"BFCL_v4_{category}.json").write_text(
        "".join(json.dumps(row["answer"]) + "\n" for row in rows)
    )
    print(f"wrote {category}: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--category", default="indomain_val_10_anon")
    parser.add_argument(
        "--variant",
        choices=("anon", "named"),
        default="anon",
        help="Anonymised aliases, or the rows that keep real tool names.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    table = pq.read_table(args.source)
    columns = {
        name: table.column(name).to_pylist()
        for name in ("context", "messages", "selected_real_name", "augmentation")
    }

    rng = random.Random(args.seed)
    sorted_rows: list[dict] = []
    shuffled_rows: list[dict] = []
    # Catalogue order exactly as the training row stores it. Both other
    # orderings re-key the catalogue, which changes the text handed to the
    # hypernetwork; this one is the only variant that can be compared directly
    # against a metric computed during training.
    source_rows: list[dict] = []

    for index, augmentation in enumerate(columns["augmentation"]):
        if ("anon" in augmentation) != (args.variant == "anon"):
            continue
        tools = [_flatten(tool) for tool in json.loads(columns["context"][index])]
        messages = columns["messages"][index]
        if isinstance(messages, str):
            messages = json.loads(messages)
        gold = columns["selected_real_name"][index]
        if gold not in {tool["name"] for tool in tools}:
            continue

        query = messages[1]["content"]
        entry_id = f"{args.category}_{len(sorted_rows)}"
        answer = {"id": entry_id, "ground_truth": [{gold: {}}], "_ground_truth_func": gold}

        ordered = sorted(tools, key=lambda tool: tool["name"])
        permuted = list(ordered)
        rng.shuffle(permuted)

        base = {"id": entry_id, "question": [[{"role": "user", "content": query}]]}
        sorted_rows.append(
            {"entry": {**base, "function": ordered}, "answer": answer}
        )
        shuffled_rows.append(
            {"entry": {**base, "function": permuted}, "answer": answer}
        )
        source_rows.append({"entry": {**base, "function": tools}, "answer": answer})
        if args.limit and len(sorted_rows) >= args.limit:
            break

    _write(f"{args.category}_sorted", sorted_rows)
    _write(f"{args.category}_shuffled", shuffled_rows)
    _write(f"{args.category}_source", source_rows)


if __name__ == "__main__":
    main()
