"""Re-emit an anonymised routing benchmark with the aliases in sorted order.

The router's training data lists anonymised catalogues as tool_a, tool_b, ...
tool_j in order, so the alias is perfectly predictable from a tool's position
and the model never has to learn which name belongs to which schema. The
benchmark shuffles them. Sorting the benchmark the same way as training isolates
how much of the routing score depends on that ordering: only the presentation
order changes, the schemas, the query and the gold answer are untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).parent / "bfcl_eval" / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="multiple_scale_hard_10_anon")
    parser.add_argument("--suffix", default="_sorted")
    args = parser.parse_args()

    source = DATA / f"BFCL_v4_{args.category}.json"
    target = DATA / f"BFCL_v4_{args.category}{args.suffix}.json"

    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    for row in rows:
        row["function"] = sorted(row["function"], key=lambda f: f["name"])
        row["id"] = row["id"].replace(args.category, f"{args.category}{args.suffix}")
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))

    answers = DATA / "possible_answer" / f"BFCL_v4_{args.category}.json"
    if answers.exists():
        answer_rows = [
            json.loads(line)
            for line in answers.read_text().splitlines()
            if line.strip()
        ]
        for row in answer_rows:
            row["id"] = row["id"].replace(args.category, f"{args.category}{args.suffix}")
        (answers.parent / f"BFCL_v4_{args.category}{args.suffix}.json").write_text(
            "".join(json.dumps(row) + "\n" for row in answer_rows)
        )

    print(f"wrote {target} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
