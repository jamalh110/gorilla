"""Score routing-only runs by tool name.

BFCL's own scorer needs a fully bound call, so it reports 0.0 for runs that
stop after the router. This compares the selected function name against the
possible-answer key instead.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
ROOT = Path(__file__).resolve().parent


def gold_names(category: str) -> dict[str, str]:
    path = ROOT / "bfcl_eval" / "data" / "possible_answer" / f"BFCL_v4_{category}.json"
    names = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            answer = entry["ground_truth"]
            if isinstance(answer, list):
                answer = answer[0]
            names[entry["id"]] = next(iter(answer))
    return names


def selected_name(result: object) -> str | None:
    if isinstance(result, list):
        result = result[0] if result else ""
    match = NAME_RE.search(str(result))
    return match.group(1) if match else None


def score(result_path: Path, category: str) -> dict[str, object]:
    gold = gold_names(category)
    correct = 0
    total = 0
    unparsed = 0
    misses = []
    with result_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            expected = gold.get(entry["id"])
            if expected is None:
                continue
            total += 1
            chosen = selected_name(entry.get("result"))
            if chosen is None:
                unparsed += 1
            elif chosen == expected:
                correct += 1
                continue
            misses.append((entry["id"], chosen, expected))
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "unparsed": unparsed,
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_paths", nargs="+", type=Path)
    parser.add_argument("--category", default="multiple_scale_hard_10_smoke")
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    for path in args.result_paths:
        summary = score(path, args.category)
        misses = summary.pop("misses")
        label = path.parent.parent.parent.name if path.is_file() else path.name
        print(f"{label}: {summary['correct']}/{summary['total']} " f"({summary['accuracy']:.0%}), unparsed={summary['unparsed']}")
        if args.show_misses:
            for entry_id, chosen, expected in misses:
                print(f"  {entry_id}: chose {chosen!r}, wanted {expected!r}")


if __name__ == "__main__":
    main()
