#!/usr/bin/env python3
"""Drop ungrounded optional arguments from Qwen FC result JSONL/JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _query_text(prompt_entry: dict) -> str:
    chunks: list[str] = []
    for turn in prompt_entry.get("question") or []:
        for msg in turn:
            if msg.get("role") == "user":
                chunks.append(str(msg.get("content") or ""))
    return "\n".join(chunks)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _value_grounded(value, query_norm: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        # Keep true only if query suggests affirmation for that slot is hard;
        # conservatively keep booleans (often schema flags tied to wording).
        return True
    if isinstance(value, (int, float)):
        return str(value) in query_norm or str(int(value)) in query_norm
    if isinstance(value, list):
        if not value:
            return False
        return all(_value_grounded(v, query_norm) for v in value)
    if isinstance(value, dict):
        return all(_value_grounded(v, query_norm) for v in value.values())
    s = str(value).strip()
    if not s:
        return False
    sn = _normalize(s)
    if sn in query_norm:
        return True
    # token overlap for short multi-word values
    toks = [t for t in re.split(r"[^a-z0-9]+", sn) if len(t) >= 3]
    if not toks:
        return sn in query_norm
    hit = sum(1 for t in toks if t in query_norm)
    return hit >= max(1, (len(toks) + 1) // 2)


def strip_tool_call_text(result_text: str, function: dict, query: str) -> str:
    params = function.get("parameters") or {}
    required = set(params.get("required") or [])
    props = params.get("properties") or {}
    query_norm = _normalize(query)

    pattern = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)

    def repl(match: re.Match) -> str:
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)
        args = call.get("arguments")
        if not isinstance(args, dict):
            return match.group(0)
        new_args = {}
        for k, v in args.items():
            if k in required or k not in props:
                new_args[k] = v
                continue
            if _value_grounded(v, query_norm):
                new_args[k] = v
        call["arguments"] = new_args
        return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"

    return pattern.sub(repl, result_text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    prompts = {
        json.loads(l)["id"]: json.loads(l)
        for l in open(args.prompts)
        if l.strip()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_changed = 0
    with open(args.out, "w") as out_f:
        for line in open(args.result):
            if not line.strip():
                continue
            row = json.loads(line)
            entry = prompts[row["id"]]
            func = entry["function"][0]
            query = _query_text(entry)
            old = row.get("result") or ""
            new = strip_tool_call_text(old, func, query)
            if new != old:
                n_changed += 1
                row["result"] = new
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} changed={n_changed}")


if __name__ == "__main__":
    main()
