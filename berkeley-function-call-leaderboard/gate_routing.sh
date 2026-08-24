#!/usr/bin/env bash
# Gate one or more D2L checkpoints on a routing-only category and score by name.
#
# Usage: gate_routing.sh CATEGORY GPU NAME=CHECKPOINT [NAME=CHECKPOINT ...]
set -uo pipefail

CATEGORY="$1"; shift
GPU="$1"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# A checkpoint trained with per-tool chunking must be evaluated the same way, so
# these mirror the training config and default to the unchunked behaviour.
CHUNK_ARGS=()
if [[ -n "${GATE_CTX_CHUNK_MODE:-}" ]]; then
    CHUNK_ARGS+=(--ctx-chunk-mode "$GATE_CTX_CHUNK_MODE")
    CHUNK_ARGS+=(--chunk-scaling "${GATE_CHUNK_SCALING:-none}")
    CHUNK_ARGS+=(--tools-per-chunk "${GATE_TOOLS_PER_CHUNK:-1}")
fi

for pair in "$@"; do
    name="${pair%%=*}"
    checkpoint="${pair#*=}"
    run="${name}_${CATEGORY}"
    if [[ ! -f "$checkpoint" ]]; then
        echo "$name: MISSING $checkpoint"
        continue
    fi
    bash run_bfcl.sh --staged-b --router-score-candidates --routing-only \
        --main-gpu "$GPU" "${CHUNK_ARGS[@]}" \
        --d2l-python /home/jah649/tool-lora/doc-to-lora/.venv/bin/python \
        --d2l-source-path /home/jah649/tool-lora/doc-to-lora/src \
        --num-threads 1 -t "$CATEGORY" -n "$run" "$checkpoint" \
        > "/tmp/gate_${run}.log" 2>&1
    # BFCL derives the output filename from the base category, dropping suffixes
    # like _anon, so locate whatever it actually wrote.
    result="$(find "result_d2l/${run}" -name '*_result.json' 2>/dev/null | head -1)"
    if [[ -n "$result" ]]; then
        python3 score_routing_only.py --category "$CATEGORY" "$result"
    else
        echo "$name: no result written, see /tmp/gate_${run}.log"
        tail -3 "/tmp/gate_${run}.log"
    fi
done
