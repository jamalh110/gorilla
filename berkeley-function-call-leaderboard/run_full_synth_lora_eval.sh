#!/usr/bin/env bash
# Full BFCL live_simple eval for synth tool-call LoRAs (no tools in context).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ADAPTER_DIR="${ADAPTER_DIR:-/home/jah649/tool-lora/doc-to-lora/train_outputs/live_simple_synth_toolcall_r8_down}"
NAME="${NAME:-synth_toolcall_r8_down_noconstrained}"
RESULT_DIR="${ROOT}/result_d2l/${NAME}"
SCORE_DIR="${ROOT}/score_d2l/${NAME}"
RAW_LOG="${RESULT_DIR}/raw/live_simple.jsonl"
GPU="${GPU:-0}"

mkdir -p "${RESULT_DIR}/raw" "${SCORE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export D2L_ADAPTER_DIR="${ADAPTER_DIR}"
export D2L_BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export D2L_ROOT="/home/jah649/tool-lora/doc-to-lora"
export D2L_PYTHON="${D2L_ROOT}/.venv/bin/python"
export D2L_MAX_NEW_TOKENS="${D2L_MAX_NEW_TOKENS:-1024}"
export D2L_RAW_LOG="${RAW_LOG}"

echo "=== Full synth LoRA BFCL eval ==="
echo "  adapters: ${ADAPTER_DIR}"
echo "  result:   ${RESULT_DIR}"
echo "  score:    ${SCORE_DIR}"
echo "  gpu:      ${GPU}"
date

cd "${ROOT}"
conda run -n BFCL --no-capture-output bfcl generate \
  --model doc-to-lora/qwen3-4b-peft \
  --test-category live_simple \
  --temperature 0 \
  --num-threads 1 \
  --result-dir "${RESULT_DIR}"

conda run -n BFCL --no-capture-output bfcl evaluate \
  --model doc-to-lora/qwen3-4b-peft \
  --test-category live_simple \
  --result-dir "${RESULT_DIR}" \
  --score-dir "${SCORE_DIR}"

echo "=== Done ==="
date
SCORE_FILE="${SCORE_DIR}/doc-to-lora_qwen3-4b-peft/live/BFCL_v4_live_simple_score.json"
if [[ -f "${SCORE_FILE}" ]]; then
  head -n 1 "${SCORE_FILE}"
fi
