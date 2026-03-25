#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="doc-to-lora/qwen3-4b"
TEST_CATEGORY="multiple"
RESTRICT_TOOLGEN=1

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <checkpoint_path>

Run BFCL generate + evaluate for a doc-to-lora checkpoint.

The result and score directories are derived automatically from the checkpoint path.
  e.g. .../runs/toucan_nemotron_test1/checkpoint-1200/pytorch_model.bin
       -> result_d2l/toucan_nemotron_test1_1200/
       -> score_d2l/toucan_nemotron_test1_1200/

Arguments:
  checkpoint_path   Path to the pytorch_model.bin checkpoint

Options:
  -m, --model NAME          Model name (default: $MODEL)
  -t, --test-category CAT   Test category (default: $TEST_CATEGORY)
  --no-restrict-toolgen     Disable D2L_RESTRICT_TOOLGEN
  --generate-only           Only run generate, skip evaluate
  --evaluate-only           Only run evaluate, skip generate
  -h, --help                Show this help message
EOF
    exit 0
}

GENERATE=1
EVALUATE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        -t|--test-category) TEST_CATEGORY="$2"; shift 2 ;;
        --no-restrict-toolgen) RESTRICT_TOOLGEN=0; shift ;;
        --generate-only) GENERATE=1; EVALUATE=0; shift ;;
        --evaluate-only) GENERATE=0; EVALUATE=1; shift ;;
        -h|--help) usage ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) CHECKPOINT_PATH="$1"; shift ;;
    esac
done

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    echo "Error: checkpoint_path is required" >&2
    usage
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

# Derive run name and checkpoint number from path
# Expected: .../runs/<run_name>/checkpoint-<num>/pytorch_model.bin
CKPT_DIR="$(dirname "$CHECKPOINT_PATH")"
CKPT_NUM="$(basename "$CKPT_DIR" | sed 's/checkpoint-//')"
RUN_NAME="$(basename "$(dirname "$CKPT_DIR")")"
DIR_NAME="${RUN_NAME}_${CKPT_NUM}"

RESULT_DIR="${SCRIPT_DIR}/result_d2l/${DIR_NAME}/"
SCORE_DIR="${SCRIPT_DIR}/score_d2l/${DIR_NAME}/"

echo "=== BFCL Run ==="
echo "  Checkpoint:     $CHECKPOINT_PATH"
echo "  Run name:       $RUN_NAME"
echo "  Checkpoint num: $CKPT_NUM"
echo "  Model:          $MODEL"
echo "  Test category:  $TEST_CATEGORY"
echo "  Result dir:     $RESULT_DIR"
echo "  Score dir:      $SCORE_DIR"
echo ""

ENV_VARS="D2L_CHECKPOINT_PATH=${CHECKPOINT_PATH}"
if [[ "$RESTRICT_TOOLGEN" -eq 1 ]]; then
    ENV_VARS="${ENV_VARS} D2L_RESTRICT_TOOLGEN=1"
fi

if [[ "$GENERATE" -eq 1 ]]; then
    echo ">>> Running generate..."
    conda run -n BFCL --no-capture-output bash -c \
        "${ENV_VARS} bfcl generate --model ${MODEL} --test-category ${TEST_CATEGORY} --result-dir ${RESULT_DIR}"
    echo ">>> Generate complete."
    echo ""
fi

if [[ "$EVALUATE" -eq 1 ]]; then
    echo ">>> Running evaluate..."
    conda run -n BFCL --no-capture-output bash -c \
        "bfcl evaluate --model ${MODEL} --test-category ${TEST_CATEGORY} --result-dir ${RESULT_DIR} --score-dir ${SCORE_DIR}"
    echo ">>> Evaluate complete."
fi
