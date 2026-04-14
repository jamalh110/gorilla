#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="doc-to-lora/qwen3-4b"
TEST_CATEGORIES=()
RESTRICT_TOOLGEN=1
TOOL_NAMES_IN_SYSTEM=0
TOOL_SYMBOLS_IN_SYSTEM=0
SKIP_INTERNALIZE=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <checkpoint_path>

Run BFCL generate + evaluate for a doc-to-lora checkpoint.

The result and score directories are derived automatically from the checkpoint path,
or you can provide a custom name with -n/--name.
  e.g. .../runs/toucan_nemotron_test1/checkpoint-1200/pytorch_model.bin
       -> result_d2l/toucan_nemotron_test1_1200/
       -> score_d2l/toucan_nemotron_test1_1200/
  With --name my_experiment:
       -> result_d2l/my_experiment/
       -> score_d2l/my_experiment/

Arguments:
  checkpoint_path   Path to the pytorch_model.bin checkpoint

Options:
  -m, --model NAME          Model name (default: $MODEL)
  -n, --name NAME           Custom name for result/score directories
                            (default: derived from checkpoint path)
  -t, --test-category CAT   Test category; repeatable and comma-separable
                            (default: multiple)
                            Examples: -t live_simple
                                      -t multiple,live_simple
                                      -t multiple -t live_simple
  --no-restrict-toolgen     Disable D2L_RESTRICT_TOOLGEN
  --tool-names-in-system    Include tool names in the system message
  --tool-symbols-in-system  Include tool names, param names, and enums in system message
  --skip-internalize        Skip LoRA internalization (base model only, for ablation)
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
        -n|--name) CUSTOM_NAME="$2"; shift 2 ;;
        -t|--test-category) IFS=',' read -ra _cats <<< "$2"; TEST_CATEGORIES+=("${_cats[@]}"); shift 2 ;;
        --no-restrict-toolgen) RESTRICT_TOOLGEN=0; shift ;;
        --tool-names-in-system) TOOL_NAMES_IN_SYSTEM=1; shift ;;
        --tool-symbols-in-system) TOOL_SYMBOLS_IN_SYSTEM=1; shift ;;
        --skip-internalize) SKIP_INTERNALIZE=1; shift ;;
        --generate-only) GENERATE=1; EVALUATE=0; shift ;;
        --evaluate-only) GENERATE=0; EVALUATE=1; shift ;;
        -h|--help) usage ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) CHECKPOINT_PATH="$1"; shift ;;
    esac
done

if [[ ${#TEST_CATEGORIES[@]} -eq 0 ]]; then
    TEST_CATEGORIES=("multiple")
fi

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    echo "Error: checkpoint_path is required" >&2
    usage
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

if [[ -n "${CUSTOM_NAME:-}" ]]; then
    DIR_NAME="$CUSTOM_NAME"
else
    # Derive from checkpoint path: .../runs/<run_name>/checkpoint-<num>/pytorch_model.bin
    CKPT_DIR="$(dirname "$CHECKPOINT_PATH")"
    CKPT_NUM="$(basename "$CKPT_DIR" | sed 's/checkpoint-//')"
    RUN_NAME="$(basename "$(dirname "$CKPT_DIR")")"
    DIR_NAME="${RUN_NAME}_${CKPT_NUM}"
fi

RESULT_DIR="${SCRIPT_DIR}/result_d2l/${DIR_NAME}/"
SCORE_DIR="${SCRIPT_DIR}/score_d2l/${DIR_NAME}/"

# Detect available GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c 'GPU ')
if [[ "$NUM_GPUS" -lt 1 ]]; then
    NUM_GPUS=1
fi

echo "=== BFCL Run ==="
echo "  Checkpoint:     $CHECKPOINT_PATH"
echo "  Output name:    $DIR_NAME"
echo "  Model:          $MODEL"
echo "  Test categories: ${TEST_CATEGORIES[*]}"
echo "  GPUs:           $NUM_GPUS"
echo "  Result dir:     $RESULT_DIR"
echo "  Score dir:      $SCORE_DIR"
echo ""

RAW_LOG_DIR="${RESULT_DIR}/raw"
mkdir -p "${RAW_LOG_DIR}"

ENV_VARS="D2L_CHECKPOINT_PATH=${CHECKPOINT_PATH}"
if [[ "$RESTRICT_TOOLGEN" -eq 1 ]]; then
    ENV_VARS="${ENV_VARS} D2L_RESTRICT_TOOLGEN=1"
fi
if [[ "$TOOL_NAMES_IN_SYSTEM" -eq 1 ]]; then
    ENV_VARS="${ENV_VARS} D2L_TOOL_NAMES_IN_SYSTEM=1"
fi
if [[ "$TOOL_SYMBOLS_IN_SYSTEM" -eq 1 ]]; then
    ENV_VARS="${ENV_VARS} D2L_TOOL_SYMBOLS_IN_SYSTEM=1"
fi
if [[ "$SKIP_INTERNALIZE" -eq 1 ]]; then
    ENV_VARS="${ENV_VARS} D2L_SKIP_INTERNALIZE=1"
fi

for cat in "${TEST_CATEGORIES[@]}"; do
    if [[ "$GENERATE" -eq 1 ]]; then
        echo ">>> Running generate for '${cat}' (${NUM_GPUS} GPU(s), ${NUM_GPUS} threads)..."
        conda run -n BFCL --no-capture-output bash -c \
            "${ENV_VARS} D2L_RAW_LOG=${RAW_LOG_DIR}/${cat}.jsonl bfcl generate --model ${MODEL} --test-category ${cat} --num-threads ${NUM_GPUS} --result-dir ${RESULT_DIR}"
        echo ">>> Generate complete for '${cat}'."
        echo ""
    fi

    if [[ "$EVALUATE" -eq 1 ]]; then
        echo ">>> Running evaluate for '${cat}'..."
        conda run -n BFCL --no-capture-output bash -c \
            "bfcl evaluate --model ${MODEL} --test-category ${cat} --result-dir ${RESULT_DIR} --score-dir ${SCORE_DIR}"
        echo ">>> Evaluate complete for '${cat}'."
    fi
done
