#!/bin/bash
# Calibrate (if needed) and evaluate one ConQuR configuration.
#
#   bash scripts/run.sh <llama2-7b|llama2-13b|llama3.2-3b|llama3-8b|llama3-70b> [--kv4] [--quant_calib] [extra evaluate.py args]
#
#   --kv4          evaluate W4A4KV4 instead of W4A4KV16
#   --quant_calib  calibrate with 4-bit activations (and 4-bit KV cache when --kv4 is set)
#
# An existing rotation under rotations/<model>/ is reused instead of recalibrating.
# Calibration uses the first visible GPU; evaluation spreads the model over all visible GPUs.
set -euo pipefail

usage() {
    echo "Usage: $0 <llama2-7b|llama2-13b|llama3.2-3b|llama3-8b|llama3-70b> [--kv4] [--quant_calib] [extra evaluate.py args]"
    exit 1
}
[ "$#" -ge 1 ] || usage

KEY=$1
shift

KV_BITS=16
QUANT_CALIB=0
EXTRA_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --kv4)         KV_BITS=4 ;;
        --quant_calib) QUANT_CALIB=1 ;;
        *)             EXTRA_ARGS+=("${arg}") ;;
    esac
done

# Calibration batch size: 2 for Llama-2 7B and Llama-3 8B, 1 otherwise (paper, Appendix D).
case "${KEY}" in
    llama2-7b)   MODEL=meta-llama/Llama-2-7b-hf;    BATCH_SIZE=2 ;;
    llama2-13b)  MODEL=meta-llama/Llama-2-13b-hf;   BATCH_SIZE=1 ;;
    llama3.2-3b) MODEL=meta-llama/Llama-3.2-3B;     BATCH_SIZE=1 ;;
    llama3-8b)   MODEL=meta-llama/Meta-Llama-3-8B;  BATCH_SIZE=2 ;;
    llama3-70b)  MODEL=meta-llama/Meta-Llama-3-70B; BATCH_SIZE=1 ;;
    *) echo "Unknown model key: ${KEY}" >&2; usage ;;
esac

if [ "${QUANT_CALIB}" -eq 1 ]; then
    # 4-bit activations during calibration; the KV cache matches the evaluation setting.
    CALIB_ACT_BITS=4; CALIB_CLIP_RATIO=0.9; CALIB_KV_BITS=${KV_BITS}
    ROT_NAME=quant_calib_a4kv${KV_BITS}
    RUN_NAME=quant_calib_w4a4kv${KV_BITS}
else
    # No quantization during calibration; one rotation serves both KV settings.
    CALIB_ACT_BITS=16; CALIB_CLIP_RATIO=1.0; CALIB_KV_BITS=16
    ROT_NAME=fp_calib
    RUN_NAME=fp_calib_w4a4kv${KV_BITS}
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROTATION="${ROOT}/rotations/${KEY}/${ROT_NAME}.pt"
OUTPUT_DIR="${ROOT}/results/${KEY}/${RUN_NAME}"
mkdir -p "$(dirname "${ROTATION}")"

if [ -f "${ROTATION}" ]; then
    echo "Using existing rotation: ${ROTATION}"
else
    echo "Calibrating: ${ROTATION}"
    CALIB_GPU="${CUDA_VISIBLE_DEVICES:-0}"
    CUDA_VISIBLE_DEVICES="${CALIB_GPU%%,*}" python "${ROOT}/conqur/calibrate.py" \
        --model "${MODEL}" \
        --nsamples 128 \
        --seqlen 2048 \
        --seed 0 \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${ROTATION}" \
        --act_bits "${CALIB_ACT_BITS}" \
        --clip_ratio "${CALIB_CLIP_RATIO}" \
        --k_bits "${CALIB_KV_BITS}" \
        --k_clip_ratio 1.0 \
        --v_bits "${CALIB_KV_BITS}" \
        --v_clip_ratio 1.0 \
        2>&1 | tee "${ROTATION%.pt}.log"
fi

echo "Evaluating: ${OUTPUT_DIR}"
python "${ROOT}/conqur/evaluate.py" \
    --model "${MODEL}" \
    --rotation "${ROTATION}" \
    --output_dir "${OUTPUT_DIR}" \
    --w_bits 4 \
    --a_bits 4 \
    --kv_bits "${KV_BITS}" \
    "${EXTRA_ARGS[@]}"
