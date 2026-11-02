#!/bin/bash
# Reproduce the "Ours f.p. calib." and "Ours quant. calib." rows of Table 1 for one model.
#
#   bash scripts/reproduce_table1.sh <llama2-7b|llama2-13b|llama3.2-3b|llama3-8b|llama3-70b> [extra evaluate.py args]
#
# Runs 3 calibrations and 4 evaluations:
#   f.p. calib.   : one rotation, evaluated at W4A4KV16 and W4A4KV4
#   quant. calib. : one rotation calibrated with A4/KV16 (evaluated at W4A4KV16)
#                   one rotation calibrated with A4/KV4  (evaluated at W4A4KV4)
# Rotations that already exist under rotations/<model>/ are reused, so pre-computed rotations can be dropped in.
# Calibration uses the first visible GPU; evaluation spreads the model over all visible GPUs.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <llama2-7b|llama2-13b|llama3.2-3b|llama3-8b|llama3-70b> [extra evaluate.py args]"
    exit 1
fi

KEY=$1
shift

# Calibration batch size: 2 for Llama-2 7B and Llama-3 8B, 1 otherwise (paper, Appendix D).
case "${KEY}" in
    llama2-7b)   MODEL=meta-llama/Llama-2-7b-hf;   BATCH_SIZE=2 ;;
    llama2-13b)  MODEL=meta-llama/Llama-2-13b-hf;  BATCH_SIZE=1 ;;
    llama3.2-3b) MODEL=meta-llama/Llama-3.2-3B;    BATCH_SIZE=1 ;;
    llama3-8b)   MODEL=meta-llama/Meta-Llama-3-8B; BATCH_SIZE=2 ;;
    llama3-70b)  MODEL=meta-llama/Meta-Llama-3-70B; BATCH_SIZE=1 ;;
    *) echo "Unknown model key: ${KEY}" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROT_DIR="${ROOT}/rotations/${KEY}"
RES_DIR="${ROOT}/results/${KEY}"
mkdir -p "${ROT_DIR}" "${RES_DIR}"

CALIB_GPU="${CUDA_VISIBLE_DEVICES:-0}"
CALIB_GPU="${CALIB_GPU%%,*}"

calibrate() {  # <MODE> <KV_BITS> <OUTPUT_PATH>
    if [ -f "$3" ]; then
        echo "Using existing rotation: $3"
    else
        echo "Calibrating: $3"
        CUDA_VISIBLE_DEVICES="${CALIB_GPU}" bash "${ROOT}/scripts/calibrate.sh" "${MODEL}" "$1" "$2" "${BATCH_SIZE}" "$3" \
            2>&1 | tee "${3%.pt}.log"
    fi
}

evaluate() {  # <KV_BITS> <ROTATION_PATH> <NAME>
    echo "Evaluating: ${RES_DIR}/$3"
    bash "${ROOT}/scripts/evaluate.sh" "${MODEL}" "$1" "$2" "${RES_DIR}/$3" "${EXTRA_ARGS[@]}"
}

EXTRA_ARGS=("$@")

calibrate fp    16 "${ROT_DIR}/fp_calib.pt"
calibrate quant 16 "${ROT_DIR}/quant_calib_a4kv16.pt"
calibrate quant 4  "${ROT_DIR}/quant_calib_a4kv4.pt"

evaluate 16 "${ROT_DIR}/fp_calib.pt"            fp_calib_w4a4kv16
evaluate 4  "${ROT_DIR}/fp_calib.pt"            fp_calib_w4a4kv4
evaluate 16 "${ROT_DIR}/quant_calib_a4kv16.pt"  quant_calib_w4a4kv16
evaluate 4  "${ROT_DIR}/quant_calib_a4kv4.pt"   quant_calib_w4a4kv4

python "${ROOT}/scripts/summarize.py" "${ROOT}/results"
