#!/bin/bash
# Calibrate ConQuR rotations.
#   MODE=fp     -> "Ours f.p. calib."   (no quantization during calibration)
#   MODE=quant  -> "Ours quant. calib." (4-bit activations, KV_BITS-bit KV cache during calibration)
set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <MODEL> <MODE: fp|quant> <KV_BITS: 4|16> <BATCH_SIZE> <OUTPUT_PATH>"
    exit 1
fi

MODEL=$1
MODE=$2
KV_BITS=$3
BATCH_SIZE=$4
OUTPUT_PATH=$5

case "${MODE}" in
    fp)    ACT_BITS=16; CLIP_RATIO=1.0; CALIB_KV_BITS=16 ;;
    quant) ACT_BITS=4;  CLIP_RATIO=0.9; CALIB_KV_BITS=${KV_BITS} ;;
    *) echo "Unknown MODE: ${MODE}" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

python "${ROOT}/conqur/calibrate.py" \
    --model "${MODEL}" \
    --nsamples 128 \
    --seqlen 2048 \
    --seed 0 \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" \
    --act_bits "${ACT_BITS}" \
    --clip_ratio "${CLIP_RATIO}" \
    --k_bits "${CALIB_KV_BITS}" \
    --k_clip_ratio 1.0 \
    --v_bits "${CALIB_KV_BITS}" \
    --v_clip_ratio 1.0
