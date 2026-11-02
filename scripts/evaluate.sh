#!/bin/bash
# Evaluate W4A4KV{KV_BITS} with a given rotation file (PPL on WikiText-2/PTB/C4 + 9 zero-shot tasks).
# Any extra arguments are forwarded to conqur/evaluate.py (e.g. --no-lm_eval for PPL only).
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <MODEL> <KV_BITS: 4|16> <ROTATION_PATH> <OUTPUT_DIR> [extra evaluate.py args]"
    exit 1
fi

MODEL=$1
KV_BITS=$2
ROTATION=$3
OUTPUT_DIR=$4
shift 4

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${ROOT}/conqur/evaluate.py" \
    --model "${MODEL}" \
    --rotation "${ROTATION}" \
    --output_dir "${OUTPUT_DIR}" \
    --w_bits 4 \
    --a_bits 4 \
    --kv_bits "${KV_BITS}" \
    "$@"
