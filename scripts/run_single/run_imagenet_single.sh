#!/usr/bin/env bash
set -euo pipefail

# Usage:
# DATA_PATH=/path/to/imagenet CKPT_DIR=/path/to/checkpoints \
# bash scripts/run_single/run_imagenet_single.sh METHOD ARCH W A SEED BUDGET

METHOD=${1:-arcs}
ARCH=${2:-deit_small_patch16_224}
W=${3:-8}
A=${4:-4}
SEED=${5:-1005}
BUDGET=${6:-128}

DATA_PATH=${DATA_PATH:-/path/to/imagenet}
CKPT_DIR=${CKPT_DIR:-pretrained}
WORKERS=${WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-64}

if [[ "$DATA_PATH" == "/path/to/imagenet" ]]; then
    echo "Set DATA_PATH to the ImageNet directory."
    exit 1
fi

python -u run_imagenet.py \
    --method "$METHOD" \
    --arch "$ARCH" \
    --data-path "$DATA_PATH" \
    --checkpoint-dir "$CKPT_DIR" \
    --w-bits "$W" \
    --a-bits "$A" \
    --budget "$BUDGET" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS"
