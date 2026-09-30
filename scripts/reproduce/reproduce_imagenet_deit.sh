#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-/path/to/imagenet}
CONFIRM_RUN=${CONFIRM_RUN:-no}

ARCH=deit_small_patch16_224
METHODS=(random confgap selectq arcs)
BITWIDTHS=("4 4" "4 8" "8 4" "8 8")
BUDGETS=(128 256 512 1024)
SEEDS=(1005 1006 1007)

if [[ "$DATA_PATH" == "/path/to/imagenet" ]]; then
    echo "ERROR: DATA_PATH is not set."
    exit 2
fi

if [[ ! -d "$DATA_PATH/train" || ! -d "$DATA_PATH/val" ]]; then
    echo "ERROR: ImageNet directory must contain train/ and val/ folders."
    exit 2
fi

echo "DeiT-S reproduction plan"
echo "Methods: ${METHODS[*]}"
echo "Bit-widths: W4A4 W4A8 W8A4 W8A8"
echo "Budgets: ${BUDGETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: 192"

for SEED in "${SEEDS[@]}"; do
    for BW in "${BITWIDTHS[@]}"; do
        read -r W A <<< "$BW"
        for BUDGET in "${BUDGETS[@]}"; do
            for METHOD in "${METHODS[@]}"; do
                CMD=(
                    bash scripts/run_single/run_imagenet_single.sh
                    "$METHOD" "$ARCH" "$W" "$A" "$SEED" "$BUDGET"
                )
                echo "Planned: DATA_PATH=$DATA_PATH ${CMD[*]}"
                if [[ "$CONFIRM_RUN" == "yes" ]]; then
                    DATA_PATH="$DATA_PATH" "${CMD[@]}"
                fi
            done
        done
    done
done

if [[ "$CONFIRM_RUN" != "yes" ]]; then
    echo "Dry run only. Set CONFIRM_RUN=yes to launch the grid."
fi
