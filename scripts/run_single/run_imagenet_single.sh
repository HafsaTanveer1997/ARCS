#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DATA_PATH=/path/to/imagenet CKPT_DIR=/path/to/checkpoints \
#   bash scripts/run_single/run_imagenet_single.sh METHOD ARCH W A SEED NUM_SAMPLES
#
# METHOD: random | confgap | selectq | arcs

METHOD=${1:-random}
ARCH=${2:-resnet18}
W=${3:-4}
A=${4:-4}
SEED=${5:-1005}
NUM_SAMPLES=${6:-1024}

DATA_PATH=${DATA_PATH:-/path/to/imagenet}
CKPT_DIR=${CKPT_DIR:-pretrained}
WORKERS=${WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-64}
SELECTOR_POOL_SIZE=${SELECTOR_POOL_SIZE:-4096}
TS_VAL_SIZE=${TS_VAL_SIZE:-4096}

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export CKPT_DIR

if [[ "$DATA_PATH" == "/path/to/imagenet" ]]; then
    echo "ERROR: DATA_PATH is not set."
    exit 2
fi

if [[ ! -d "$DATA_PATH/train" || ! -d "$DATA_PATH/val" ]]; then
    echo "ERROR: ImageNet directory must contain train/ and val/ folders."
    echo "Current DATA_PATH: $DATA_PATH"
    exit 2
fi

case "$ARCH" in
    resnet18|resnet50)
        FEATURE_MODULE="avgpool"
        ;;
    mobilenetv2)
        FEATURE_MODULE="features"
        ;;
    regnetx_600m)
        FEATURE_MODULE="head.avg_pool"
        ;;
    deit_small_patch16_224)
        FEATURE_MODULE="norm"
        ;;
    *)
        echo "Unsupported paper architecture: $ARCH"
        exit 1
        ;;
esac

LOGDIR="logs/imagenet/${ARCH}/w${W}a${A}/seed${SEED}"
mkdir -p "$LOGDIR"

COMMON_ARGS=(
    --arch "$ARCH"
    --n_bits_w "$W"
    --n_bits_a "$A"
    --num_samples "$NUM_SAMPLES"
    --selector_pool_size "$SELECTOR_POOL_SIZE"
    --selector_feature_module "$FEATURE_MODULE"
    --run_temp_scaling
    --ts_val_size "$TS_VAL_SIZE"
    --conf_threshold 0.70
    --seed "$SEED"
    --data_path "$DATA_PATH"
    --workers "$WORKERS"
    --batch_size "$BATCH_SIZE"
    --skip_cat
)

# The reported DeiT-S experiments use calibration-only PTQ.
if [[ "$ARCH" == "deit_small_patch16_224" ]]; then
    COMMON_ARGS+=(
        --vit_calibration_only
        --proxy_calib_size 128
        --act_calib_batch_size 32
    )
fi

case "$METHOD" in
    random|confgap)
        CMD=(
            python -u run_imagenet_baselines.py
            --calib_selection "$METHOD"
            "${COMMON_ARGS[@]}"
        )
        ;;
    selectq)
        CMD=(
            python -u run_imagenet_baselines.py
            --calib_selection selectq
            --selectq_pool_size "$SELECTOR_POOL_SIZE"
            "${COMMON_ARGS[@]}"
        )
        ;;
    arcs)
        CMD=(
            python -u run_imagenet_arcs.py
            "${COMMON_ARGS[@]}"
            --adaptive_auto_refs
            --adaptive_ref_percentile 90
            --adaptive_low_rep_ratio 0.75
            --adaptive_low_band_ratio 0.05
            --adaptive_random_ratio 0.20
            --adaptive_high_rep_ratio 0.68
            --adaptive_high_band_ratio 0.12
            --adaptive_high_random_ratio 0.20
            --adaptive_cont_conf_weight 0.5
            --adaptive_cont_kl_weight 0.5
            --adaptive_cont_flip_weight 0.0
            --adaptive_continuous_severity
            --adaptive_band_range_mode continuous
            --adaptive_low_sev_low_pct 75
            --adaptive_low_sev_high_pct 95
            --adaptive_high_sev_low_pct 70
            --adaptive_high_sev_high_pct 90
            --adaptive_ignore_flip_severity
            --adaptive_score_scope remaining
            --adaptive_band_mode score_only
            --adaptive_reliability_mode top
            --adaptive_class_cap 2
        )
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Allowed methods: random, confgap, selectq, arcs"
        exit 1
        ;;
esac

LOG="$LOGDIR/${METHOD}_${ARCH}_w${W}a${A}_n${NUM_SAMPLES}_seed${SEED}.log"

echo "============================================================"
echo "Method:      $METHOD"
echo "Arch:        $ARCH"
echo "Bits:        W${W}A${A}"
echo "Seed:        $SEED"
echo "Samples:     $NUM_SAMPLES"
echo "Pool:        $SELECTOR_POOL_SIZE"
echo "Feature:     $FEATURE_MODULE"
echo "Log:         $LOG"
echo "============================================================"

printf '%q ' "${CMD[@]}" | tee "$LOGDIR/last_command.txt"
printf '\n' | tee -a "$LOGDIR/last_command.txt"
"${CMD[@]}" 2>&1 | tee "$LOG"
