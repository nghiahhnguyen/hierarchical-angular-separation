#!/usr/bin/env bash
set -euo pipefail
IMAGENET_ROOT=${IMAGENET_ROOT:?Set IMAGENET_ROOT to the directory containing train/}
DEVICE=${DEVICE:-cuda}
# Features are reused across Figure 2 and Table 1. Extract one model at a time.
for model in random_vit openai datacomp1b raw_pixels; do
    python -m has.real_data extract --imagenet-root "$IMAGENET_ROOT" --model "$model" --device "$DEVICE"
done
for seed in 1 2; do
    python -m has.real_data extract --imagenet-root "$IMAGENET_ROOT" --model random_vit \
        --seed "$seed" --skip-text --device "$DEVICE"
done
python -m has.real_data figure2
python -m has.real_data table1
