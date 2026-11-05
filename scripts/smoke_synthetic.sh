#!/usr/bin/env bash
set -euo pipefail
# Small test of the complete synthetic training and artifact-writing path.
RUN_ROOT=${RUN_ROOT:-results/smoke-synthetic}
mkdir -p "$RUN_ROOT"
MLFLOW_TRACKING_URI="sqlite:///$PWD/$RUN_ROOT/mlflow.db" \
python synthetic/run_experiment.py sweep \
    --depths 2 --branching_factors 2 --embedding_dim 8 --images_per_leaf 6 \
    --caption_modes balanced --loss_types contrastive_strict --temperatures 1 \
    --lambda_regs 0.01 --optimizers lbfgs --lrs 1 --epochs 2 --checkpoint_interval 1 \
    --strict_sigma 1e-5 --strict_theta_mode geometric_decay --strict_theta_values 85 \
    --strict_identity_encoder 1 --strict_normalize 0 --prototype_shift_norms 1e-4 \
    --device cpu --use_mlflow 1 --save_checkpoints 1 --output_dir "$RUN_ROOT"
