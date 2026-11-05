#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Full runs should use a batch allocation.
DEVICE=${DEVICE:-cuda}
RUN_ROOT=${RUN_ROOT:-results/synthetic}
mkdir -p "$RUN_ROOT/trace" "$RUN_ROOT/sweep"
COMMON=(--depths 5 --branching_factors 3 --loss_types contrastive_strict
        --temperatures 1 --learnable_temperatures 0 --d_over_v 1 --lrs 1
        --optimizers lbfgs --strict_sigma 1e-5 --strict_p_over_d 1
        --strict_theta_mode geometric_decay --strict_theta_values 85
        --strict_save_artifacts 1 --strict_identity_encoder 1 --strict_normalize 0
        --loss_direction both --images_per_leaf 150 --seed 42 --device "$DEVICE"
        --use_mlflow 1 --save_checkpoints 1)
MLFLOW_TRACKING_URI="sqlite:///$PWD/$RUN_ROOT/trace/mlflow.db" \
python synthetic/run_experiment.py sweep "${COMMON[@]}" \
    --caption_modes uniform --lambda_regs 1e-3 --epochs 50 --checkpoint_interval 1 \
    --output_dir "$RUN_ROOT/trace"
MLFLOW_TRACKING_URI="sqlite:///$PWD/$RUN_ROOT/sweep/mlflow.db" \
python synthetic/run_experiment.py sweep "${COMMON[@]}" \
    --caption_modes balanced --prototype_shift_norms 1e-4 \
    --lambda_regs 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1 3 10 30 100 \
    --epochs 500 --checkpoint_interval 20 --output_dir "$RUN_ROOT/sweep"
python synthetic/export_results.py --run-root "$RUN_ROOT"
python - "$RUN_ROOT" <<'PLOT'
import sys
from pathlib import Path
from has.plotting import plot_figure5
root = Path(sys.argv[1])
plot_figure5(root/'measurements', root/'plots')
PLOT
