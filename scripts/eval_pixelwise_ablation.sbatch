#!/bin/bash
#SBATCH --job-name=eval_pixelwise_ablation
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/ourdisk/hpc/ai2es/tman/logs/eval_pixelwise_ablation_%j_out.txt
#SBATCH --error=/ourdisk/hpc/ai2es/tman/logs/eval_pixelwise_ablation_%j_err.txt

# Post-hoc evaluate.py + performance-diagrams for the pixelwise ablation checkpoint
# (ablation_terrain_pressure_levels_surface_vars_full_pixelwise, W&B run n1whwkgl,
# git commit ee5449c) — it was killed after 45 epochs with no post-hoc stats ever
# computed, so its obs_rel_freq_pointwise reliability curve (does nbs_include_pixel
# raise the pointwise ceiling relative to per_front_metrics_3d and model_1702?) has
# never actually been generated. Submit from the repo root on branch
# chore/eval-pixelwise-ablation:
#
#     sbatch scripts/eval_pixelwise_ablation.sbatch
#
# Prerequisites: branch checked out on schooner, pixi -e schooner solved, and
# /ourdisk/hpc/ai2es/tman/models/ablation_terrain_pressure_levels_surface_vars_full_pixelwise/_best_loss.keras
# present (written by ModelCheckpoint during the original training run).
#
# Resubmit-safe: skips straight to plotting if stats_derived.nc already exists.

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="configs/eval_pixelwise_ablation.yaml"
STATS_DIR="/ourdisk/hpc/ai2es/tman/models/stats/ablation_terrain_pressure_levels_surface_vars_full_pixelwise"
cd "$REPO_DIR"

run() {
    echo "==== $(date -u +%FT%TZ) :: $*"
    pixi run -e schooner "$@"
}

if [ -f "$STATS_DIR/stats_derived.nc" ]; then
    echo "==== skipping evaluate.py (found $STATS_DIR/stats_derived.nc)"
else
    run python -m fronts.evaluate --config_path "$CONFIG"
fi

run python -m fronts.plot.plot performance-diagrams --config_path "$CONFIG"

echo "==== $(date -u +%FT%TZ) :: done — stats + performance_{CF,WF,SF,OF,DL}.png under $STATS_DIR"
