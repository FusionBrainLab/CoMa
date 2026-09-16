SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
export REPO_ROOT
export MASSING_METRIC_ROOT="${MASSING_METRIC_ROOT:-$REPO_ROOT}"

HYDRA_FULL_ERROR=1 python "$MASSING_METRIC_ROOT/hydra_run.py" \
--config_path "$MASSING_METRIC_ROOT/experiments/benchmarks/coma/results_visualization/visualization_configs/publication/multimodal_line_grid_visualization.yaml"
