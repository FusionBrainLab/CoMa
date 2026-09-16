SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export REPO_ROOT

HYDRA_FULL_ERROR=1 python "$REPO_ROOT/hydra_run.py" \
--config_path "$REPO_ROOT/experiments/benchmarks/coma/results_visualization/visualization_configs/crossmodal_bar_visualization.yaml" \
paths.results="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/reproduced_results/lower_results" \
paths.image="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/visualizations/lower_origin/crossmodal_bar.png"

HYDRA_FULL_ERROR=1 python "$REPO_ROOT/hydra_run.py" \
--config_path "$REPO_ROOT/experiments/benchmarks/coma/results_visualization/visualization_configs/crossmodal_line_grid_visualization.yaml" \
paths.results="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/reproduced_results/lower_results" \
paths.image="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/visualizations/lower_origin/crossmodal_line_grid.png"

HYDRA_FULL_ERROR=1 python "$REPO_ROOT/hydra_run.py" \
--config_path "$REPO_ROOT/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line_grid_visualization.yaml" \
paths.results="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/reproduced_results/lower_results" \
paths.image="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/visualizations/lower_origin/multimodal_line_grid.png"

HYDRA_FULL_ERROR=1 python "$REPO_ROOT/hydra_run.py" \
--config_path "$REPO_ROOT/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_contour_grid_visualization.yaml" \
paths.results="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/reproduced_results/lower_results" \
paths.image="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/visualizations/lower_origin/multimodal_contour_grid.png"

HYDRA_FULL_ERROR=1 python "$REPO_ROOT/hydra_run.py" \
--config_path "$REPO_ROOT/experiments/benchmarks/coma/results_visualization/visualization_configs/crossmodal_bar_visualization.yaml" \
paths.results="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/reproduced_results/upper_results" \
paths.image="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/visualizations/upper_origin/crossmodal_bar.png"

HYDRA_FULL_ERROR=1 python "$REPO_ROOT/hydra_run.py" \
--config_path "$REPO_ROOT/experiments/benchmarks/coma/results_visualization/visualization_configs/crossmodal_line_grid_visualization.yaml" \
paths.results="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/reproduced_results/upper_results" \
paths.image="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/visualizations/upper_origin/crossmodal_line_grid.png"

HYDRA_FULL_ERROR=1 python "$REPO_ROOT/hydra_run.py" \
--config_path "$REPO_ROOT/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_line_grid_visualization.yaml" \
paths.results="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/reproduced_results/upper_results" \
paths.image="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/visualizations/upper_origin/multimodal_line_grid.png"

HYDRA_FULL_ERROR=1 python "$REPO_ROOT/hydra_run.py" \
--config_path "$REPO_ROOT/experiments/benchmarks/coma/results_visualization/visualization_configs/multimodal_contour_grid_visualization.yaml" \
paths.results="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/reproduced_results/upper_results" \
paths.image="$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/visualizations/upper_origin/multimodal_contour_grid.png"
