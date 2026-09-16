SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export REPO_ROOT

HYDRA_FULL_ERROR=1 python "$REPO_ROOT/hydra_run.py" \
--config_path "$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/massing_similarity_heatmap_visualization.yaml" \
"$@"
