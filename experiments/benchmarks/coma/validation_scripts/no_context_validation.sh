SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export REPO_ROOT

HYDRA_FULL_ERROR=1 python "$REPO_ROOT/hydra_run.py" \
--config_path "$REPO_ROOT/experiments/benchmarks/coma/validation_config.yaml" \
--multirun \
model_size=2,4,8 \
train_type=no_context \
context_1d_count=0 \
context_2d_count=0 \
context_3d_count=0