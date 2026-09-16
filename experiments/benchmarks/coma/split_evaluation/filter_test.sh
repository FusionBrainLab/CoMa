SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export REPO_ROOT

PYTHONPATH="$REPO_ROOT" \
python "$REPO_ROOT/experiments/benchmarks/coma/split_evaluation/filter_test.py"
