#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
export REPO_ROOT
ROOT="${MASSING_METRIC_ROOT:-$REPO_ROOT}"
RESULTS=${ROOT}/experiments/benchmarks/coma/results_no_orientation
OUT=${ROOT}/experiments/benchmarks/coma/results_visualization/visualizations_publication_no_orientation

COMMON=(
  env
  HYDRA_FULL_ERROR=1
  python "${ROOT}/hydra_run.py"
)

"${COMMON[@]}" \
  --config_path "${ROOT}/experiments/benchmarks/coma/results_visualization/visualization_configs/publication/multimodal_contour_grid_visualization.yaml" \
  method.functions.0.function.results_folder="${RESULTS}" \
  method.functions.1.function.path="${OUT}/multimodal_contour_grid" \
  'method.functions.1.function.formats=[pdf,png]'

"${COMMON[@]}" \
  --config_path "${ROOT}/experiments/benchmarks/coma/results_visualization/visualization_configs/publication/multimodal_line_grid_visualization.yaml" \
  method.functions.0.function.base_creator.results_folder="${RESULTS}" \
  method.functions.1.function.path="${OUT}/multimodal_line_grid" \
  'method.functions.1.function.formats=[pdf,png]'

"${COMMON[@]}" \
  --config_path "${ROOT}/experiments/benchmarks/coma/results_visualization/visualization_configs/publication/crossmodal_line_grid_visualization.yaml" \
  method.functions.0.function.base_creator.results_folder="${RESULTS}" \
  method.functions.1.function.path="${OUT}/crossmodal_line_grid" \
  'method.functions.1.function.formats=[pdf,png]'

"${COMMON[@]}" \
  --config_path "${ROOT}/experiments/benchmarks/coma/results_visualization/visualization_configs/publication/crossmodal_bar_visualization.yaml" \
  method.functions.0.function.base_creator.results_folder="${RESULTS}" \
  method.functions.1.function.path="${OUT}/crossmodal_bar" \
  'method.functions.1.function.formats=[pdf,png]'
