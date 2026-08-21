# metrics_validation

Evaluates contextual-relevance metrics as binary classifiers on the CoMa
negative-sampling set, and learns an ensemble metric (LogReg / HistGBDT / CatBoost)
on top of them. Standard repo flow: a hydra YAML config instantiates a top-level
`Function` via `_target_`; all functional code lives under `src/`.

A learned metric is **already trained** and ships in `models/`
(`learned_ensemble_catboost_all17.joblib`, CatBoost on all 17 features, GroupKFold-5
**AUC 0.915 ± 0.003 / F1 0.848 ± 0.006**), so it is ready to use as a validator.

`results/`, `models/`, and `figures/` are versioned with DVC (`results.dvc`,
`models.dvc`, `figures.dvc`). Run `dvc pull` to fetch them; `dvc push` after
regenerating.

## Layout

```
feature_pool/coma.yaml              shared 17-feature pool (single source of truth, -> ${_feature_pool})
ensemble_validation_config.yaml     the study:     score -> analysis -> ensembles -> figures
train_ensemble_config.yaml          tune metric:   fit + serialise -> models/*.joblib
validate_learned_metric_config.yaml use metric:    run a trained metric as a Validator
results/  figures/  models/         artefacts (DVC-tracked: results, models, figures)

src/metric_validation/   FeatureScoreMatrix, EnsembleModel, LearnedEnsembleExperiment,
                         LearnedEnsembleTrainer, FeatureAnalysisExperiment,
                         MetricFiguresRenderer, MetricsValidationExperiment
src/sample_metric/learned_ensemble_metric.py   LearnedEnsembleMetric (drop-in SampleMetric)
```

All commands run from the repo root with the `massing` conda environment active.

## Run the study

Scores the feature matrix, ranks features, benchmarks the learned ensembles, and
renders figures + `results/consolidated_results.json`.

```bash
python hydra_run.py \
  --config_path experiments/benchmarks/coma/metrics_validation/ensemble_validation_config.yaml
```

`results/scores.npz` (the cached 17×~25k score matrix) is reused when it matches the
feature pool, so reruns skip the slow scoring — set `feature_matrix.force_rescore=true`
to recompute.

## Validate the learned metric on a new dataset

The learned metric is a normal `SampleMetric`, so it runs through the repo's standard
`Validator`. To score a **new** dataset, point `paths.val` at it and `paths.ctx` at its
context pool — the trained weights in `models/` are reused as-is:

```bash
python hydra_run.py \
  --config_path experiments/benchmarks/coma/metrics_validation/validate_learned_metric_config.yaml \
  paths.val=/path/to/new/testset \
  paths.ctx=/path/to/new/context
# -> results/learned_metric_validation.json   {ROC-AUC, F1}
```

To (re)train the metric on a different benchmark first, run the trainer with the same
override style (it reports GroupKFold CV, then serialises the artifact):

```bash
python hydra_run.py \
  --config_path experiments/benchmarks/coma/metrics_validation/train_ensemble_config.yaml \
  paths.val=/path/to/bench/testset paths.ctx=/path/to/bench/context \
  method.feature_matrix.cache_path=/path/to/bench/scores.npz \
  method.model_output_path=/path/to/model.joblib
```

The artifact stores only the combiner (estimator + feature order + scaling); features
come from `${_feature_pool}`. Swap the model or feature subset via `method.model` /
`method.feature_set` overrides.

## No-orientation ablation

The orientation ablation removes the direction and explicit orientation-distribution
features from the learned CR metric:

`direction_nearest`, `direction_mean`, `frechet_orient`, `orient_kde`, and
`frechet_classic`. The rotation-invariant `anglediv_nearest` shape feature remains.

Run the full local ablation workflow from the repo root:

```bash
python experiments/benchmarks/coma/run_no_orientation_ablation.py --pull-dvc
```

This retrains `models/learned_ensemble_catboost_no_orientation.joblib`, evaluates it
from `results/scores.npz` into `results/learned_metric_no_orientation_validation.json`, rescors any generated
submissions available under `experiments/benchmarks/coma/methods_validation/` into
`experiments/benchmarks/coma/results_no_orientation/`, and renders publication plots
under `experiments/benchmarks/coma/results_visualization/visualizations_publication_no_orientation/`.

If the generated submissions are not materialized locally, run the metric-only part:

```bash
python experiments/benchmarks/coma/run_no_orientation_ablation.py --pull-dvc --skip-methods --skip-visualizations
```
