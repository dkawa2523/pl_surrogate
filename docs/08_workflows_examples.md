# 08 Workflow Examples

Install the package in editable mode before running these examples:

```bash
python -m pip install -e ".[dev]"
```

For torch-backed runs, also enable the backend:

```bash
export PLASMA_SURROGATE_ENABLE_TORCH=1
```

On Windows PowerShell:

```powershell
$env:PLASMA_SURROGATE_ENABLE_TORCH = "1"
```

## Benchmark Examples

Global frozen reference:

```bash
plasma-surrogate benchmark run \
  --config tests/fixtures/benchmark_periodic_real_m7_global_frozen_ref.yaml
```

UNet mainline:

```bash
plasma-surrogate benchmark run \
  --config tests/fixtures/benchmark_periodic_real_m7_unet_isolated_mainline.yaml
```

FNO mainline:

```bash
plasma-surrogate benchmark run \
  --config tests/fixtures/benchmark_periodic_real_m7_fno_isolated_mainline.yaml
```

DeepONet mainline:

```bash
plasma-surrogate benchmark run \
  --config tests/fixtures/benchmark_periodic_real_m7_deeponet_isolated_mainline.yaml
```

Main artifacts:

- `leaderboard.csv`
- `resolved_benchmark.json`
- `manifest.json`

## Compare Example

```bash
python scripts/compare_selected_models.py \
  --config tests/fixtures/benchmark_periodic_real_compare_mainline_fno_deeponet_global.yaml
```

The compare step reads benchmark leaderboards and writes one
`selected_models_comparison.csv`.  Headers are target-driven, so do not add
fixed `ne`, `ni`, `Te`, or `phi` columns to new compare code unless the active
dataset declares those targets.

## Spatial Plot Example

```bash
python scripts/plot_spatial_distribution_summary.py \
  --compare-csv runs/periodic_real_tuned_v83/compare_mainline_fno_deeponet_global/selected_models_comparison.csv \
  --out-dir runs/periodic_real_tuned_v83/compare_mainline_fno_deeponet_global/plots_interp \
  --protocol interp \
  --vars ne ni Te phi \
  --include-ground-truth \
  --reference-name global_v2_frozen_ref
```

`--vars` must match the compare CSV and ground-truth target names.

## Pipeline Example

```bash
plasma-surrogate pipeline --config configs/mainline.yaml
```

The pipeline task executes the configured stages and writes per-stage
manifests under the run artifact directory.

## Stage-By-Stage Examples

```bash
plasma-surrogate cleanse --config configs/mainline.yaml
plasma-surrogate feature --config configs/mainline.yaml
plasma-surrogate preprocess --config configs/mainline.yaml
plasma-surrogate train --config configs/mainline.yaml
plasma-surrogate infer --config configs/mainline.yaml
plasma-surrogate evaluate --config configs/mainline.yaml
```

Use stage-by-stage execution when introducing a new dataset, target set,
feature pack, or model adapter. It catches contract drift earlier than a full
pipeline run.

## Fixtures And Production Configs

`tests/fixtures` contains small regression templates and smoke-test examples.
Production configs and long-running experiment configs should live under
`configs/` or outside the package tree.

When moving from the m7 example dataset to another dataset, review:

- `dataset.targets[]`
- `dataset.cond_columns`
- `dataset.geometry_root`
- `preprocessing.scalers.target_transforms`
- `train.<model>.target_vars`
- `benchmark.eval.target_vars_for_score`
- physics symbol mappings
- plot `--vars`
