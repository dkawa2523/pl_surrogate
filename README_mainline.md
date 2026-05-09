# Plasma Surrogate Mainline

This repository provides a pipeline-oriented package for plasma surrogate
experiments: data cleaning, feature preparation, preprocessing, training,
inference, evaluation, visualization, and benchmark comparison.

The current mainline goal is to keep the scientific workflow intact while
making the codebase easier to maintain, extend, and use from a third-party
environment.

## Scope

- Mainline models: `global_mlp`, `unet`, `fno`, `deeponet_plasma`
- Optional or experimental models: `unetpp`, `unetpp_attn`, `ffno`,
  `coord_mlp_fourier`, `coord_mlp_siren`, `coord_mlp_pod_residual`, `deeponet_pod`,
  `deeponet_plasma_pod`, `geom_deeponet_pod`, `u_no`, `cno`,
  `geom_deeponet_siren`
- Pipeline tasks: `cleanse`, `feature`, `preprocess`, `train`, `infer`,
  `evaluate`, `pipeline`, `benchmark`, `viz`
- Target contract: active targets come from `dataset.targets[]`; fixed
  `ne`, `ni`, `Te`, `phi` assumptions should not be added to new code.

## Install

For local development:

```bash
python -m pip install -e ".[dev]"
```

For CUDA/PyTorch runs, install the appropriate PyTorch wheel for the machine,
then install the optional package dependencies:

```bash
python -m pip install -e ".[torch,optuna,dev]"
```

The legacy `PYTHONPATH=src` workflow is no longer the preferred path. Editable
install keeps CLI, tests, subprocesses, and user scripts on the same import
path.

## Quick Commands

Run the full pipeline:

```bash
plasma-surrogate pipeline --config configs/mainline.yaml
```

Run a benchmark fixture:

```bash
PLASMA_SURROGATE_ENABLE_TORCH=1 plasma-surrogate benchmark run --config tests/fixtures/benchmark_periodic_real_m7_fno_isolated_mainline.yaml
```

Compare selected benchmark outputs:

```bash
python scripts/compare_selected_models.py --config tests/fixtures/benchmark_periodic_real_compare_mainline_fno_deeponet_global.yaml
```

Equivalent module invocation is also supported:

```bash
python -m plasma_surrogate benchmark run --config <benchmark.yaml>
```

## Contract Essentials

- `dataset.targets[]` is the source of truth for output variables.
- Preprocessing target transforms live under
  `preprocessing.scalers.target_transforms.<var>`.
- Benchmark and compare output columns are generated from active targets.
- Inference physics symbols are configured with:
  - `inference.ood.physics.symbols`
  - `inference.ood.boundary_operator.symbols`
- Model input-mode support is defined in
  `src/plasma_surrogate/core/model_specs.py`.

## Mainline Fixture Set

The m7 fixtures are examples and regression fixtures, not production configs:

- `tests/fixtures/benchmark_periodic_real_m7_global_frozen_ref.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_unet_isolated_mainline.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_fno_isolated_mainline.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_deeponet_isolated_mainline.yaml`
- `tests/fixtures/benchmark_periodic_real_compare_mainline_fno_deeponet_global.yaml`

Production or long-running experiment configs should live under `configs/` or
outside the package tree.

## Outputs To Check

- `runs/.../leaderboard.csv`
- `runs/.../resolved_benchmark.json`
- `runs/.../selected_models_comparison.csv`
- `runs/.../artifacts/<stage>/manifest.json`

## Development Notes

- Keep user-facing validation strict at YAML, dataset, checkpoint, and model
  adapter boundaries.
- Avoid adding model-specific branches to train, infer, benchmark, and
  checkpoint code in parallel. Add the stable capability metadata to
  `model_specs.py` first, then wire the smallest model-specific adapter needed.
- Keep generated reports, converted datasets, benchmark outputs, and large
  archives outside the core package tree.

## Tests

Default pytest runs skip generated config catalog checks, full benchmark
smoke/regression runs, and tests that require an enabled torch runtime:

```bash
pytest
```

Run slow benchmark checks explicitly:

```bash
pytest -m benchmark_slow tests/integration
```

Run torch-runtime checks explicitly after installing/enabling torch:

```bash
PLASMA_SURROGATE_ENABLE_TORCH=1 pytest -m torch_runtime
```

Run generated/config inventory checks explicitly:

```bash
pytest -m config_catalog
```

## Documentation

- Architecture: `docs/01_architecture.md`
- Data contract: `docs/02_data_contract.md`
- Preprocess and features: `docs/03_preprocess_and_features.md`
- Training models: `docs/04_training_models.md`
- Inference and evaluation: `docs/05_inference_and_evaluation.md`
- YAML reference: `docs/06_yaml_reference.md`
- Extension guide: `docs/07_extension_guide.md`
- Workflow examples: `docs/08_workflows_examples.md`
- Dual input-mode runtime: `docs/11_dual_input_mode_runtime.md`
