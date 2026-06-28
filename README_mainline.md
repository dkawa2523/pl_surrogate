# Plasma Surrogate

`plasma_surrogate` is a product foundation for plasma simulation surrogate
models. It connects condition tables, structure data, grid/coordinate features,
models, inference, benchmarking, and optimization through preprocessing
artifacts.

The mainline contract is target-driven. A workflow must not depend on fixed
physics target names, benchmark-only assumptions, or hidden runtime
compatibility paths.

## Install

```bash
python -m pip install -e ".[dev]"
```

For torch, Optuna, and local benchmark work:

```bash
python -m pip install -e ".[torch,optuna,dev]"
```

## Quickstart

```bash
plasma-surrogate pipeline --config path/to/config.yaml
plasma-surrogate preprocess --config path/to/config.yaml
plasma-surrogate train --config path/to/config.yaml
plasma-surrogate infer --config path/to/config.yaml
plasma-surrogate benchmark run --config path/to/benchmark.yaml
```

Equivalent module invocation is supported:

```bash
python -m plasma_surrogate benchmark run --config path/to/benchmark.yaml
```

## Product Contract

- Target definitions start at `dataset.targets[]`.
- Target order after preprocessing is `preprocessing/schema/output_layout.json`.
- Target role metadata is `preprocessing/schema/target_role_schema.json`.
- Raw target metadata starts in `dataset.targets[]`; reversible transforms,
  scalers, and clipping are preprocessing artifacts.
- Feature and channel order come from `coord_feature_pack_meta.json` and
  `channel_map.json`.
- Train, infer, and benchmark use the same preprocessing artifact bundle.
- Runtime input mode is only `table_only` or `table_plus_structure`.
- Runtime structure metadata is limited to `feature_profile`, `adapter_mode`,
  and `provider_mode`.
- Checkpoint, inference, and benchmark metadata validate `target_schema_hash`
  and `feature_schema_hash` fail-fast.
- Model capability is defined in `src/plasma_surrogate/core/model_specs.py`.
- First-class product examples use `output_heads.mode: shared`.
- Benchmark selection defaults to lower-better `surrogate_quality_score`.
- Optimization uses `inference.optimize.objective`.

## Outputs To Check

- `runs/.../preprocessing/schema/output_layout.json`
- `runs/.../preprocessing/schema/target_role_schema.json`
- `runs/.../preprocessing/schema/channel_map.json`
- `runs/.../preprocessing/features/coord_feature_pack_meta.json`
- `runs/.../preprocessing/validation/runtime_schema_hashes.json`
- `runs/.../leaderboard.csv`
- `runs/.../inference/optimize/summary.json`

## Development

Start model additions in `src/plasma_surrogate/core/model_specs.py`, then add
the smallest implementation and adapter needed. Keep generated reports,
experiment catalogs, and dataset-specific notes outside product docs.

Default tests:

```bash
pytest
```

Torch-runtime tests:

```bash
PLASMA_SURROGATE_ENABLE_TORCH=1 pytest -m torch_runtime
```

## Documentation

- Product policy: `docs/00_product_foundation_policy.md`
- Architecture: `docs/01_architecture.md`
- Data contract: `docs/02_data_contract.md`
- Preprocess and features: `docs/03_preprocess_and_features.md`
- Training and models: `docs/04_training_models.md`
- Inference and evaluation: `docs/05_inference_and_evaluation.md`
- YAML reference: `docs/06_yaml_reference.md`
- Extension guide: `docs/07_extension_guide.md`
