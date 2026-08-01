# 07 Extension Guide

Use the smallest implemented contract surface. There is no full ModelPlugin
system yet; new work should start from the current registries and adapters.

## New Target

1. Add the field to `dataset.targets[]` with `id`, optional `source_key`, and
   role metadata such as `role`, `field_family`, and `positive`.
2. Keep `dataset.targets[].value_transform: identity` for current `csv_npz`
   product datasets.
3. Add reversible transforms, scaler, fit scope, and clipping under
   `preprocessing.scalers.target_transforms.<target>`.
4. Run preprocessing and check `preprocessing/schema/output_layout.json` and
   `preprocessing/schema/target_role_schema.json`.
5. Add `physics.symbols` only when role/family metadata cannot resolve a unique
   density, temperature, or potential.

Metrics, target groups, and positive diagnostics derive from
`output_layout.vars` plus `target_role_schema`; do not add target-name branches.

## New Transform Or Scaler

Add target value transforms and scaler behavior in
`preprocessing/scalers.py`. That module owns:

- allowed `target_transforms.<target>.value_transform` values
- scaler construction and serialization
- `TransformBundle` forward/inverse behavior
- clipping and fit-scope handling

Do not implement target transforms in dataset loading, training, inference, or
benchmark code.

## New Model

1. Add the model id and capability in `core/model_specs.py`.
2. Add the implementation under `models/`.
3. Add construction in `models/factory.py`.
4. Add checkpoint save/load support in the relevant checkpoint module.
5. Reuse an existing train lane in `train/model_adapters.py` when possible.
   Add a new adapter only if no existing lane fits.
6. Put lane-specific validation in a narrow helper such as
   `train/grid_contracts.py` or `train/deeponet_contracts.py`.
7. Add inference-specific code only when generic checkpoint/model inference is
   insufficient.
8. For first-class benchmark inclusion, set `benchmark_scope` in
   `core/model_specs.py`; benchmark scope maps are derived from model specs.

Do not copy model capability tables into train, infer, and benchmark. If a new
table seems necessary, first check whether it belongs in `ModelSpec`.

## New Output Head Mode

`output_heads.mode: shared` is the default baseline. Implement new modes only as
opt-in behavior.

Current grouped-head code lives in `models/heads/role_grouped.py`:

- mode validation
- custom group parsing
- target group metadata serialization
- lightweight grouped conv head construction

For a new head mode, add validation there, wire only supported model families,
preserve `output_layout.vars` order, and add checkpoint/inference contract
tests. Non-default group heads such as `poisson_hybrid` are planned experiment
lanes unless explicitly implemented and tested.

## New Physics Term

Physics terms are configured through `physics.terms`; removed keys such as
`lambda_poisson` stay rejected.

1. Add the term name to `REGISTERED_PHYSICS_TERMS` in
   `core/physics_contract.py` only when the term is implemented.
2. Keep symbols top-level in `physics.symbols`; term-local `symbols` are not
   supported.
3. Extend `train/physics_terms.py` if the term needs resolution metadata beyond
   `name`, `enabled`, and `weight`.
4. Add the numerical equation in `train/losses.py` for NumPy and/or
   `train/torch_losses.py` for Torch.
5. Compose the term in `train/loss_composer.py`; `LossComposer` should combine
   resolved terms, not own symbol resolution policy.
6. Add tests for config validation, symbol resolution, and enabled/disabled
   behavior.

Do not document a physics term as available until both validation and loss
calculation exist for the intended trainer.

## New Feature

1. Build the feature artifact in preprocessing.
2. Save feature order, shape, and metadata with the artifact.
3. Read the same metadata in train, infer, and benchmark.
4. Fail fast when a required feature is missing.

Keep preprocessing feature construction outside train dispatch code.

## New Metric Or Diagnostic

Metric rows should be target-driven and validity-aware. Default benchmark
columns should stay compact:

- target-wise RMSE/R2
- plasma target-wise RMSE/R2
- group-wise RMSE/R2
- plasma group-wise RMSE/R2
- `surrogate_quality_score`

Region, positive, physics, or detailed spatial diagnostics should remain
opt-in through benchmark/eval diagnostics.

## Minimal Tests

Pick the smallest test set that covers the changed contract:

- dataset target parsing
- preprocessing artifact roundtrip
- transform/scaler inverse behavior
- feature channel order
- model spec and train adapter selection
- checkpoint save/load
- inference symbol mapping
- output head shape/order and metadata
- physics term validation and contribution
- benchmark metric row smoke
