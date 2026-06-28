# 01 Architecture

The product architecture is an artifact pipeline. Each stage reads the previous
stage's declared artifacts instead of rediscovering target order, feature order,
or runtime mode from local assumptions.

## Pipeline

- `preprocess`: builds splits, schemas, scalers, feature packs, target role
  metadata, and runtime schema hashes.
- `train`: reads preprocessing artifacts, builds a model from `model_specs.py`,
  trains it, and writes a checkpoint with runtime metadata.
- `infer`: reads checkpoint and preprocessing artifacts, generates fields, QoI,
  diagnostics, and optional optimization outputs.
- `evaluate`: computes target-aware metrics and validity flags.
- `benchmark`: compares model runs with an explicit evaluation protocol and
  selection metric.

The CLI is only an execution entrypoint. The contracts live in dataset config,
preprocessing artifacts, model specs, runtime metadata, and evaluation protocol.

## Artifact Flow

- Target order: `preprocessing/schema/output_layout.json`.
- Field layout metadata: `preprocessing/schema/field_layout.json` for current
  grid2d fields; not a replacement for target order.
- Target roles: `preprocessing/schema/target_role_schema.json`.
- Condition schema: `preprocessing/schema/cond_schema.json`.
- Feature metadata: `preprocessing/features/*_meta.json`.
- Channel map: `preprocessing/schema/channel_map.json`.
- Target transforms: `preprocessing/scalers/y_scalers.json`.
- Runtime hashes: `preprocessing/validation/runtime_schema_hashes.json`.

## Runtime Contracts

`RuntimeContract` is the compact internal boundary for input mode, structure
feature profile, adapter mode, provider mode, and schema hashes. Pending hashes
are allowed only while building metadata, not while executing train, infer, or
benchmark.

`EvaluationProtocol` records mode, primary split, primary metric, objective
mode, target variables, region bands, and dual-axis weights. Benchmark resolved
config writes this protocol so third-party readers can see how a leaderboard was
selected.

## Model Lane

Model capability belongs in `src/plasma_surrogate/core/model_specs.py`. Train,
infer, and benchmark should dispatch from capabilities and adapters, not from
parallel hard-coded model policies.
