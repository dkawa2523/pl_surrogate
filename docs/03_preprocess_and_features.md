# 03 Preprocess And Features

`preprocess` creates the artifacts shared by train, infer, evaluation, and
benchmark. Later stages should read these artifacts instead of rebuilding local
contracts.

## Canonical Artifacts

- `schema/output_layout.json`: target order.
- `schema/field_layout.json`: grid2d field metadata copied from the current
  output layout for future readers; it is not the runtime target-order source.
- `schema/target_role_schema.json`: role, positivity, and region metadata.
- `schema/cond_schema.json`: condition and axis inputs.
- `schema/channel_map.json`: model input channels.
- `features/*_meta.json`: feature pack channel order and shape.
- `scalers/y_scalers.json`: fitted target scaler artifacts.
- `validation/runtime_schema_hashes.json`: target and feature schema hashes.

## Target Transforms

`preprocessing.scalers.target_transforms.<target>` is the source of truth for
runtime target value transforms, scaler type, fit scope, and clipping. The
resolved contract is written to scaler artifacts and loaded through
`TransformBundle`.

Clipping modes are `none`, train-only `quantile`, and explicit
`physical_bounds`. Prefer `physical_bounds` when low-power, off-state, or other
rare physical regimes are meaningful; quantile clipping intentionally removes
the clipped tails from both training and inverse prediction.

`dataset.targets[]` describes how to read raw fields and attach metadata. It
does not replace preprocessing target transform config.

## Field Layout

`field_layout.json` is a small metadata artifact for the current fixed 2D grid
lane. It records `layout_type: grid2d`, `vars`, `[C,H,W]` shape, storage order,
and axes. Train, infer, and benchmark still use `output_layout.json` as the
canonical target order.

Graph, time-series, and variable-mesh layouts are not implemented product
layouts yet.

## Runtime Structure

Runtime input mode is one of:

- `table_only`
- `table_plus_structure`

The structure contract has three main knobs: `feature_profile`, `adapter_mode`,
and `provider_mode`. `table_only` uses no structure profile. `table_plus_structure`
requires a concrete feature profile.

## Feature Rules

- Feature order is an artifact contract.
- Train, infer, and benchmark must not rewrite feature lists independently.
- Descriptor and latent profiles are optional lane metadata, not required
  product runtime keys.
- Missing required feature artifacts should fail before model construction.
- `smooth_structure_v1` is the order-invariant grid default when structure
  imprinting is a risk. It uses coordinates, plasma mask, signed distance,
  boundary proximity, and union-of-solids proximity, and excludes raw normals,
  curvature, and nearest/second-part summaries.
