# 02 Data Contract

All product datasets declare target fields through `dataset.targets[]`.

```yaml
dataset:
  targets:
    - id: electron_density
      source_key: electron_density_field
      role: density_electron
      positive: true
      field_family: density
      default_region: plasma_only
      value_transform: identity
      units: m^-3
      dtype: float32
```

## Target Entry

- `id`: logical target name used by the product.
- `source_key`: source field name in the raw dataset.
- `role`: physics and diagnostics identity.
- `positive`: whether negative predictions are physically invalid.
- `field_family`: broad family used for role fallback.
- `default_region`: default region for loss and metrics.
- `value_transform`: currently metadata only for product docs; the `csv_npz`
  loader accepts `identity` and rejects non-identity target transforms.

The target list order is an input declaration. After preprocessing, the runtime
target order is `output_layout.vars`.

## Role Resolution

Physics and diagnostics resolve targets in this order:

1. Explicit `physics.symbols` or `inference.ood.*.symbols`.
2. Unique `role` in `target_role_schema.json`.
3. Unique `field_family` in `target_role_schema.json`.
4. Fail fast with a short missing or ambiguous role error.

Target names are labels, not physics contracts.

## Transform Boundary

Runtime value transforms, scalers, fit scope, and clipping are configured under
`preprocessing.scalers.target_transforms.<target>` and persisted in
`preprocessing/scalers/y_scalers.json`. Downstream code uses `TransformBundle`
for forward and inverse transforms. Model, loss, metric, and inference code
should not infer special transforms from target names or source field names.
