# 06 YAML Reference

製品 docs では最小 YAML fragment だけを示す。詳細な実験設定や dataset-specific note は product contract にしない。

## table_only

```yaml
runtime:
  input_mode: table_only
  structure:
    adapter_mode: none
    provider_mode: fixed

dataset:
  type: csv_npz
  root: data/product_dataset
  index_csv: index.csv
  cond_columns: [pressure, power, gap]
  fields_npz_column: fields_npz
  case_id_column: case_id
  targets:
    - id: electron_density
      source_key: electron_density_field
      role: density_electron
      positive: true
      field_family: density
      default_region: plasma_only
      value_transform: identity

model:
  name: global_mlp
```

## table_plus_structure

```yaml
runtime:
  input_mode: table_plus_structure
  structure:
    feature_profile: geom_v1_mainline
    adapter_mode: grid_pack
    provider_mode: fixed

dataset:
  type: csv_npz
  root: data/product_dataset
  index_csv: index.csv
  cond_columns: [pressure, power, gap]
  fields_npz_column: fields_npz
  case_id_column: case_id
  geometry_root: geometry
  targets:
    - id: electron_density
      source_key: electron_density_field
      role: density_electron
      positive: true
      field_family: density
      default_region: plasma_only
      value_transform: identity
    - id: plasma_potential
      source_key: plasma_potential_field
      role: potential
      positive: false
      field_family: electrostatic
      value_transform: identity

preprocessing:
  coord_features:
    enabled: true
    channels_from_profile: geom_v1_mainline

model:
  name: fno
```

## First-Class Model Fragments

`model.name` の capability は `src/plasma_surrogate/core/model_specs.py` が正本である。
全 first-class model の分類は `docs/04_training_models.md` を参照する。

```yaml
model:
  name: global_mlp
```

```yaml
model:
  name: unetpp_attn
train:
  unetpp_attn:
    input_features:
      mode: geom_feature_pack
    model_cfg:
      backend: torch
      output_heads:
        mode: shared
```

```yaml
model:
  name: geom_deeponet_siren
runtime:
  input_mode: table_plus_structure
  structure:
    feature_profile: geom_v1_mainline
    adapter_mode: hybrid_pack_descriptor
    provider_mode: fixed
```

## Loss Protocol

```yaml
train:
  loss:
    protocol: plasma_surrogate_v2
```

Override は必要な key だけを書く。

```yaml
train:
  loss:
    protocol: plasma_surrogate_v2
    supervised:
      spatial_consistency:
        lambda: 0.03
      positive_penalty:
        lambda: 0.005
```

## Optimization

```yaml
inference:
  ood:
    physics:
      enabled: true
      symbols:
        density: electron_density
        temperature: electron_temperature
        potential: plasma_potential
  optimize:
    enabled: true
    backend: optuna
    n_trials: 16
    space:
      pressure: [0.1, 1.0]
      power: [0.1, 1.0]
      gap: [0.1, 1.0]
    objective:
      mode: weighted_sum
      terms:
        - key: uniformity
          direction: min
          weight: 1.0
        - key: boundary_gamma_uniformity
          direction: min
          weight: 0.3
        - key: poisson_residual_norm
          direction: min
          weight: 0.2
          transform: log1p_abs
          scale: 1.0
    constraints:
      - key: poisson_residual_norm
        upper: 0.05
    output:
      save_fields: top_k
      top_k: 3
```

## Benchmark Quality Selection

```yaml
benchmark:
  eval:
    primary_metric: surrogate_quality_score
    objective_mode: min
    quality_score:
      enabled: true
      weights:
        nrmse: 0.45
        boundary: 0.20
        continuity: 0.15
        physics: 0.15
        sign: 0.05
```

R2 / RMSE は補助指標である。benchmark selection は lower-better の `surrogate_quality_score` を既定にする。
