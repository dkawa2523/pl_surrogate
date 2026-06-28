# 06 YAML Reference

This file shows minimal product YAML fragments. Dataset-specific experiments
should live in `configs/experimental/`.

## Table Only

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

preprocessing:
  scalers:
    target_transforms:
      electron_density:
        value_transform: identity
        scaler: zscore
        fit_scope: plasma_only
        clip:
          mode: none

model:
  name: global_mlp
```

## Table Plus Structure

```yaml
runtime:
  input_mode: table_plus_structure
  structure:
    feature_profile: geom_v1_mainline
    adapter_mode: grid_pack
    provider_mode: fixed

benchmark:
  eval_protocol:
    mode: dual_axis
    primary_split: interp
    interp_weight: 0.5
    extrap_weight: 0.5
  eval:
    primary_metric: surrogate_quality_score
    objective_mode: min
    target_vars_for_score: auto
```

## Training

```yaml
train:
  loss:
    protocol: plasma_surrogate_v2
    supervised:
      type: mse
      mask: plasma_only
    group_weighting:
      mode: none
  unetpp_attn:
    model_cfg:
      backend: torch
      output_heads:
        mode: shared
```

Opt-in loss balancing by target group:

```yaml
train:
  loss:
    protocol: plasma_surrogate_v2
    group_weighting:
      mode: uniform_by_group
```

Opt-in Huber supervised loss. MSE remains the default.

```yaml
train:
  loss:
    protocol: plasma_surrogate_v2
    supervised:
      type: huber
      huber_delta: 1.0
      mask: plasma_only
```

Opt-in custom grouped heads for supported grid/operator models:

```yaml
train:
  fno:
    model_cfg:
      backend: torch
      output_heads:
        mode: custom_groups
        strict: true
        groups:
          density:
            targets: [electron_density, ion_density]
          thermal:
            targets: [electron_temperature]
          electrostatic:
            targets: [plasma_potential]
```

Physics residuals are configured separately from the supervised loss protocol:

```yaml
physics:
  enabled: true
  symbols:
    density: electron_density
    temperature: electron_temperature
    potential: plasma_potential
  terms:
    poisson:
      enabled: true
      weight: 0.1
    boundary_operator:
      enabled: false
      weight: 0.0
```

## Optimization

Optional inference-time derived fields:

```yaml
inference:
  derived_fields_strict: false
  derived_fields:
    - id: electric_field
      operator: negative_gradient
      source: plasma_potential
    - id: electric_field_magnitude
      operator: vector_magnitude
      sources: [electric_field_x, electric_field_y]
```

```yaml
inference:
  optimize:
    backend: random
    objective:
      mode: weighted_sum
      terms:
        - key: uniformity
          direction: min
          weight: 1.0
    constraints:
      - key: poisson_residual_norm
        upper: 100.0
```
