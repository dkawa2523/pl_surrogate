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

Use physical bounds when rare but valid regimes must survive fitting. Bounds
are specified in physical units and persisted in both physical and transformed
spaces:

```yaml
preprocessing:
  scalers:
    target_transforms:
      electron_density:
        value_transform: log10_floor
        floor: 1.0e-30
        scaler: zscore
        fit_scope: plasma_only
        clip:
          mode: physical_bounds
          min: 1.0e8
          max: 1.0e19
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

For case-varying parts, make the row-to-geometry relation explicit and persist
split static/case packs. `case_structure_feature_pack.npz` carries `case_ids`;
training and evaluation reject misaligned rows.

```yaml
runtime:
  input_mode: table_plus_structure
  structure:
    feature_profile: part_lite_v1
    descriptor_profile: none
    adapter_mode: coord_pack
    provider_mode: parametric_parts

dataset:
  structure_npz_column: structure_npz

preprocessing:
  coord_features:
    enabled: true
    channels_from_profile: part_lite_v1
    static_output: features/static_spatial_feature_pack.npz
    case_structure_output: features/case_structure_feature_pack.npz
    require_case_variation: true
    scaling:
      enabled: true
      mode: zscore
      fit_scope: train_split
```

Static channels and case channels are reassembled in registered profile order.
Raw supervision geometry is materialized separately from transformed model
features, so case masks also control target-scaler fitting, loss, validation,
and evaluation. A static fallback is used only when the case pack does not
declare raw mask/distance channels.

A table-only baseline can still use case geometry as objective metadata without
feeding it into the model. Declare that intent explicitly:

```yaml
runtime:
  input_mode: table_only
  structure:
    feature_profile: none
    adapter_mode: none

preprocessing:
  coord_features:
    enabled: true
    usage: supervision_only
    channels_from_profile: part_lite_v1
    static_output: features/static_spatial_feature_pack.npz
    case_structure_output: features/case_structure_feature_pack.npz
```

`usage: supervision_only` affects scaler fitting, loss, checkpoint selection,
and evaluation, but never extends the model condition vector.

## Training

```yaml
train:
  loss:
    protocol: plasma_surrogate_v3
    supervised:
      type: huber
      huber_delta: 1.0
      mask: plasma_only
      normalization: sample_mean
      spatial:
        gradient_weight: 0.10
        gradient_spacing: [1.0, 1.0]
        gradient_normalization: target_rms  # none | target_rms
        gradient_epsilon: 0.05
        multiscale_weight: 0.05
        multiscale_scales: [2, 4]
        boundary_weight: 0.25
        boundary_band_px: 2.0
        boundary_distance_channels: [distance_any]
    group_weighting:
      mode: uniform_by_group
  fno:
    selection:
      mode: best_val_spatial_objective
      eval_every_n_epochs: 2
      warmup_epochs: 8
      spatial:
        point_weight: 1.0
        gradient_weight: 0.10
        gradient_normalization: target_rms
        gradient_epsilon: 0.05
        boundary_weight: 0.25
        boundary_band_px: 2.0
        gradient_spacing: [1.0, 1.0]
      case_aggregation:
        median_weight: 1.0
        p90_weight: 0.25
        worst_weight: 0.10
```

The same loss and selection contract applies to FNO, coordinate DeepONet,
coordinate MLP/POD residual, and field-output baselines. Model names do not
change metric semantics.

Version 3 defaults to `uniform_by_group`: every non-empty `field_family` gets
equal influence, and targets within a family share that influence. Therefore
`target_role_schema.json` must cover every trained output exactly once. Use
`uniform_by_target` only when every target channel should have equal influence,
or `none` when reproducing legacy summed-target behavior.

Target-family balancing can also be selected explicitly:

```yaml
train:
  loss:
    protocol: plasma_surrogate_v3
    group_weighting:
      mode: uniform_by_group
```

For the legacy point-only protocol, Huber is opt-in and MSE remains the
default:

```yaml
train:
  loss:
    protocol: plasma_surrogate_v2
    supervised:
      type: huber
      huber_delta: 1.0
      mask: plasma_only
```

Spatial terms can be overridden without replacing the other version-3
defaults:

```yaml
train:
  loss:
    protocol: plasma_surrogate_v3
    supervised:
      spatial:
        gradient_weight: 0.10
        multiscale_weight: 0.05
        multiscale_scales: [2, 4]
```

Finite differences use only edges whose two cells are active in the supervised
mask. Multiscale terms use masked pooling, so neither term crosses the plasma
boundary.

`gradient_normalization: target_rms` makes the gradient term dimensionless per
case. `gradient_epsilon` prevents nearly uniform target fields from producing
an unstable divisor. The defaults are `none` and `0.05`, so legacy
configurations retain their previous numerical behavior.

For the mainline plasma DeepONet with case-varying geometry, pool the
case-level sensor set in the branch while retaining local geometry in the
trunk:

```yaml
runtime:
  input_mode: table_plus_structure
  structure:
    feature_profile: part_lite_v1
    adapter_mode: coord_pack
    provider_mode: parametric_parts

train:
  loss:
    supervised:
      spatial:
        # Chamber/plasma edge plus the nearest case-varying part surface.
        boundary_distance_channels: [distance_any, part_sdf_nearest]
  deeponet_plasma:
    input_features:
      mode: geom_feature_pack
      features:
        - x
        - y
        - mask_plasma
        - distance_signed
        - distance_any
        - normal_x
        - normal_y
        - curvature_proxy
        - boundary_band
        - part_sdf_nearest
        - part_sdf_second
        - part_gap_proxy
        - solid_proximity
    model_cfg:
      trunk_input_mode: geom_feature_pack
      branch_mode: set_mlp_pool
      sensor_pool_mode: set_mlp_pool
```

`deeponet_plasma` and coordinate MLP models accept `coord_pack` (or `auto`,
which resolves to it). Descriptor adapters are accepted only by model lanes
that explicitly load and validate descriptor artifacts.

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
