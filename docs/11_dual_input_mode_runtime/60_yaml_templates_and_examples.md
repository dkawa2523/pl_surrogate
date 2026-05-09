# 60. YAML Templates and Examples

## 1. table_only / global_mlp

```yaml
runtime:
  input_mode: table_only
  strict_input_mode: error
  allow_mode_fallback: false
  structure:
    feature_profile: none
    descriptor_profile: none
    latent_profile: none
    adapter_mode: none
    provider_mode: fixed

dataset:
  type: csv_npz
  root: data/m7
  cond_columns: [pressure, power, flow]
  geometry_root: data/m7/geometry

train:
  models: [global_mlp]
  global_mlp:
    target_family: allvars
    target_vars: [log_ne, Te, phi]
    selection:
      mode: best_val_allvars_balance
```

## 2. table_only / deeponet_pod

```yaml
runtime:
  input_mode: table_only
  structure:
    feature_profile: none
    descriptor_profile: none
    latent_profile: none
    adapter_mode: none
    provider_mode: fixed

train:
  models: [deeponet_pod]
  deeponet_pod:
    target_family: allvars
    target_vars: [log_ne, Te, phi]
    model_cfg:
      basis:
        rank: 32
        fit_scope: train_only
        per_var: true
        center: true
```

## 3. table_plus_structure / ffno / part_lite_v1

```yaml
runtime:
  input_mode: table_plus_structure
  strict_input_mode: error
  allow_mode_fallback: false
  structure:
    feature_profile: part_lite_v1
    descriptor_profile: none
    latent_profile: none
    adapter_mode: grid_pack
    provider_mode: parametric_parts

preprocessing:
  coord_features:
    channels_from_profile: part_lite_v1

train:
  models: [ffno]
  ffno:
    target_family: allvars
    target_vars: [log_ne, Te, phi]
    input_features:
      mode: geom_feature_pack
      profile: part_lite_v1
      require_pack: error
```

## 3b. table_plus_structure / u_no / part_lite_v1

```yaml
runtime:
  input_mode: table_plus_structure
  strict_input_mode: error
  allow_mode_fallback: false
  structure:
    feature_profile: part_lite_v1
    descriptor_profile: none
    latent_profile: none
    adapter_mode: grid_pack
    provider_mode: parametric_parts

model:
  name: u_no
  backend: torch
  n_modes: 12
  uno_cfg:
    width: 64
    n_layers: 4
    dropout: 0.0

preprocessing:
  coord_features:
    channels_from_profile: part_lite_v1

train:
  u_no:
    target_family: allvars
    input_features:
      mode: geom_feature_pack
      profile: part_lite_v1
      require_pack: error
```

## 4. table_plus_structure / coord_mlp_fourier

```yaml
runtime:
  input_mode: table_plus_structure
  structure:
    feature_profile: part_lite_v1
    descriptor_profile: none
    latent_profile: none
    adapter_mode: coord_pack
    provider_mode: parametric_parts

train:
  models: [coord_mlp_fourier]
  coord_mlp_fourier:
    target_family: allvars
    target_vars: [log_ne, Te, phi]
    input_features:
      mode: geom_feature_pack
      profile: part_lite_v1
      require_pack: error
```

## 4b. table_plus_structure / cno / part_lite_v1

```yaml
runtime:
  input_mode: table_plus_structure
  strict_input_mode: error
  allow_mode_fallback: false
  structure:
    feature_profile: part_lite_v1
    descriptor_profile: none
    latent_profile: none
    adapter_mode: grid_pack
    provider_mode: parametric_parts

model:
  name: cno
  backend: torch
  cno_cfg:
    width: 64
    n_layers: 4
    dropout: 0.0
    kernel_size: 3

preprocessing:
  coord_features:
    channels_from_profile: part_lite_v1

train:
  cno:
    target_family: allvars
    input_features:
      mode: geom_feature_pack
      profile: part_lite_v1
      require_pack: error
```

## 4c. table_plus_structure / geom_deeponet_siren / hybrid_pack_descriptor

```yaml
runtime:
  input_mode: table_plus_structure
  strict_input_mode: error
  allow_mode_fallback: false
  structure:
    feature_profile: part_lite_v1
    descriptor_profile: struct_desc_v1
    latent_profile: none
    adapter_mode: hybrid_pack_descriptor
    provider_mode: parametric_parts

model:
  name: geom_deeponet_siren
  backend: torch
  geom_deeponet_siren_cfg:
    latent_dim: 64
    trunk_hidden: 96
    trunk_layers: 3
    branch_hidden: 128
    branch_layers: 2
    dropout: 0.0
    trunk_w0: 20.0

train:
  geom_deeponet_siren:
    target_family: allvars
    input_features:
      mode: geom_feature_pack
      profile: part_lite_v1
      require_pack: error
```

## 5. table_plus_structure / deeponet_pod / descriptor

```yaml
runtime:
  input_mode: table_plus_structure
  structure:
    feature_profile: boundary_plus_v1
    descriptor_profile: struct_desc_v1
    latent_profile: none
    adapter_mode: descriptor_branch
    provider_mode: parametric_parts

train:
  models: [deeponet_pod]
  deeponet_pod:
    target_family: allvars
    target_vars: [log_ne, Te, phi]
    input_features:
      mode: descriptor_pack
      profile: struct_desc_v1
```

## 実装上の補足

- `target_vars` の例はサンプル。実コードでは `output_layout.vars` を真実源とする。
- `profile` は registry から解決する。YAML でチャネル列を直書きしない。

## Phase 5 Addendum (Latent Hook Contract)

- When `runtime.structure.latent_profile != none` (v1 hook-only), provide external latent artifact at:
  - `dataset/geometry/latent_feature_pack.npz`
- Preprocess copies it to:
  - `preprocessing/features/latent_feature_pack.npz`
- Missing latent artifact is fail-fast.
