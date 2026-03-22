# 20. Geometry Core and Feature Lanes

## 基本方針

構造特徴量は model ごとに個別生成しない。  
まず `Geometry Core` を作り、その上に `feature lane` を作る。

## Geometry Core の正本 artifact

### table_only
`Geometry Core` は output context としてだけ使う。  
v1 では既存固定 geometry のままでよい。

最低限:
- `mask_plasma.npy`
- `distance_signed.npy`
- `distance_any.npy`
- `coord_grid.npy` もしくは fallback 生成可能な情報

### table_plus_structure
`Geometry Core` は model input の元になる。

追加:
- `parts_manifest.json`
- `parts_pack.npz`
- `sdf_union_solid.npy`
- `geometry_core_hash.json`

`parts_pack.npz` は最低でも次を持つ。
- `part_ids`
- `part_roles`
- `mask_stack`
- `sdf_stack`

optional:
- `normal_x_stack`
- `normal_y_stack`
- `curvature_stack`

## feature lanes

### Grid lane
対象:
- `unet`
- `unetpp`
- `unetpp_attn`
- `fno`
- `ffno`
- 将来 `u_no`, `cno`

内容:
- fixed-size spatial channels
- 例: `x, y, mask_plasma, distance_signed, distance_any`
- structure mode では profile に応じて additional channels を加える

### Coord lane
対象:
- `coord_mlp_fourier`
- `coord_mlp_siren`
- 将来 `geom_deeponet_siren`

内容:
- query-point と同じ座標系で評価できる channels
- SDF 系と相性がよい

### Descriptor lane
対象:
- `deeponet_pod`
- grid model の FiLM conditioning
- future reduced-order models

内容:
- per-part / system-level の低次元特徴
- fixed length vector

### Latent lane
対象:
- future shape encoder
- `deeponet_pod` の branch augment
- `geom_deeponet_siren` の branch augment

v1 では metadata と hook のみ入れ、学習済み latent encoder は optional にする。

## feature profile

### `geom_v1_mainline`
既存 mainline 互換 profile
- `x`
- `y`
- `mask_plasma`
- `distance_signed`
- `distance_any`

### `boundary_plus_v1`
boundary 近傍の情報を強める profile
- `geom_v1_mainline`
- `normal_x`
- `normal_y`
- `curvature_proxy`

### `part_lite_v1`
part 数に依らず固定次元で運用する profile
- `boundary_plus_v1`
- `sdf_union_solid`
- `sdf_nearest_part`
- `sdf_second_part`
- `solid_proximity`
- `part_separation_proxy`

### `part_semantic_v1`
part の意味が固定している場合のみ使う
- `part_role.<role>.sdf`
- `part_role.<role>.normal_x`
- `part_role.<role>.normal_y`

v1 では role 数が安定しない案件に使わない。

## descriptor profile

### `struct_desc_v1`
低次元で保守的な descriptor
- `n_parts`
- `solid_area_frac`
- `plasma_area_frac`
- `min_part_gap`
- `mean_part_gap`
- `mean_distance_to_plasma_boundary`
- per-part:
  - `area_frac`
  - `centroid_x`
  - `centroid_y`
  - `bbox_w`
  - `bbox_h`
  - `principal_angle`
  - `perimeter`
  - `min_gap_to_plasma`
  - `min_gap_to_other_parts`

pairwise の全展開はしない。  
必要なら summary 統計へ落とす。

## latent profile

### `shape_ae_v1`
union SDF 由来の `z_geom`

### `part_latent_v1`
part-aware latent の `z_part_i`

v1 の実装では、まず metadata と adapter 受け口だけを定義する。
