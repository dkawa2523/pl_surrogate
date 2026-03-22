# 70. Benchmark, Tests, and Rollout

## benchmark の扱い

benchmark row に少なくとも以下を追加する。

- `input_mode_effective`
- `structure_feature_profile_effective`
- `structure_descriptor_profile_effective`
- `structure_latent_profile_effective`
- `structure_adapter_mode_effective`
- `geometry_provider_mode_effective`

### 比較ルール
- 既定では `table_only` と `table_plus_structure` を混ぜない
- `compare.require_same_input_mode = true` を default にする
- global reference も同じ input_mode のものを優先する

## 追加する評価列

### 共通
- `primary_metric_effective`
- `input_mode_effective`

### structure mode でのみ
- `part_boundary_band_rmse`
- `critical_gap_band_rmse`
- `structure_profile_hash`
- `geometry_core_hash`

## テスト方針

### unit
- input mode 正規化
- mode × model policy validator
- structure profile registry
- table_only で structure profile を reject
- table_plus_structure で profile 不足を reject

### integration
- table_only / global_mlp train smoke
- table_only / deeponet_pod train+infer smoke
- table_plus_structure / ffno train+infer smoke
- table_plus_structure / coord_mlp_fourier smoke
- table_plus_structure / deeponet_pod descriptor smoke
- optimize with geom_space smoke

## rollout 順

1. mode contract + metadata
2. table_only cleanup
3. structure profile registry
4. dispatch / infer / benchmark integration
5. geom_space
6. descriptor lane
7. future models

## 昇格条件

次の条件を満たした profile / model だけを mainline 候補へ上げる。

- 2 つ以上の data regime で改善
- 3 seeds 以上で一貫
- mode metadata が欠けていない
- compare / benchmark が同一 mode で再現可能
- 運用コストが profile 数の増殖を招かない
