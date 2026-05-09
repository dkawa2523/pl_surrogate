# 10. Architecture and Contract

## 新しい top-level ルーティングキー

本拡張では、学習・推論・benchmark の入口で必ず次を解決する。

```yaml
runtime:
  input_mode: table_only | table_plus_structure
  strict_input_mode: error
  allow_mode_fallback: false

  structure:
    feature_profile: none | geom_v1_mainline | boundary_plus_v1 | part_lite_v1 | part_semantic_v1
    descriptor_profile: none | struct_desc_v1
    latent_profile: none | shape_ae_v1 | part_latent_v1
    adapter_mode: auto | none | grid_pack | coord_pack | descriptor_branch | hybrid_pack_descriptor
    provider_mode: fixed | parametric_parts
```

## モードの定義

### `table_only`
- 目的: cond-only あるいは cond-dominant baseline
- 構造データは model input に入れない
- geometry は output context / evaluation context としてのみ使える
- `structure.feature_profile`, `descriptor_profile`, `latent_profile` はすべて `none`
- `geom_space` は禁止

### `table_plus_structure`
- 目的: cond + structure による予測と最適化
- `structure.feature_profile` は必須
- `structure.adapter_mode` は model family と整合している必要がある
- `geom_space` は optional

## Truth source は据え置く

この mode 拡張は target 契約を変えない。

- target 定義: `dataset.targets[].id`
- target 順序: `output_layout.vars`
- transform 契約: `preprocessing.scalers.target_transforms.<var>`

## 新規 registry

### `src/plasma_surrogate/core/input_modes.py`
役割:
- mode 定数
- normalize / validate helper
- checkpoint / benchmark metadata の key 定義

最低限入れるべき定数:
- `INPUT_MODES = ("table_only", "table_plus_structure")`
- `TABLE_ONLY = "table_only"`
- `TABLE_PLUS_STRUCTURE = "table_plus_structure"`

### `src/plasma_surrogate/features/structure_feature_registry.py`
役割:
- 許可チャネル
- feature profile
- descriptor profile
- latent profile の登録と正規化

### `src/plasma_surrogate/core/model_input_policy.py`
役割:
- model family ごとの許可 input_mode
- adapter_mode の許可組み合わせ
- silent ignore を防ぐ validator

## 設計上の厳守事項

- model 名で feature profile を増やさない
- profile は registry で解決し、metadata に effective 値を保存する
- `table_only` と `table_plus_structure` を benchmark で混ぜない
- `auto` は convenience であって silent fallback ではない
- 既存 mainline contract は壊さない

## internal adapter の考え方

ユーザーが見るのは 2 mode だけだが、内部では adapter を持つ。

- `none`
  - cond-only
- `grid_pack`
  - spatial feature channels を grid model に渡す
- `coord_pack`
  - spatial feature channels を coordinate decoder に渡す
- `descriptor_branch`
  - low-dimensional descriptor を branch / cond side に渡す
- `hybrid_pack_descriptor`
  - spatial pack と descriptor の両方を使う

この internal adapter を切り出すことで、mode と model family を直交させる。

## Benchmark Runtime Guardrail (Implemented)

- Benchmark execution enforces strict mode contract checks (`strict_input_mode=error`).
- Benchmark execution forbids metadata fallback (`allow_mode_fallback=false`).
