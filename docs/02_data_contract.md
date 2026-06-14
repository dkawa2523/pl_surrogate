# 02 Data Contract

この文書は dataset から preprocessing artifact までの target contract を定義する。

## Target Entry

target 定義の唯一の入口は `dataset.targets[]` である。

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
      units: "m^-3"
      dtype: float32
```

- `id`: repo 内で使う logical target 名。
- `source_key`: source data 側の field 名。省略時は `id` と同じ。
- `role`: physics や role-aware loss が読む target role。
- `positive`: sign penalty や validation に使う metadata。
- `field_family`: density、temperature、electrostatic などの分類。
- `default_region`: loss / metric の既定領域。
- `value_transform`: target ごとの reversible transform。

`dataset.targets[]` の順序は入力定義である。preprocess 後の target 順序の正本は `preprocessing/schema/output_layout.json` の `vars` である。

## Role Schema

preprocess は target metadata を `preprocessing/schema/target_role_schema.json` に保存する。train / infer / benchmark は target 名ではなく、次の順序で physics target を解決する。

1. 明示された `physics.symbols` / `inference.ood.*.symbols`
2. `target_role_schema.json` の一意な `role`
3. `target_role_schema.json` の一意な `field_family`
4. 解決できなければ fail-fast

同じ role / field family に複数 target がある場合は、symbols を明示する。

## Transform Boundary

`value_transform` は一般の reversible transform である。特定 target 種別だけの policy にはしない。物理スケールへの逆変換は `TransformBundle` と preprocessing artifact の責務に閉じる。

## Product Rules

- target 名を physics role の代わりに使わない。
- target 名から transform を推定しない。
- source dataset の field 名を製品 contract として固定しない。
- transform、target order、feature order を train / infer / benchmark で再定義しない。
