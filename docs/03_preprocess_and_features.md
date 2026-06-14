# 03 Preprocess And Features

`preprocess` は train / infer / benchmark が共有する正本 artifact を作る stage である。

## Canonical Artifacts

- target order: `preprocessing/schema/output_layout.json`
- target role metadata: `preprocessing/schema/target_role_schema.json`
- condition schema: `preprocessing/schema/cond_schema.json`
- channel order: `preprocessing/schema/channel_map.json`
- coordinate feature metadata: `preprocessing/features/coord_feature_pack_meta.json`
- target scaler / inverse transform: `preprocessing/scalers/y_scalers.json`
- runtime schema hash: `preprocessing/validation/runtime_schema_hashes.json`

`runtime_schema_hashes.json` は次を保存する。

- `target_schema_hash`
- `feature_schema_hash`

checkpoint / inference request / benchmark row はこの 2 つの hash を同じ metadata contract として持つ。不一致は fail-fast とする。

## Runtime Structure Contract

runtime input mode は 2 種類だけである。

- `table_only`
- `table_plus_structure`

`runtime.structure` の第一級 key は次の 3 つだけである。

- `feature_profile`
- `adapter_mode`
- `provider_mode`

`table_only` は structure profile を使わない。`table_plus_structure` は `feature_profile` を必須にする。

## Feature Contract

feature list は train / infer / benchmark で再定義しない。必ず preprocessing artifact を読む。

- `coord_feature_pack_meta.json`: coordinate feature 名、順序、shape
- `channel_map.json`: model input channel 名、順序、role

descriptor / latent profile は必要なモデルだけの optional metadata であり、runtime input-mode の必須 contract にはしない。

## Transform Contract

target transform は `y_scalers.json` と `TransformBundle` が担当する。model、loss、metric、inference は target 名を見て transform を推定しない。
