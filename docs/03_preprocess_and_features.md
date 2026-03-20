# 03 Preprocess And Features

## 1. 何を固定する段階か

mainline では `cleanse`, `feature`, `preprocess` はそれぞれ役割が違います。  
3 つをまとめて「前処理」と呼びがちですが、コード上は固定している契約が異なります。

### `cleanse`

- データが学習可能かを監査する段階
- 欠損、shape 不整合、必要列不足、変換不能値を見つける
- ここではまだ学習用の統計値や split は固定しない

### `feature`

- geometry / coordinate 由来の入力特徴を作る段階
- 後段の `unet`, `fno`, `deeponet_plasma` が参照する空間 feature を artifact 化する

### `preprocess`

- split を決める
- schema を決める
- target / cond / coord の transforms と scalers を決める
- train / infer / evaluate が前提とする truth source を固定する

## 2. 前処理契約の中心

正本は以下です。

- `src/plasma_surrogate/preprocessing/runner.py`
- `src/plasma_surrogate/preprocessing/scalers.py`

mainline では旧 `target_transform_policy` は使いません。  
使う契約は `preprocessing.scalers.target_transforms.<var>` のみです。

## 3. `target_transforms` の意味

```yaml
preprocessing:
  scalers:
    enforce_target_transforms: true
    target_transforms:
      density:
        value_transform: log10
        scaler: zscore
        fit_scope: plasma_only
        clip:
          mode: quantile
          q_low: 0.01
          q_high: 0.99
```

ここで指定する変換は、`dataset.targets[]` で logical target にした後、学習へ入る前に適用されます。

## 4. 変換順序

target 変換順序は固定です。

1. `value_transform`
2. `clip`
3. `scaler`

逆変換も固定です。

1. inverse scaler
2. inverse value transform

順序を変えると学習値と推論復元値の意味が崩れるため、ここは変更しない前提で読みます。

## 5. `value_transform`

現在 mainline 前処理で使える値:

- `identity`
- `log10`

### `log10` の利用条件

- target 値が strictly positive であること
- 非正値が混ざると fail-fast する

### どんなときに使うか

- dynamic range が大きく、linear のままでは学習しにくいとき
- 物理量のスケール差を縮めたいとき

### 使わない方がよいケース

- 0 や負値が意味を持つ target
- log 化しても物理的解釈が悪くなる target

## 6. `clip`

現在の clip mode:

- `none`
- `quantile`

### 役割

- 外れ値に対する統計的な切り詰め
- scaler fitting 時の極端な分布の暴れを抑える

### `quantile` の用途

- target の重尾分布や少数の極端値がある場合
- 全削除ではなく、緩やかな切り詰めをしたい場合

## 7. `scaler`

現在の scaler:

- `none`
- `zscore`
- `minmax`

### 選び分けの目安

- `zscore`: 標準的。分布の中心化とスケール調整が必要なとき
- `minmax`: 値域を制限したいとき
- `none`: 既に十分安定したスケールで、変換を加えたくないとき

## 8. `fit_scope`

現在の fit scope:

- `all`
- `plasma_only`

### 意味

どの領域を統計量 fitting の対象にするかを表します。

- `all`: grid 全体
- `plasma_only`: plasma 領域のみ

これは target の値域設計に強く効くため、dataset 変更時に最初に見直すべきキーです。

## 9. preprocess が生成する artifact

`preprocess` は単なる scaler fitting ではなく、下流の契約一式を生成します。

### split 系

- `preprocessing/split/*.json`

役割:

- train / val / test のケース分割を固定する

### schema 系

- `preprocessing/schema/cond_schema.json`
- `preprocessing/schema/axis_schema.json`
- `preprocessing/schema/channel_map.json`
- `preprocessing/schema/output_layout.json`

役割:

- 条件変数順序
- axis 意味
- feature channel 順序
- target 順序

### scaler 系

- `preprocessing/scalers/cond_scaler.json`
- `preprocessing/scalers/y_scalers.json`
- `preprocessing/scalers/coord_scaler.json`
- `preprocessing/scalers/fit_policy.json`
- `preprocessing/scalers/xgrid_channel_scalers.json`
- `preprocessing/scalers/distance_transform_stats.json`
- `preprocessing/scalers/coord_feature_scaler.json`

役割:

- train と infer が同じ変換を使うための統計値

### feature 系

- `preprocessing/features/coord_feature_pack.npz`
- `preprocessing/features/coord_feature_pack_meta.json`

役割:

- geometry / coord feature の本体
- channel 定義、feature pack の構成情報

### stage manifest

- `artifacts/preprocessing/manifest.json`

役割:

- この stage が何を入力にし、何を生成したかを記録する

## 10. `coord_feature_pack` の位置付け

`coord_feature_pack` は、grid 上の空間特徴を preprocess で固めて downstream へ渡す仕組みです。  
mainline の `unet`, `fno`, `deeponet_plasma` は、これを使う構成を強く前提にしています。

代表チャネル:

- `x`
- `y`
- `mask_plasma`
- `distance_signed`
- `distance_any`

追加可能なチャネル:

- `normal_x`
- `normal_y`
- `curvature_proxy`

## 11. runtime 側との接続

### infer との接続

`InferenceEngine` は preprocess artifact を読み込みます。

- `coord_feature_pack`
- `coord_distance_transform_stats`
- schema
- scaler

このため、preprocess と infer が別の feature 契約を想定すると壊れます。

### `bounded_auto` との関係

runtime で `distance_transform.mode=bounded_auto` を使う場合、preprocess の `distance_transform_stats` が参照されます。  
つまり distance 系の振る舞いは preprocess と infer の共同契約です。

## 12. 新しい特徴量を追加したときに変わるもの

新しい geometry/coord feature を追加すると、少なくとも次が変わります。

- `coord_feature_pack.npz`
- `coord_feature_pack_meta.json`
- `channel_map.json`
- `coord_feature_scaler.json`
- 必要に応じて infer 側の feature 解決

確認ポイント:

- channel 順序が train / infer で一致しているか
- `require_pack=error` の mainline 契約を壊していないか
- new feature が absence fallback に依存していないか

## 13. よくある失敗

- `target_transforms` が target 全数分定義されていない
- dataset 側 `value_transform` と preprocess 側 `value_transform` の役割を混同する
- `coord_feature_pack` を作ったのに model 側で別チャネル想定にしてしまう
- `distance_transform_stats` を期待する runtime 設定なのに preprocess で出していない
