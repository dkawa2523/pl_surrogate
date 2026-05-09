# 00. User View

## まず何を選ぶか

この拡張でユーザーが最初に決めることは、**今回の問題が `table_only` なのか `table_plus_structure` なのか**です。

### `table_only` を選ぶとき
- 入力は process condition や recipe parameter のようなテーブルだけ
- geometry はケース間で固定
- geometry をモデル入力にしたくない
- cond-only baseline を先に作りたい
- 形状最適化は不要

### `table_plus_structure` を選ぶとき
- 構造差が出力に効く
- geometry を feature として使いたい
- SDF / part descriptor / latent を試したい
- 形状も最適化したい

## 重要な誤解防止

`table_only` は「geometry ファイルが存在してはいけない」ではありません。  
`table_only` でも、固定 geometry は **出力復元、mask、評価、physics post-process** のために使ってよいです。  
ただし、それは **学習入力ではない** というのが本仕様です。

## mode ごとの要求ファイル

### `table_only`
必須:
- index CSV
- `dataset.cond_columns`
- target fields
- fixed output context 用 geometry（必要な場合）

禁止:
- structure feature profile を有効化すること
- `geom_space` を指定すること
- structure-aware model を選ぶこと

### `table_plus_structure`
必須:
- index CSV
- `dataset.cond_columns`
- target fields
- geometry root
- `runtime.structure.feature_profile`

任意:
- `runtime.structure.descriptor_profile`
- `runtime.structure.latent_profile`
- `optimize.geom_space`

## 使えるモデル

### `table_only`
- `global_mlp`
- `deeponet_pod`（cond-only / reduced-order として使う場合）
- 将来候補: `tabm_mlp`

### `table_plus_structure`
- `unet`
- `unetpp`
- `unetpp_attn`
- `fno`
- `ffno`
- `coord_mlp_fourier`
- `coord_mlp_siren`
- `deeponet_pod`（descriptor/latent branch を使う場合）
- 実装済み: `u_no`, `cno`
- optional model: `geom_deeponet_siren` (implemented)

## ユーザー向け fail-fast

### よくある誤設定 1
`runtime.input_mode=table_only` なのに `runtime.structure.feature_profile=part_lite_v1`

期待する挙動:
- preprocess / train / infer の入口で即エラー
- 「table_only では structure profile を有効化できない」と明示

### よくある誤設定 2
`runtime.input_mode=table_plus_structure` なのに `global_mlp` を選ぶ

期待する挙動:
- 入口で即エラー
- 「structure mode では structure-aware adapter を持つ model を選べ」と明示
- silent ignore はしない

### よくある誤設定 3
`table_plus_structure` なのに geometry artifact が不足

期待する挙動:
- `geometry provider mode` と `required profiles` を出したうえで不足 artifact を列挙して落とす

## 実行イメージ

### table_only
- `cleanse`
- `feature`
- `preprocess`
- `train` (`global_mlp` or `deeponet_pod`)
- `infer`
- `evaluate`

### table_plus_structure
- `cleanse`
- `feature`
- `preprocess`（geometry core / feature profile / descriptor 作成）
- `train`
- `infer`
- `evaluate`
- `optimize`（必要なら `geom_space` も使う）
