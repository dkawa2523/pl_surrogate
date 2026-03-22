# 30. Preprocess / Train / Infer Runtime

## Phase 1 の中心目標

- `table_only` を first-class mode にする
- `table_plus_structure` を profile-driven にする
- preprocess / train / infer のすべてが `runtime.input_mode` を見る

## preprocess

### `table_only`
必須:
- cond schema
- axis schema
- output layout
- y scalers
- fixed geometry output context に必要な artifact

禁止:
- structure feature profile の解決
- part pack の作成
- descriptor / latent artifact の作成

保存する metadata:
- `input_mode_effective = "table_only"`
- `structure_feature_profile_effective = "none"`
- `structure_descriptor_profile_effective = "none"`
- `structure_latent_profile_effective = "none"`
- `has_structure_inputs_effective = false`

### `table_plus_structure`
必須:
- Geometry Core の解決
- `structure.feature_profile` の解決
- profile に対応する spatial channels の pack 生成
- descriptor profile が有効なら descriptor table 生成

保存する metadata:
- `input_mode_effective = "table_plus_structure"`
- `structure_feature_profile_effective`
- `structure_descriptor_profile_effective`
- `structure_latent_profile_effective`
- `structure_feature_channels_effective`
- `geometry_provider_mode_effective`
- `geometry_core_hash`
- `has_structure_inputs_effective = true`

## train

train では、model dispatch の前に mode を解決する。

### `table_only`
- structure-aware model を reject
- `global_mlp` / `deeponet_pod` の cond-only path を通す
- `coord_feature_pack` 読み込みはしない
- ただし fixed geometry output context は infer/eval 用に持ってよい

### `table_plus_structure`
- structure feature profile の metadata が必須
- model family に応じて adapter を自動解決または明示指定
- `grid_pack` / `coord_pack` / `descriptor_branch` のいずれかを通す

## infer

### `table_only`
- 推論入力は cond + optional axis
- geometry は fixed output context
- `geom_ref.geom_param` は禁止
- `structure profile` metadata は `none` であること

### `table_plus_structure`
- 推論入力は cond + geometry reference
- `geom_ref` は `geom_id` または `geom_param` を持てる
- provider が Geometry Core を組み直し、profile を使って feature を生成する
- optimize を行うなら `geom_space` を使える

## metadata の必須保存先

- preprocess outputs
- model checkpoint meta
- inference run summary
- benchmark row
- compare intermediate row

## 既存コードで最初に触る場所

- `src/plasma_surrogate/preprocessing/runner.py`
- `src/plasma_surrogate/train/model_dispatch.py`
- `src/plasma_surrogate/infer/engine.py`
- `src/plasma_surrogate/models/mlp/io.py`
- `src/plasma_surrogate/cli/workflows.py`

## fail-fast ルール

- mode と profile が矛盾したら preprocess 前に落とす
- mode と model が矛盾したら train 入口で落とす
- checkpoint meta と infer request が矛盾したら infer 入口で落とす
