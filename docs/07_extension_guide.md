# 07 Extension Guide

## 1. 基本原則

拡張時は、カテゴリ責務と truth source を壊さないことが最優先です。

壊してはいけない基準:

- target 定義は `dataset.targets[]`
- target 順序は `output_layout.vars`
- 前処理契約は `target_transforms`
- strict validation は `model_dispatch`
- physics 参照は symbol mapping
- compare / benchmark は dynamic target 列

## 2. カテゴリ別の修正入口

### `cleanse`

#### どこを触るか

- `src/plasma_surrogate/core/data_cleaning_audit.py`
- `src/plasma_surrogate/cli/workflows.py`

#### 何を壊しやすいか

- 必須列チェックと target 非固定性の両立
- 監査結果の artifact 出力

#### 追加後に確認するもの

- `artifacts/data_cleaning/manifest.json`
- required output / schema 整合

### `feature`

#### どこを触るか

- `src/plasma_surrogate/preprocessing/runner.py`
- 必要に応じて `src/plasma_surrogate/infer/engine.py`

#### 何を壊しやすいか

- feature channel 順序
- preprocess と infer の feature 契約一致
- `coord_feature_pack_meta.json` の更新漏れ

#### 追加後に確認するもの

- `coord_feature_pack.npz`
- `coord_feature_pack_meta.json`
- `channel_map.json`

### `preprocess`

#### どこを触るか

- `src/plasma_surrogate/preprocessing/scalers.py`
- `src/plasma_surrogate/preprocessing/runner.py`

#### 何を壊しやすいか

- transform と inverse transform の不一致
- target 全数分の transform 定義漏れ
- `distance_transform_stats` を使う runtime 契約との不整合

#### 追加後に確認するもの

- `y_scalers.json`
- `fit_policy.json`
- `output_layout.json`
- `distance_transform_stats.json`

### `train`

#### どこを触るか

- `src/plasma_surrogate/train/model_dispatch.py`
- `src/plasma_surrogate/train/trainer.py`
- `src/plasma_surrogate/train/torch_trainer.py`

#### 何を壊しやすいか

- strict contract と trainer 実装の役割分離
- target 順序と output head の不一致
- selection の target 数依存ロジック

#### 追加後に確認するもの

- mainline strict validation
- selection score
- diagnostics 列

### `infer`

#### どこを触るか

- `src/plasma_surrogate/infer/engine.py`

#### 何を壊しやすいか

- preprocess artifact 依存
- symbol mapping
- geometry feature fallback

#### 追加後に確認するもの

- inference 出力
- symbol 未解決時の fail-fast

### `evaluate`

#### どこを触るか

- `src/plasma_surrogate/eval/metrics_builder.py`
- `src/plasma_surrogate/benchmark/runner.py`
- `scripts/compare_selected_models.py`

#### 何を壊しやすいか

- dynamic target 列
- primary metric の target 解釈
- compare のヘッダ固定化

#### 追加後に確認するもの

- `leaderboard.csv`
- `resolved_benchmark.json`
- `selected_models_comparison.csv`

## 3. feature を追加する方法

1. preprocess で feature 本体を生成する
2. channel metadata を更新する
3. infer 側で同じ channel 名を解決できるようにする
4. model 側の `input_features.features` と一致させる

必須確認:

- channel 順序
- missing feature 時の挙動
- `require_pack=error` で mainline が期待通り止まるか

## 4. transform / scaler を追加する方法

1. `preprocessing/scalers.py` に forward を追加
2. inverse transform を追加
3. validation を追加
4. serialization / reload を揃える

避けるべきこと:

- forward だけ実装して inverse を省略する
- target ごとの例外処理を増やす

## 5. モデルを追加する方法

### 接続点

- `src/plasma_surrogate/models/*`
- `src/plasma_surrogate/models/mlp/io.py`
- `src/plasma_surrogate/train/model_dispatch.py`
- 必要なら `trainer.py` / `torch_trainer.py`
- benchmark に載せるなら profile / fixture / compare

### mainline へ昇格させる条件

- target 非固定契約に従う
- preprocess artifact に依存して一貫した feature 契約を持つ
- compare / benchmark が dynamic target 列のまま解釈できる
- strict validation を `model_dispatch` に定義できる

### 必須テスト観点

- build/load
- train
- infer
- benchmark
- compare

## 6. physics term を追加する方法

### 接続点

- `src/plasma_surrogate/train/physics_terms.py`
- loss helper / composer

### 原則

- registry へ登録する
- target 名直参照をしない
- symbol mapping 前提で解決する

### 必須テスト観点

- alias 解決
- enabled/disabled
- symbol 未解決時の fail-fast

## 7. compare / metrics 拡張時の原則

避けるべき変更:

- target 名の再ハードコード
- fixed compare header の再導入
- `ne / ni / Te / phi` 固定列の復活

守るべきこと:

- `test_rmse_<var>`
- `test_r2_<var>`
- `test_r2_<var>_plasma_*`

のような dynamic pattern を維持する

## 8. よく壊れる箇所

### `dataset.targets[]`

target を変えたのに downstream の `target_vars`, `weights`, `--vars`, `symbols` を変えない

### `output_layout.vars`

preprocess と学習で target 順序がずれる

### symbol mapping

physics が target 名直参照に戻る

### feature 契約

preprocess と infer で channel 集合や順序がずれる

## 9. 拡張時の実務的な順序

1. `dataset.targets[]` と preprocess 契約を先に固定
2. 必要なら feature を追加
3. `model_dispatch` に strict 契約を追加
4. trainer / infer / metrics を接続
5. benchmark / compare / plot を確認

この順序を守ると、設定不整合を早い段階で止めやすくなります。
