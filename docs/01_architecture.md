# 01 Architecture

## 1. システム全体像

`plasma_surrogate` は、カテゴリ単位の処理を組み合わせる pipeline 型基盤です。  
利用者は CLI から stage を選び、各 stage は artifact を残しながら次段へ受け渡されます。

最上位の制御は次の順で流れます。

1. `src/plasma_surrogate/cli/main.py`
   - CLI 引数を受け取る
   - `cleanse`, `feature`, `preprocess`, `train`, `infer`, `evaluate`, `pipeline`, `benchmark run`, `benchmark sweep` を公開する
2. `src/plasma_surrogate/pipeline/task_runner.py`
   - CLI 上の task 名と workflow 実装を結び付ける
3. `src/plasma_surrogate/cli/workflows.py`
   - 各 stage の実処理を呼ぶ
   - artifact manifest を書く
4. 各実装層
   - `core/*`: データ契約、geometry、task spec
   - `preprocessing/*`: split、schema、scaler、feature pack
   - `train/*`: 契約検証、学習実行、loss、physics
   - `infer/*`: 推論、OOD、推論時 physics
   - `eval/*`: metrics 構築
   - `benchmark/*`: 複数 run 集計、leaderboard、resolved 契約出力

## 2. 実行カテゴリの責務

### `cleanse`

- データ監査を行う
- 欠損、shape 不一致、必要列不足、前処理不能な値を見つける
- ここでは学習用の変換や split はまだ固定しない

### `feature`

- geometry / coordinate ベースの特徴量を作る
- `coord_feature_pack` を中心に、grid 上の空間特徴を artifact 化する

### `preprocess`

- split を固定する
- target / condition / coordinate の scaler を fitting する
- `output_layout.vars` を含む schema を固定する
- 以降の学習・推論・評価が使う truth source を生成する

### `train`

- 既に固定された schema と transforms を読み込む
- `model_dispatch` で契約を検証する
- trainer が実際の最適化を行う

### `infer`

- 学習済みモデルと preprocess artifact から予測分布を作る
- 必要なら physics symbol 解決を行う

### `evaluate`

- GT と予測から metrics を作る
- mainline は target 非固定なので、列名は active target に応じて動的に決まる

### `pipeline`

- 複数 stage をまとめて実行する
- 内部 task 名は `pipeline.run`

### `benchmark run`

- protocol、split、leaderboard 集計まで含めて実行する
- `resolved_benchmark.json` を作り、実行時の実効契約を保存する

### `benchmark sweep`

- 複数 benchmark 設定を繰り返し回す

## 3. 通常 pipeline と benchmark の違い

### 通常 pipeline

- 単一設定、単一モデル、単一 run が中心
- 手元検証、単体学習、推論確認に向く

### benchmark

- protocol ごとの比較、leaderboard、compare 用出力が目的
- 単なる学習結果ではなく、比較可能な run summary を作る
- `resolved_benchmark.json` に effective settings を残す

## 4. Artifact 契約

各 stage は `artifacts/<stage>/manifest.json` を出します。  
manifest は `cli/workflows.py` の `_write_stage_manifest(...)` で統一生成されます。

主要キー:

- `stage`
- `run_id`
- `inputs`
- `outputs`
- `upstream_manifests`
- `schema_hash`
- `config_path`

この形式を統一することで、単独実行でも一括実行でも downstream は同じ artifact 契約を読めます。

## 5. 設定がどこで解釈されるか

### CLI

- 何を実行するかだけを決める
- YAML の意味解釈はしない

### Workflow

- stage の順序と artifact の受け渡しを管理する
- 実際のデータ契約・学習契約の解釈は下位層に委譲する

### Preprocess

- split、schema、transforms を固定する
- 学習と推論が共通で使う truth source を作る

### `model_dispatch`

- 学習契約の唯一の入口
- mainline で何が許可され、何が禁止かをここで決める

### trainer / torch_trainer

- validation 済み設定を実行する
- 契約を独自再解釈しない

## 6. Truth source の整理

第三者がコードを追うとき、以下を起点に見ると混乱しません。

- target 定義: `dataset.targets[]`
- target 順序: `output_layout.vars`
- preprocess 契約: `preprocessing.scalers.target_transforms`
- model strict 契約: `train/model_dispatch.py`
- physics term 解決: `train/physics_terms.py`
- inference symbol 解決: `infer/engine.py`
- benchmark 集計対象 target: `target_vars_for_score_effective`

## 7. Mainline モデルの違い

### `global_mlp`

- 入力: 条件ベクトルのみ
- geometry との結び付き: 弱い
- 空間情報: 直接 field を生成
- 向く場面: 基準モデル、条件依存の大局傾向確認

### `unet`

- 入力: 条件ベクトル + geometry/coord feature を持つ grid tensor
- geometry との結び付き: 強い
- 空間情報: 局所畳み込みで扱う
- 向く場面: 境界形状や近傍構造が重要な固定 grid 問題

### `fno`

- 入力: 条件ベクトル + spatial feature
- geometry との結び付き: 中程度から強い
- 空間情報: spectral operator で広域依存を扱う
- 向く場面: fixed grid 上で非局所構造が重要な問題

### `deeponet_plasma`

- 入力:
  - branch: 条件ベクトル
  - trunk: query/geometry feature
- geometry との結び付き: trunk 側で強く扱う
- 空間情報: operator 形で扱う
- 向く場面: 条件依存な operator として分布生成を扱いたい場合

## 8. よく混乱する点

- `dataset.targets[]` はデータ側の logical target 契約
- `target_transforms` は学習前の統計変換契約
- `benchmark profile` は target 名を定義する場所ではない
- m7 fixture の `ne / ni / Te / phi` は実装例であり、基盤自体は target 非固定
