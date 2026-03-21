# 90. Benchmark / Test / Fixture Plan

このファイルは、各モデル仕様から共通化した **実装完了条件** と **fixture の置き方** をまとめる。

## 1. なぜこれが必要か

- repo の benchmark は profile lock と dynamic target 集計を持つ
- compare も dynamic key を保持する
- したがって、新モデル追加時は「学習が動く」だけでは不十分
- build / load / infer / benchmark / compare の動線が必要

## 2. 追加するテストの命名規約

### integration
既存 naming に合わせる。

推奨:
- `tests/integration/test_train_unetpp_smoke.py`
- `tests/integration/test_train_ffno_smoke.py`
- `tests/integration/test_train_coord_mlp_smoke.py`
- `tests/integration/test_train_pod_deeponet_smoke.py`
- `tests/integration/test_infer_unetpp_smoke.py`
- `tests/integration/test_infer_ffno_smoke.py`
- `tests/integration/test_infer_coord_mlp_smoke.py`
- `tests/integration/test_infer_pod_deeponet_smoke.py`

### unit
推奨:
- `tests/unit/models/test_unetpp_model.py`
- `tests/unit/models/test_ffno_model.py`
- `tests/unit/models/test_coord_mlp_model.py`
- `tests/unit/models/test_pod_deeponet_model.py`
- `tests/unit/train/test_model_dispatch_unetpp.py`
- `tests/unit/train/test_model_dispatch_ffno.py`
- `tests/unit/train/test_model_dispatch_coord_mlp.py`
- `tests/unit/train/test_model_dispatch_pod_deeponet.py`

## 3. fixture の置き方

### 原則
- `tests/fixtures` は **テスト用テンプレ**
- 運用で回す YAML 正本は `configs/experimental/same_fidelity_low_data/` 側に置く

### テスト用 fixture 名

#### mainline-candidate
- `benchmark_periodic_real_m7_unetpp_isolated_mainline.yaml`
- `benchmark_periodic_real_m7_unetpp_attn_isolated_mainline.yaml`
- `benchmark_periodic_real_m7_ffno_isolated_mainline.yaml`

#### experimental
- `benchmark_periodic_real_m7_coord_mlp_fourier_experimental.yaml`
- `benchmark_periodic_real_m7_coord_mlp_siren_experimental.yaml`
- `benchmark_periodic_real_m7_deeponet_pod_experimental.yaml`

## 4. benchmark へ入れる順序

### Step 1
model 単体の train/infer smoke を通す。

### Step 2
isolated benchmark fixture を追加する。

### Step 3
必要なら compare fixture を追加する。

### Step 4
mainline-candidate だけ `benchmark/profiles.py` への昇格を検討する。

## 5. compare 追加時の注意

- compare は dynamic key を保持する
- 固定の `test_rmse_phi` のみを基準にしない
- `objective_metric=auto_primary` を使う場合も dynamic target 前提で通ることを確認する
- `global_reference_mode=frozen` を使う場合、`global_mlp` 行が存在する fixture にする

## 6. 推奨 benchmark 順

### Phase 1
- `global_mlp` vs `unet` vs `unetpp`
- `fno` vs `ffno`

### Phase 2
- `global_mlp` vs `coord_mlp_fourier`
- `coord_mlp_fourier` vs `coord_mlp_siren`
- `unetpp` vs `unetpp_attn`

### Phase 3
- `global_mlp` vs `deeponet_pod`
- `deeponet_plasma` vs `deeponet_pod`

## 7. 追加評価指標

既存 benchmark を壊さない前提で、補助 CSV として次の評価を追加してよい。

- boundary band RMSE
- plasma mask 内 RMSE
- plasma mask 外 RMSE
- POD reconstruction error（`deeponet_pod` のみ）
- high-frequency band error（`coord_mlp_fourier`, `coord_mlp_siren`, `ffno` 比較用）

これらは leaderboard の必須列にはしない。

## 8. go / no-go 基準

### mainline-candidate (`unetpp`, `ffno`)
Go:
- strict contract を明示できる
- isolated benchmark が通る
- compare が dynamic target のまま通る

No-Go:
- feature contract が崩れる
- benchmark で固定列を前提にする変更が必要
- `output_layout.vars` と別順で出力している

### experimental (`coord_mlp_*`, `deeponet_pod`)
Go:
- build/load/train/infer が通る
- benchmark を isolated experimental fixture で回せる

No-Go:
- 新 preprocess artifact を大量に追加しないと infer できない（Phase 3 以前）
- 既存 mainline モデルの挙動変更が必要

## 9. Codex 実行単位

Codex には 1 回で全部やらせず、次の単位で指示する。

1. `unetpp` だけ
2. `ffno` だけ
3. `unetpp_attn` だけ
4. `coord_mlp_fourier` だけ
5. `coord_mlp_siren` だけ
6. `deeponet_pod` だけ

各単位で:
- 新 model class
- builder / load / save
- dispatch
- unit / integration smoke
までを完了させる。
