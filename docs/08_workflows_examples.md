# 08 Workflow Examples

## 1. この章の位置付け

ここでは、現行 mainline 実装と一致する実行例を示します。  
`tests/fixtures` はテンプレであり、実運用の正本は将来的に `configs/` へ置く想定です。

この章で使う m7 fixture は、periodic dataset の例です。  
`ne / ni / Te / phi` はあくまで例であり、別データセットでは `dataset.targets[]` を差し替えます。

## 2. benchmark 実行例

### global frozen reference

```bash
PLASMA_SURROGATE_ENABLE_TORCH=1 PYTHONPATH=src .venv-torch/bin/python \
  -m plasma_surrogate.cli.main benchmark run \
  --config tests/fixtures/benchmark_periodic_real_m7_global_frozen_ref.yaml
```

何が起きるか:

- global baseline を frozen reference として評価する
- leaderboard と resolved benchmark を出す

主に確認する生成物:

- `leaderboard.csv`
- `resolved_benchmark.json`

### UNet mainline

```bash
PLASMA_SURROGATE_ENABLE_TORCH=1 PYTHONPATH=src .venv-torch/bin/python \
  -m plasma_surrogate.cli.main benchmark run \
  --config tests/fixtures/benchmark_periodic_real_m7_unet_isolated_mainline.yaml
```

何が起きるか:

- UNet 単独の dual-axis benchmark を回す
- preprocess artifact を使って `geom_feature_pack` 入力を構築する

### FNO mainline

```bash
PLASMA_SURROGATE_ENABLE_TORCH=1 PYTHONPATH=src .venv-torch/bin/python \
  -m plasma_surrogate.cli.main benchmark run \
  --config tests/fixtures/benchmark_periodic_real_m7_fno_isolated_mainline.yaml
```

何が起きるか:

- FNO 単独の mainline benchmark を回す
- spectral config と dynamic target score で集計する

### DeepONet mainline

```bash
PLASMA_SURROGATE_ENABLE_TORCH=1 PYTHONPATH=src .venv-torch/bin/python \
  -m plasma_surrogate.cli.main benchmark run \
  --config tests/fixtures/benchmark_periodic_real_m7_deeponet_isolated_mainline.yaml
```

何が起きるか:

- DeepONet plain mainline を `cond_only + geom_feature_pack` 契約で回す

## 3. compare 実行例

```bash
PYTHONPATH=src .venv-torch/bin/python scripts/compare_selected_models.py \
  --config tests/fixtures/benchmark_periodic_real_compare_mainline_fno_deeponet_global.yaml
```

何が起きるか:

- 各 benchmark の leaderboard から代表行を拾う
- 1 枚の `selected_models_comparison.csv` にまとめる

主な生成物:

- `selected_models_comparison.csv`

確認ポイント:

- 列が fixed header ではなく target 依存で増減していること
- `model_id`, `primary_metric`, `primary_metric_value` が意図通りであること

## 4. spatial plot 実行例

`--vars` は必須です。  
target 名は compare CSV と GT 側の論理列名に一致させてください。

```bash
PYTHONPATH=src .venv-torch/bin/python scripts/plot_spatial_distribution_summary.py \
  --compare-csv runs/periodic_real_tuned_v83/compare_mainline_fno_deeponet_global/selected_models_comparison.csv \
  --out-dir runs/periodic_real_tuned_v83/compare_mainline_fno_deeponet_global/plots_interp \
  --protocol interp \
  --vars ne ni Te phi \
  --include-ground-truth \
  --reference-name global_v2_frozen_ref
```

何が起きるか:

- compare CSV を基にモデルごとの分布を再読込する
- GT と並べた summary plot を出す

確認ポイント:

- `--vars` と compare / GT 列が一致していること
- implicit fallback を前提にしていないこと

## 5. pipeline 実行例

CLI コマンドは `pipeline` です。内部 task 名は `pipeline.run` です。

```bash
PYTHONPATH=src .venv-torch/bin/python -m plasma_surrogate.cli.main pipeline \
  --config configs/mainline.yaml
```

何が起きるか:

- config の stage 構成に従ってまとめて実行する
- 各 stage が個別 manifest を残す

向いている用途:

- 本番想定の一括実行
- 単一 YAML を入口にした再現運用

## 6. カテゴリ単独実行例

```bash
PYTHONPATH=src .venv-torch/bin/python -m plasma_surrogate.cli.main cleanse --config configs/mainline.yaml
PYTHONPATH=src .venv-torch/bin/python -m plasma_surrogate.cli.main feature --config configs/mainline.yaml
PYTHONPATH=src .venv-torch/bin/python -m plasma_surrogate.cli.main preprocess --config configs/mainline.yaml
PYTHONPATH=src .venv-torch/bin/python -m plasma_surrogate.cli.main train --config configs/mainline.yaml
PYTHONPATH=src .venv-torch/bin/python -m plasma_surrogate.cli.main infer --config configs/mainline.yaml
PYTHONPATH=src .venv-torch/bin/python -m plasma_surrogate.cli.main evaluate --config configs/mainline.yaml
```

使い分け:

- `cleanse`: データ監査だけしたい
- `feature`: feature pack の生成だけ確認したい
- `preprocess`: split/scaler/schema を固定したい
- `train`: 同じ preprocess artifact で複数モデルを試したい
- `infer`: 既存 checkpoint の予測だけ取りたい
- `evaluate`: GT との差分だけ再計算したい

## 7. 生成物の確認場所

### benchmark

- `runs/.../leaderboard.csv`
- `runs/.../resolved_benchmark.json`
- `runs/.../manifest.json`

### compare

- `runs/.../selected_models_comparison.csv`

### pipeline / stage

- `runs/.../artifacts/<stage>/manifest.json`

### train / infer / eval の詳細

- checkpoint
- per-stage metrics
- inference result
- spatial evaluation summary

は run 配下の各 stage 出力を見る

## 8. `tests/fixtures` と `configs/` の考え方

### `tests/fixtures`

- 再現可能なテンプレ
- テストや benchmark の実体例
- 現在の mainline 実装がどう使われているかのサンプル

### `configs/`

- 実運用の正本を置く場所として想定
- dataset 固有値、運用用 output dir、評価条件はここへ切り出すべき

## 9. m7 から別データセットへ移るときに差し替える場所

最低限見直す箇所:

1. `dataset.targets[]`
2. `dataset.cond_columns`
3. `dataset.geometry_root`
4. `preprocessing.scalers.target_transforms`
5. `train.<model>.target_vars`
6. `benchmark.eval.target_vars_for_score`
7. 必要なら `physics.symbols`
8. plot の `--vars`

そのまま流用しやすいもの:

- CLI 実行形
- stage 構成
- artifact 契約
- compare 実行手順

## 10. 実運用での推奨順序

1. `cleanse`
2. `feature`
3. `preprocess`
4. `train`
5. `infer`
6. `evaluate`
7. 必要なら `benchmark run` と `compare`

初回導入時は、pipeline 一括実行よりカテゴリ単独実行で contract を確認しながら進めた方が安全です。
