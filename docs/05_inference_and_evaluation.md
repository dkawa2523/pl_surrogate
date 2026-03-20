# 05 Inference And Evaluation

## 1. `infer` の役割

`infer` は、学習済みモデルと preprocess artifact を組み合わせて予測 field を作る stage です。  
中心実装は `src/plasma_surrogate/infer/engine.py` です。

### 主な入力

- checkpoint
- `preprocessing/schema/output_layout.json`
- `preprocessing/scalers/y_scalers.json`
- `preprocessing/features/coord_feature_pack.npz`
- `preprocessing/scalers/distance_transform_stats.json`
- geometry 情報

### 主な出力

- 予測 field
- 物理量スケールへ戻した field
- 必要なら OOD / boundary / optimization 用の補助出力

## 2. `evaluate` の役割

`evaluate` は、推論結果と GT を比較して metrics を作る stage です。  
mainline は target 非固定のため、列名は active target に応じて動的に決まります。

代表的な列:

- `test_rmse_<var>`
- `test_r2_<var>`
- `test_rmse_<var>_plasma`
- `test_r2_<var>_plasma`

## 3. benchmark と通常 evaluate の違い

### 通常 `infer -> evaluate`

- 単一 run の結果確認に向く
- モデル開発、学習成否確認、GT との差の把握が目的

### `benchmark run`

- split や protocol ごとの集計まで含める
- compare 用の leaderboard を作る
- `resolved_benchmark.json` で effective contract を残す

## 4. benchmark の主要生成物

### `leaderboard.csv`

モデルごとの主要指標をまとめた表です。  
mainline では fixed header ではなく、active target 群に応じて列が増減します。

例:

- `test_rmse_density`
- `test_r2_density`
- `test_r2_density_plasma`
- `test_r2_density_plasma_interp`

### `resolved_benchmark.json`

実行時に最終的に使われた契約をまとめた JSON です。  
比較で困ったときは、まずこれを見ます。

重要キー:

- `target_vars_effective`
- `eval.target_vars_for_score_effective`
- `eval.primary_metric_effective`
- モデルごとの `*_feature_contract_effective`

### `selected_models_comparison.csv`

`scripts/compare_selected_models.py` が作る比較表です。  
各 leaderboard から最適行を拾い、動的 target 列を維持したまま 1 表にまとめます。

## 5. dynamic target 列の読み方

mainline compare / benchmark は、active target ごとに列を作ります。

基本列:

- `test_rmse_<var>`
- `test_r2_<var>`
- `test_rmse_<var>_plasma`
- `test_r2_<var>_plasma`

dual-axis benchmark の列:

- `test_r2_<var>_plasma_interp`
- `test_r2_<var>_plasma_extrap`
- `test_r2_<var>_plasma_dual`

### `target_vars_effective` と `target_vars_for_score_effective`

- `target_vars_effective`
  - その run が実際に持っている target 群
- `target_vars_for_score_effective`
  - 集計や primary metric に使う target 群

多くの mainline run では同じですが、評価対象を部分集合にしたい場合は別れます。

## 6. inference 時の physics symbol mapping

推論時 physics は target 名直参照では動きません。  
`InferenceEngine` は symbol resolver を使って必要物理量を引き当てます。

### `inference.ood.physics.symbols`

```yaml
inference:
  ood:
    physics:
      enabled: true
      symbols:
        density: density
        temperature: temperature
        potential: potential
```

### `inference.ood.boundary_operator.symbols`

```yaml
inference:
  ood:
    boundary_operator:
      enabled: true
      symbols:
        density: density
        potential: potential
```

### 何のために必要か

- target 名を dataset ごとに変えても physics を使えるようにする
- `log_ne` や `Te` 固定のような dataset 依存を排除する

### symbol 未解決時の挙動

- `physics.enabled=true` で未解決なら fail-fast
- `boundary_operator.enabled=true` で未解決なら fail-fast

エラーメッセージは `infer/engine.py` 内で明示されています。

## 7. target 名変更時にどこを直すか

target 名を変えたときに見る場所:

1. `dataset.targets[]`
2. `preprocessing.scalers.target_transforms`
3. `train.<model>.target_vars`
4. `benchmark.eval.target_vars_for_score`
5. `physics.symbols`
6. `inference.ood.physics.symbols`
7. plot の `--vars`

## 8. plot スクリプト

空間可視化スクリプトは `scripts/plot_spatial_distribution_summary.py` です。

### mainline 上の前提

- `--vars` は必須
- compare CSV と GT 側に同じ target 列が存在すること
- 暗黙変換はしない
- `ne -> log_ne` のような fallback は mainline では使わない

### どういうときに壊れるか

- compare CSV 側は logical target 名、GT 側は別名のまま
- `dataset.targets[]` を変えたのに `--vars` を古いまま使う

## 9. 修正入口

- inference symbol 解決: `src/plasma_surrogate/infer/engine.py`
- metrics: `src/plasma_surrogate/eval/metrics_builder.py`
- benchmark 集計: `src/plasma_surrogate/benchmark/runner.py`
- compare: `scripts/compare_selected_models.py`
- spatial plot: `scripts/plot_spatial_distribution_summary.py`
