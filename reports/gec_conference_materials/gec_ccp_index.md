# GEC-CCP 学会用画像・専用目次

FFNO・DeepONet・Global MLPの共通testケース空間分布と対数学習曲線は、[3モデルの学会用詳細図](model_detail_assets/index.md)にまとめています。

[プロジェクト入口](../../README.md) → [レポート一覧](../index.md) → GEC-CCP学会用画像

GEC-CCPの発表用画像は、主に[`independent_assets/`](independent_assets/index.md)に
集約しています。このページはGEC-ICP素材と混在せず、GEC-CCPだけを目的別に選ぶための
入口です。画像の正本は移動・複製せず、PNG、PDF、SVG、metadataを同じbasenameで管理します。

## 最初に使う図

| 発表内容 | 推奨図 |
|---|---|
| 装置と計算領域 | [ジオメトリー SVG](independent_assets/ccp_geometry_conference.svg) |
| 代表的なプラズマ分布 | [電子密度2D SVG](independent_assets/ccp_electron_density_2d.svg) |
| データセットの応答範囲 | [応答多様性 SVG](independent_assets/ccp_dataset_response_coverage.svg) |
| モデルの値精度と空間形状精度 | [全モデル散布図 SVG](independent_assets/ccp_model_accuracy_scatter.svg) |
| 物性別のモデル精度 | [電子密度–電子温度 R²](independent_assets/ccp_model_r2_ne_vs_te.svg) / [イオン密度–電位 R²](independent_assets/ccp_model_r2_ni_vs_phi.svg) |
| センサー同化・入力最適化 | [multi-seed達成率](optimization_assets/ccp_opt_multiseed_attainment_rate.png) / [TPE trial profile](optimization_assets/ccp_opt_tpe_trial_profiles_all_measurements.png) |

## 4物性の空間分布

通常2D図は定量説明、平面斜視図は表紙・概要向けです。密度は対数色、電子温度と電位は
線形色で表示しています。

| 物理量 | 通常2D | 平面斜視 |
|---|---|---|
| 電子密度 $n_e$ | [PNG](independent_assets/ccp_electron_density_2d.png) / [PDF](independent_assets/ccp_electron_density_2d.pdf) / [SVG](independent_assets/ccp_electron_density_2d.svg) / [metadata](independent_assets/ccp_electron_density_2d_metadata.json) | [PNG](independent_assets/ccp_electron_density_perspective.png) / [PDF](independent_assets/ccp_electron_density_perspective.pdf) / [SVG](independent_assets/ccp_electron_density_perspective.svg) / [metadata](independent_assets/ccp_electron_density_perspective_metadata.json) |
| イオン密度 $n_i$ | [PNG](independent_assets/ccp_ion_density_2d.png) / [PDF](independent_assets/ccp_ion_density_2d.pdf) / [SVG](independent_assets/ccp_ion_density_2d.svg) / [metadata](independent_assets/ccp_ion_density_2d_metadata.json) | [PNG](independent_assets/ccp_ion_density_perspective.png) / [PDF](independent_assets/ccp_ion_density_perspective.pdf) / [SVG](independent_assets/ccp_ion_density_perspective.svg) / [metadata](independent_assets/ccp_ion_density_perspective_metadata.json) |
| 電子温度 $T_e$ | [PNG](independent_assets/ccp_electron_temperature_2d.png) / [PDF](independent_assets/ccp_electron_temperature_2d.pdf) / [SVG](independent_assets/ccp_electron_temperature_2d.svg) / [metadata](independent_assets/ccp_electron_temperature_2d_metadata.json) | [PNG](independent_assets/ccp_electron_temperature_perspective.png) / [PDF](independent_assets/ccp_electron_temperature_perspective.pdf) / [SVG](independent_assets/ccp_electron_temperature_perspective.svg) / [metadata](independent_assets/ccp_electron_temperature_perspective_metadata.json) |
| 電位 $\phi$ | [PNG](independent_assets/ccp_electric_potential_2d.png) / [PDF](independent_assets/ccp_electric_potential_2d.pdf) / [SVG](independent_assets/ccp_electric_potential_2d.svg) / [metadata](independent_assets/ccp_electric_potential_2d_metadata.json) | [PNG](independent_assets/ccp_electric_potential_perspective.png) / [PDF](independent_assets/ccp_electric_potential_perspective.pdf) / [SVG](independent_assets/ccp_electric_potential_perspective.svg) / [metadata](independent_assets/ccp_electric_potential_perspective_metadata.json) |

## 学習モデル別の真値・予測値・空間誤差

学会用の要約散布図だけでなく、各モデル・各物性の空間分布を確認する詳細評価です。

| 対象 | 内容 | 詳細ページ |
|---|---|---|
| 共通比較モデル | Validation-onlyで選択したモデルのTruth / Prediction / Errorと全テスト集約 | [NN / Neural Operator比較](../../runs/gec_ccp_nn_operator_comparison_v1/evaluation/spatial_truth_pred_error/index.md) |
| 採用POD-DeepONet | branch調整後モデルのTruth / Prediction / Error | [POD branch tuned v2](../../runs/gec_ccp_pod_branch_tuned_v2/evaluation/spatial_truth_pred_error/index.md) |
| 単発追加モデル | DeepONet plasma、CNO、U-Net++の追加評価 | [single-run additions](../../runs/gec_ccp_nn_operator_comparison_v2_single_run_additions/evaluation/spatial_truth_pred_error/index.md) |

これらの詳細図はモデルごとに代表ケースが異なる場合があります。モデル間を同じ座標上で
比較するときは、次節の共通テスト集約図とCSVを併用してください。

## モデル比較

| 図 | ファイル |
|---|---|
| 値誤差–空間勾配誤差 | [PNG](independent_assets/ccp_model_accuracy_scatter.png) / [PDF](independent_assets/ccp_model_accuracy_scatter.pdf) / [SVG](independent_assets/ccp_model_accuracy_scatter.svg) |
| 高精度領域拡大 | [PNG](independent_assets/ccp_model_accuracy_scatter_zoom.png) / [PDF](independent_assets/ccp_model_accuracy_scatter_zoom.pdf) / [SVG](independent_assets/ccp_model_accuracy_scatter_zoom.svg) |
| 4物性別の精度 | [PNG](independent_assets/ccp_model_accuracy_by_field.png) / [PDF](independent_assets/ccp_model_accuracy_by_field.pdf) / [SVG](independent_assets/ccp_model_accuracy_by_field.svg) |
| 電子密度–電子温度 R² | [PNG](independent_assets/ccp_model_r2_ne_vs_te.png) / [PDF](independent_assets/ccp_model_r2_ne_vs_te.pdf) / [SVG](independent_assets/ccp_model_r2_ne_vs_te.svg) |
| イオン密度–電位 R² | [PNG](independent_assets/ccp_model_r2_ni_vs_phi.png) / [PDF](independent_assets/ccp_model_r2_ni_vs_phi.pdf) / [SVG](independent_assets/ccp_model_r2_ni_vs_phi.svg) |

数値は[総合・物性別CSV](independent_assets/ccp_model_accuracy_scatter_data.csv)と
[R² CSV](independent_assets/ccp_model_r2_scatter_data.csv)にあります。モデルの特徴と参考文献は
[独立パーツ集のモデル表](independent_assets/index.md#比較対象モデルと位置づけ)を参照してください。

## その他のGEC-CCP素材

| 分類 | ファイル |
|---|---|
| COMSOLジオメトリー・メッシュ | [ジオメトリー](independent_assets/ccp_geometry_conference.svg) / [メッシュ](independent_assets/ccp_mesh_conference.svg) |
| データセット品質 | [応答多様性](independent_assets/ccp_dataset_response_coverage.svg) |
| 前処理 | [Z-score前](independent_assets/ccp_zscore_before.svg) / [Z-score後](independent_assets/ccp_zscore_after.svg) |
| 計算手順・loss | [Z-score計算](ccp_zscore_calculation.svg) / [loss計算](gec_ccp_loss_calculation.svg) / [簡略loss](gec_ccp_loss_design.svg) |
| センサー同化・入力最適化 | [学会用最適化グラフ集](optimization_assets/index.md) |
| 全素材の詳細と選択指針 | [独立パーツ集](independent_assets/index.md) |
| 機械可読カタログ | [asset_catalog.csv](independent_assets/asset_catalog.csv) |

## 再生成

```powershell
.\.venv-test\Scripts\python.exe -W error experiments\conference\scripts\plot_gec_conference_independent_assets.py
.\.venv-test\Scripts\python.exe -W error experiments\conference\scripts\plot_gec_ccp_model_comparison_scatter.py
```

生成元は[`experiments/conference/scripts/`](../../experiments/conference/scripts/)にあります。
