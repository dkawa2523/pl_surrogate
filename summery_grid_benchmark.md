# summery_grid_benchmark

## 実行概要
- データ: `data/outputs_merged_td_csv_periodic_ext0520`
- 比較対象: `unet`, `unetpp`, `unetpp_attn`
- モデル別最適設定:
  - `unet`: `weight_decay=1e-3`（lr既定）
  - `unetpp`: `lr=4e-4`
  - `unetpp_attn`: `lr=4e-4`
- 学習条件（公平化）: `epochs=120`, `batch_size_cases=6`, `effective_steps(interp/extrap)=1080/1080`
- 主要指標: `test_r2_plasma_mean_dual`（高いほど良い）

## 最終比較テーブル

| Rank | Model | Setting | R2 dual | R2 interp | R2 extrap | deep RMSE mean (ne/ni/Te/phi) | Train time [min] |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | unetpp_attn | lr=4e-4 | 0.868001 | 0.909093 | 0.826909 | 4.189e14 | 41.52 |
| 2 | unetpp | lr=4e-4 | 0.867740 | 0.906068 | 0.829411 | 4.384e14 | 26.42 |
| 3 | unet | wd=1e-3 | 0.818275 | 0.888699 | 0.747851 | 4.550e14 | 16.68 |

> 注: `unetpp_attn` と `unetpp` の dual R2 差は非常に小さく（約0.00026）、精度面はほぼ同等です。計算時間を重視するなら `unetpp`、最高R2を優先するなら `unetpp_attn` が選択肢です。

## 空間分布グラフ（更新版）

### interp
- `docs/reports/grid_benchmark_20260331/spatial_fields_interp_case_385ea312f0817d4cbc3a2bb92bcf483d0d694011_with_gt_grid.png`
- `docs/reports/grid_benchmark_20260331/spatial_fields_interp_mean_3cases_with_gt_grid.png`
- `docs/reports/grid_benchmark_20260331/spatial_fields_interp_case_385ea312f0817d4cbc3a2bb92bcf483d0d694011_diff_vs_unetpp_attn_lr4e4.png`
- `docs/reports/grid_benchmark_20260331/spatial_fields_interp_mean_3cases_diff_vs_unetpp_attn_lr4e4.png`

### extrap
- `docs/reports/grid_benchmark_20260331/spatial_fields_extrap_case_8bf07ace38ae8ca4b9f6b2952033257759116be7_with_gt_grid.png`
- `docs/reports/grid_benchmark_20260331/spatial_fields_extrap_mean_3cases_with_gt_grid.png`
- `docs/reports/grid_benchmark_20260331/spatial_fields_extrap_case_8bf07ace38ae8ca4b9f6b2952033257759116be7_diff_vs_unetpp_attn_lr4e4.png`
- `docs/reports/grid_benchmark_20260331/spatial_fields_extrap_mean_3cases_diff_vs_unetpp_attn_lr4e4.png`

## 参照ファイル
- compare設定: `docs/reports/grid_benchmark_20260331/compare_unet_family_final_opt.yaml`
- compare結果CSV: `docs/reports/grid_benchmark_20260331/selected_models_comparison.csv`
- 学習レポート:
  - `runs/ext0520/validation_report_unet_final_opt_wd1e3_e120_b6.json`
  - `runs/ext0520/validation_report_unetpp_family_final_opt_lr4e4_e120_b6.json`

## 短い解釈
- 精度最上位は `unetpp_attn` だが、`unetpp` とほぼ同等。
- `unet` は学習時間は最短だが extrap 側R2が相対的に低く、総合では3位。
- 運用上は「精度最優先: `unetpp_attn`」「時間効率とのバランス: `unetpp`」が妥当。
