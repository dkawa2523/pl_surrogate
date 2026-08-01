# GEC-CCP n78: Truth / Prediction / Spatial error

Validation-onlyで選んだ各モデルについて、物性ごとの真値・予測値・空間誤差をまとめたページです。

- 対象物性: `ne`, `ni`, `Te`, `phi`
- 図の並び: Structure / Mask、Truth、Prediction、Signed error
- Truth / Predictionはケース内で同一カラースケール
- Signed error: `100 * (prediction - truth) / max(|truth|)`（plasma target内）
- 代表ケース: 4物性平均rel.RMSEのbest / p25 / median / p75 / worst
- 誤差色域: interp `±30%`

## 全テストケースの物性別平均相対RMSE

値はvalidation-onlyで選んだ代表seedの全テストケースから計算しています。3 seed平均ではありません。

| モデル | split | cases | `ne` | `ni` | `Te` | `phi` | 4物性平均 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `global_mlp` | `interp` | 13 | 0.3161 | 0.3115 | 0.3408 | 0.1764 | 0.2862 |
| `global_resmlp` | `interp` | 13 | 0.0275 | 0.0274 | 0.0448 | 0.0188 | 0.0296 |
| `global_densemlp` | `interp` | 13 | 0.0414 | 0.0409 | 0.0430 | 0.0249 | 0.0376 |
| `unet` | `interp` | 13 | 0.0424 | 0.0405 | 0.0369 | 0.0238 | 0.0359 |
| `fno` | `interp` | 13 | 0.0266 | 0.0261 | 0.0395 | 0.0187 | 0.0277 |
| `ffno` | `interp` | 13 | 0.0181 | 0.0180 | 0.0258 | 0.0117 | 0.0184 |
| `u_no` | `interp` | 13 | 0.0115 | 0.0111 | 0.0161 | 0.0068 | 0.0114 |
| `deeponet_pod` | `interp` | 13 | 0.0440 | 0.0423 | 0.0661 | 0.0283 | 0.0452 |

## モデル別ページ

- [global_mlp](model_summaries/global_mlp.md) — representative seed `413`; interp × 4物性 × 5代表ケース
- [global_resmlp](model_summaries/global_resmlp.md) — representative seed `413`; interp × 4物性 × 5代表ケース
- [global_densemlp](model_summaries/global_densemlp.md) — representative seed `413`; interp × 4物性 × 5代表ケース
- [unet](model_summaries/unet.md) — representative seed `412`; interp × 4物性 × 5代表ケース
- [fno](model_summaries/fno.md) — representative seed `412`; interp × 4物性 × 5代表ケース
- [ffno](model_summaries/ffno.md) — representative seed `412`; interp × 4物性 × 5代表ケース
- [u_no](model_summaries/u_no.md) — representative seed `413`; interp × 4物性 × 5代表ケース
- [deeponet_pod](model_summaries/deeponet_pod.md) — representative seed `411`; interp × 4物性 × 5代表ケース

## 完全性

- requested / plotted models: `8` / `8`
- all-case metric rows: `104`
- selected model/split/case rows: `40`
- PNG/PDF pairs: `160`
- skipped models: `0`

## CSV

- [model_field_summary.csv](model_field_summary.csv): モデル×split×物性の全ケース集約
- [publication_field_plots.csv](publication_field_plots.csv): 全160代表図のパスとrel.RMSE
- [selected_spatial_plots.csv](selected_spatial_plots.csv): 代表ケース選択
- [spatial_plot_summary.csv](spatial_plot_summary.csv): 全104ケースの物性別誤差
- [skipped_models.csv](skipped_models.csv)

## 解釈上の注意

- 各モデルのmedianケースは誤差順位から個別に選ぶため、モデル間で同一caseとは限りません。
- 空間図はvalidation履歴だけで選んだ代表seedであり、テスト指標によるseed選択はしていません。
- `Td`と構造IDが1対1対応するため、本図だけから構造入力の因果効果や未見構造汎化は評価できません。
