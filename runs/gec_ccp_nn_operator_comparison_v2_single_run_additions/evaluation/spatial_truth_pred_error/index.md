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
| `deeponet_plasma` | `interp` | 13 | 0.1399 | 0.1266 | 0.2109 | 0.1302 | 0.1519 |
| `cno` | `interp` | 13 | 0.0415 | 0.0412 | 0.0517 | 0.0254 | 0.0400 |
| `unetpp` | `interp` | 13 | 0.0650 | 0.0679 | 0.0489 | 0.0351 | 0.0542 |

## モデル別ページ

- [deeponet_plasma](model_summaries/deeponet_plasma.md) — representative seed `412`; interp × 4物性 × 5代表ケース
- [cno](model_summaries/cno.md) — representative seed `412`; interp × 4物性 × 5代表ケース
- [unetpp](model_summaries/unetpp.md) — representative seed `412`; interp × 4物性 × 5代表ケース

## 完全性

- requested / plotted models: `3` / `3`
- all-case metric rows: `39`
- selected model/split/case rows: `15`
- PNG/PDF pairs: `60`
- skipped models: `0`

## CSV

- [model_field_summary.csv](model_field_summary.csv): モデル×split×物性の全ケース集約
- [publication_field_plots.csv](publication_field_plots.csv): 全60代表図のパスとrel.RMSE
- [selected_spatial_plots.csv](selected_spatial_plots.csv): 代表ケース選択
- [spatial_plot_summary.csv](spatial_plot_summary.csv): 全39ケースの物性別誤差
- [skipped_models.csv](skipped_models.csv)

## 解釈上の注意

- 各モデルのmedianケースは誤差順位から個別に選ぶため、モデル間で同一caseとは限りません。
- 空間図はvalidation履歴だけで選んだ代表seedであり、テスト指標によるseed選択はしていません。
- `Td`と構造IDが1対1対応するため、本図だけから構造入力の因果効果や未見構造汎化は評価できません。
