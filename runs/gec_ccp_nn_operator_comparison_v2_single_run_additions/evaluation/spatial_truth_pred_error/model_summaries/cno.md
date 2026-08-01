# cno: spatial truth / prediction / error

[← 全モデルの集約へ](../index.md)

- validation-only representative seed: `412`
- run: `runs/gec_ccp_nn_operator_comparison_v2_single_run_additions/seed_412/n78/cno`
- 代表ケースは4物性の平均相対RMSEで best / p25 / median / p75 / worst を選択
- 各図は Structure / Mask、Truth、Prediction、Signed error の順
- Truth と Prediction は同じカラースケール

## interp: marginal補間

Signed error の表示範囲は `±30%`。飽和色はこの値以上の誤差を表します。

| 物性 | ケース数 | mean rel.RMSE | SD | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ne` | 13 | 0.0415 | 0.0144 | 0.0384 | 0.0239 | 0.0603 |
| `ni` | 13 | 0.0412 | 0.0144 | 0.0366 | 0.0245 | 0.0599 |
| `Te` | 13 | 0.0517 | 0.0172 | 0.0494 | 0.0307 | 0.0776 |
| `phi` | 13 | 0.0254 | 0.0060 | 0.0238 | 0.0163 | 0.0369 |

### ne

[![cno interp ne median](../publication_by_field/cno/ne/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_ne.png)](../publication_by_field/cno/ne/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_ne.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0252 | [PNG](../publication_by_field/cno/ne/interp/best_case_td003_pp0_5_gamma_007__steady_ne.png) / [PDF](../publication_by_field/cno/ne/interp/best_case_td003_pp0_5_gamma_007__steady_ne.pdf) |
| p25 | `case_td003_pa050_pp0_3_gamma_010__steady` | 0.0384 | [PNG](../publication_by_field/cno/ne/interp/p25_case_td003_pa050_pp0_3_gamma_010__steady_ne.png) / [PDF](../publication_by_field/cno/ne/interp/p25_case_td003_pa050_pp0_3_gamma_010__steady_ne.pdf) |
| median | `case_td030_pa200_pp0_5_gamma_004__steady` | 0.0239 | [PNG](../publication_by_field/cno/ne/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_ne.png) / [PDF](../publication_by_field/cno/ne/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_ne.pdf) |
| p75 | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0586 | [PNG](../publication_by_field/cno/ne/interp/p75_case_td030_pa050_pp0_1_gamma_007__steady_ne.png) / [PDF](../publication_by_field/cno/ne/interp/p75_case_td030_pa050_pp0_1_gamma_007__steady_ne.pdf) |
| worst | `case_td030_pa200_pp0_1_gamma_004__steady` | 0.0603 | [PNG](../publication_by_field/cno/ne/interp/worst_case_td030_pa200_pp0_1_gamma_004__steady_ne.png) / [PDF](../publication_by_field/cno/ne/interp/worst_case_td030_pa200_pp0_1_gamma_004__steady_ne.pdf) |

### ni

[![cno interp ni median](../publication_by_field/cno/ni/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_ni.png)](../publication_by_field/cno/ni/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_ni.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0248 | [PNG](../publication_by_field/cno/ni/interp/best_case_td003_pp0_5_gamma_007__steady_ni.png) / [PDF](../publication_by_field/cno/ni/interp/best_case_td003_pp0_5_gamma_007__steady_ni.pdf) |
| p25 | `case_td003_pa050_pp0_3_gamma_010__steady` | 0.0366 | [PNG](../publication_by_field/cno/ni/interp/p25_case_td003_pa050_pp0_3_gamma_010__steady_ni.png) / [PDF](../publication_by_field/cno/ni/interp/p25_case_td003_pa050_pp0_3_gamma_010__steady_ni.pdf) |
| median | `case_td030_pa200_pp0_5_gamma_004__steady` | 0.0245 | [PNG](../publication_by_field/cno/ni/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_ni.png) / [PDF](../publication_by_field/cno/ni/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_ni.pdf) |
| p75 | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0599 | [PNG](../publication_by_field/cno/ni/interp/p75_case_td030_pa050_pp0_1_gamma_007__steady_ni.png) / [PDF](../publication_by_field/cno/ni/interp/p75_case_td030_pa050_pp0_1_gamma_007__steady_ni.pdf) |
| worst | `case_td030_pa200_pp0_1_gamma_004__steady` | 0.0580 | [PNG](../publication_by_field/cno/ni/interp/worst_case_td030_pa200_pp0_1_gamma_004__steady_ni.png) / [PDF](../publication_by_field/cno/ni/interp/worst_case_td030_pa200_pp0_1_gamma_004__steady_ni.pdf) |

### Te

[![cno interp Te median](../publication_by_field/cno/Te/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_Te.png)](../publication_by_field/cno/Te/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_Te.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0354 | [PNG](../publication_by_field/cno/Te/interp/best_case_td003_pp0_5_gamma_007__steady_Te.png) / [PDF](../publication_by_field/cno/Te/interp/best_case_td003_pp0_5_gamma_007__steady_Te.pdf) |
| p25 | `case_td003_pa050_pp0_3_gamma_010__steady` | 0.0307 | [PNG](../publication_by_field/cno/Te/interp/p25_case_td003_pa050_pp0_3_gamma_010__steady_Te.png) / [PDF](../publication_by_field/cno/Te/interp/p25_case_td003_pa050_pp0_3_gamma_010__steady_Te.pdf) |
| median | `case_td030_pa200_pp0_5_gamma_004__steady` | 0.0776 | [PNG](../publication_by_field/cno/Te/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_Te.png) / [PDF](../publication_by_field/cno/Te/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_Te.pdf) |
| p75 | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0494 | [PNG](../publication_by_field/cno/Te/interp/p75_case_td030_pa050_pp0_1_gamma_007__steady_Te.png) / [PDF](../publication_by_field/cno/Te/interp/p75_case_td030_pa050_pp0_1_gamma_007__steady_Te.pdf) |
| worst | `case_td030_pa200_pp0_1_gamma_004__steady` | 0.0726 | [PNG](../publication_by_field/cno/Te/interp/worst_case_td030_pa200_pp0_1_gamma_004__steady_Te.png) / [PDF](../publication_by_field/cno/Te/interp/worst_case_td030_pa200_pp0_1_gamma_004__steady_Te.pdf) |

### phi

[![cno interp phi median](../publication_by_field/cno/phi/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_phi.png)](../publication_by_field/cno/phi/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_phi.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0163 | [PNG](../publication_by_field/cno/phi/interp/best_case_td003_pp0_5_gamma_007__steady_phi.png) / [PDF](../publication_by_field/cno/phi/interp/best_case_td003_pp0_5_gamma_007__steady_phi.pdf) |
| p25 | `case_td003_pa050_pp0_3_gamma_010__steady` | 0.0218 | [PNG](../publication_by_field/cno/phi/interp/p25_case_td003_pa050_pp0_3_gamma_010__steady_phi.png) / [PDF](../publication_by_field/cno/phi/interp/p25_case_td003_pa050_pp0_3_gamma_010__steady_phi.pdf) |
| median | `case_td030_pa200_pp0_5_gamma_004__steady` | 0.0238 | [PNG](../publication_by_field/cno/phi/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_phi.png) / [PDF](../publication_by_field/cno/phi/interp/median_case_td030_pa200_pp0_5_gamma_004__steady_phi.pdf) |
| p75 | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0311 | [PNG](../publication_by_field/cno/phi/interp/p75_case_td030_pa050_pp0_1_gamma_007__steady_phi.png) / [PDF](../publication_by_field/cno/phi/interp/p75_case_td030_pa050_pp0_1_gamma_007__steady_phi.pdf) |
| worst | `case_td030_pa200_pp0_1_gamma_004__steady` | 0.0369 | [PNG](../publication_by_field/cno/phi/interp/worst_case_td030_pa200_pp0_1_gamma_004__steady_phi.png) / [PDF](../publication_by_field/cno/phi/interp/worst_case_td030_pa200_pp0_1_gamma_004__steady_phi.pdf) |

