# unet: spatial truth / prediction / error

[← 全モデルの集約へ](../index.md)

- validation-only representative seed: `412`
- run: `runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/unet`
- 代表ケースは4物性の平均相対RMSEで best / p25 / median / p75 / worst を選択
- 各図は Structure / Mask、Truth、Prediction、Signed error の順
- Truth と Prediction は同じカラースケール

## interp: marginal補間

Signed error の表示範囲は `±30%`。飽和色はこの値以上の誤差を表します。

| 物性 | ケース数 | mean rel.RMSE | SD | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ne` | 13 | 0.0424 | 0.0156 | 0.0401 | 0.0245 | 0.0679 |
| `ni` | 13 | 0.0405 | 0.0145 | 0.0361 | 0.0229 | 0.0604 |
| `Te` | 13 | 0.0369 | 0.0078 | 0.0356 | 0.0251 | 0.0530 |
| `phi` | 13 | 0.0238 | 0.0043 | 0.0235 | 0.0186 | 0.0327 |

### ne

[![unet interp ne median](../publication_by_field/unet/ne/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_ne.png)](../publication_by_field/unet/ne/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_ne.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0252 | [PNG](../publication_by_field/unet/ne/interp/best_case_td003_pp0_5_gamma_007__steady_ne.png) / [PDF](../publication_by_field/unet/ne/interp/best_case_td003_pp0_5_gamma_007__steady_ne.pdf) |
| p25 | `case_td003_pp0_3_gamma_004__steady` | 0.0318 | [PNG](../publication_by_field/unet/ne/interp/p25_case_td003_pp0_3_gamma_004__steady_ne.png) / [PDF](../publication_by_field/unet/ne/interp/p25_case_td003_pp0_3_gamma_004__steady_ne.pdf) |
| median | `case_td003_pa050_pp0_3_gamma_010__steady` | 0.0401 | [PNG](../publication_by_field/unet/ne/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_ne.png) / [PDF](../publication_by_field/unet/ne/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_ne.pdf) |
| p75 | `case_td030_pp0_1_gamma_004__steady` | 0.0591 | [PNG](../publication_by_field/unet/ne/interp/p75_case_td030_pp0_1_gamma_004__steady_ne.png) / [PDF](../publication_by_field/unet/ne/interp/p75_case_td030_pp0_1_gamma_004__steady_ne.pdf) |
| worst | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0679 | [PNG](../publication_by_field/unet/ne/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_ne.png) / [PDF](../publication_by_field/unet/ne/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_ne.pdf) |

### ni

[![unet interp ni median](../publication_by_field/unet/ni/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_ni.png)](../publication_by_field/unet/ni/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_ni.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0229 | [PNG](../publication_by_field/unet/ni/interp/best_case_td003_pp0_5_gamma_007__steady_ni.png) / [PDF](../publication_by_field/unet/ni/interp/best_case_td003_pp0_5_gamma_007__steady_ni.pdf) |
| p25 | `case_td003_pp0_3_gamma_004__steady` | 0.0305 | [PNG](../publication_by_field/unet/ni/interp/p25_case_td003_pp0_3_gamma_004__steady_ni.png) / [PDF](../publication_by_field/unet/ni/interp/p25_case_td003_pp0_3_gamma_004__steady_ni.pdf) |
| median | `case_td003_pa050_pp0_3_gamma_010__steady` | 0.0361 | [PNG](../publication_by_field/unet/ni/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_ni.png) / [PDF](../publication_by_field/unet/ni/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_ni.pdf) |
| p75 | `case_td030_pp0_1_gamma_004__steady` | 0.0504 | [PNG](../publication_by_field/unet/ni/interp/p75_case_td030_pp0_1_gamma_004__steady_ni.png) / [PDF](../publication_by_field/unet/ni/interp/p75_case_td030_pp0_1_gamma_004__steady_ni.pdf) |
| worst | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0604 | [PNG](../publication_by_field/unet/ni/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_ni.png) / [PDF](../publication_by_field/unet/ni/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_ni.pdf) |

### Te

[![unet interp Te median](../publication_by_field/unet/Te/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_Te.png)](../publication_by_field/unet/Te/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_Te.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0251 | [PNG](../publication_by_field/unet/Te/interp/best_case_td003_pp0_5_gamma_007__steady_Te.png) / [PDF](../publication_by_field/unet/Te/interp/best_case_td003_pp0_5_gamma_007__steady_Te.pdf) |
| p25 | `case_td003_pp0_3_gamma_004__steady` | 0.0349 | [PNG](../publication_by_field/unet/Te/interp/p25_case_td003_pp0_3_gamma_004__steady_Te.png) / [PDF](../publication_by_field/unet/Te/interp/p25_case_td003_pp0_3_gamma_004__steady_Te.pdf) |
| median | `case_td003_pa050_pp0_3_gamma_010__steady` | 0.0286 | [PNG](../publication_by_field/unet/Te/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_Te.png) / [PDF](../publication_by_field/unet/Te/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_Te.pdf) |
| p75 | `case_td030_pp0_1_gamma_004__steady` | 0.0417 | [PNG](../publication_by_field/unet/Te/interp/p75_case_td030_pp0_1_gamma_004__steady_Te.png) / [PDF](../publication_by_field/unet/Te/interp/p75_case_td030_pp0_1_gamma_004__steady_Te.pdf) |
| worst | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0333 | [PNG](../publication_by_field/unet/Te/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_Te.png) / [PDF](../publication_by_field/unet/Te/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_Te.pdf) |

### phi

[![unet interp phi median](../publication_by_field/unet/phi/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_phi.png)](../publication_by_field/unet/phi/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_phi.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0188 | [PNG](../publication_by_field/unet/phi/interp/best_case_td003_pp0_5_gamma_007__steady_phi.png) / [PDF](../publication_by_field/unet/phi/interp/best_case_td003_pp0_5_gamma_007__steady_phi.pdf) |
| p25 | `case_td003_pp0_3_gamma_004__steady` | 0.0190 | [PNG](../publication_by_field/unet/phi/interp/p25_case_td003_pp0_3_gamma_004__steady_phi.png) / [PDF](../publication_by_field/unet/phi/interp/p25_case_td003_pp0_3_gamma_004__steady_phi.pdf) |
| median | `case_td003_pa050_pp0_3_gamma_010__steady` | 0.0233 | [PNG](../publication_by_field/unet/phi/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_phi.png) / [PDF](../publication_by_field/unet/phi/interp/median_case_td003_pa050_pp0_3_gamma_010__steady_phi.pdf) |
| p75 | `case_td030_pp0_1_gamma_004__steady` | 0.0247 | [PNG](../publication_by_field/unet/phi/interp/p75_case_td030_pp0_1_gamma_004__steady_phi.png) / [PDF](../publication_by_field/unet/phi/interp/p75_case_td030_pp0_1_gamma_004__steady_phi.pdf) |
| worst | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0296 | [PNG](../publication_by_field/unet/phi/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_phi.png) / [PDF](../publication_by_field/unet/phi/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_phi.pdf) |

