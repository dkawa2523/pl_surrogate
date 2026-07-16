# ffno: spatial truth / prediction / error

[← 全モデルの集約へ](../index.md)

- validation-only representative seed: `412`
- run: `runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/ffno`
- 代表ケースは4物性の平均相対RMSEで best / p25 / median / p75 / worst を選択
- 各図は Structure / Mask、Truth、Prediction、Signed error の順
- Truth と Prediction は同じカラースケール

## interp: marginal補間

Signed error の表示範囲は `±30%`。飽和色はこの値以上の誤差を表します。

| 物性 | ケース数 | mean rel.RMSE | SD | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ne` | 13 | 0.0181 | 0.0064 | 0.0176 | 0.0113 | 0.0323 |
| `ni` | 13 | 0.0180 | 0.0062 | 0.0173 | 0.0105 | 0.0312 |
| `Te` | 13 | 0.0258 | 0.0078 | 0.0246 | 0.0173 | 0.0451 |
| `phi` | 13 | 0.0117 | 0.0020 | 0.0121 | 0.0091 | 0.0146 |

### ne

[![ffno interp ne median](../publication_by_field/ffno/ne/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_ne.png)](../publication_by_field/ffno/ne/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_ne.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0114 | [PNG](../publication_by_field/ffno/ne/interp/best_case_td003_pp0_5_gamma_007__steady_ne.png) / [PDF](../publication_by_field/ffno/ne/interp/best_case_td003_pp0_5_gamma_007__steady_ne.pdf) |
| p25 | `case_td003_pa200_pp0_5_gamma_004__steady` | 0.0130 | [PNG](../publication_by_field/ffno/ne/interp/p25_case_td003_pa200_pp0_5_gamma_004__steady_ne.png) / [PDF](../publication_by_field/ffno/ne/interp/p25_case_td003_pa200_pp0_5_gamma_004__steady_ne.pdf) |
| median | `case_td003_pa200_pp0_1_gamma_010__steady` | 0.0219 | [PNG](../publication_by_field/ffno/ne/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_ne.png) / [PDF](../publication_by_field/ffno/ne/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_ne.pdf) |
| p75 | `case_td030_pp0_1_gamma_004__steady` | 0.0225 | [PNG](../publication_by_field/ffno/ne/interp/p75_case_td030_pp0_1_gamma_004__steady_ne.png) / [PDF](../publication_by_field/ffno/ne/interp/p75_case_td030_pp0_1_gamma_004__steady_ne.pdf) |
| worst | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0323 | [PNG](../publication_by_field/ffno/ne/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_ne.png) / [PDF](../publication_by_field/ffno/ne/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_ne.pdf) |

### ni

[![ffno interp ni median](../publication_by_field/ffno/ni/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_ni.png)](../publication_by_field/ffno/ni/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_ni.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0105 | [PNG](../publication_by_field/ffno/ni/interp/best_case_td003_pp0_5_gamma_007__steady_ni.png) / [PDF](../publication_by_field/ffno/ni/interp/best_case_td003_pp0_5_gamma_007__steady_ni.pdf) |
| p25 | `case_td003_pa200_pp0_5_gamma_004__steady` | 0.0137 | [PNG](../publication_by_field/ffno/ni/interp/p25_case_td003_pa200_pp0_5_gamma_004__steady_ni.png) / [PDF](../publication_by_field/ffno/ni/interp/p25_case_td003_pa200_pp0_5_gamma_004__steady_ni.pdf) |
| median | `case_td003_pa200_pp0_1_gamma_010__steady` | 0.0208 | [PNG](../publication_by_field/ffno/ni/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_ni.png) / [PDF](../publication_by_field/ffno/ni/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_ni.pdf) |
| p75 | `case_td030_pp0_1_gamma_004__steady` | 0.0224 | [PNG](../publication_by_field/ffno/ni/interp/p75_case_td030_pp0_1_gamma_004__steady_ni.png) / [PDF](../publication_by_field/ffno/ni/interp/p75_case_td030_pp0_1_gamma_004__steady_ni.pdf) |
| worst | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0312 | [PNG](../publication_by_field/ffno/ni/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_ni.png) / [PDF](../publication_by_field/ffno/ni/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_ni.pdf) |

### Te

[![ffno interp Te median](../publication_by_field/ffno/Te/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_Te.png)](../publication_by_field/ffno/Te/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_Te.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0173 | [PNG](../publication_by_field/ffno/Te/interp/best_case_td003_pp0_5_gamma_007__steady_Te.png) / [PDF](../publication_by_field/ffno/Te/interp/best_case_td003_pp0_5_gamma_007__steady_Te.pdf) |
| p25 | `case_td003_pa200_pp0_5_gamma_004__steady` | 0.0232 | [PNG](../publication_by_field/ffno/Te/interp/p25_case_td003_pa200_pp0_5_gamma_004__steady_Te.png) / [PDF](../publication_by_field/ffno/Te/interp/p25_case_td003_pa200_pp0_5_gamma_004__steady_Te.pdf) |
| median | `case_td003_pa200_pp0_1_gamma_010__steady` | 0.0182 | [PNG](../publication_by_field/ffno/Te/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_Te.png) / [PDF](../publication_by_field/ffno/Te/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_Te.pdf) |
| p75 | `case_td030_pp0_1_gamma_004__steady` | 0.0296 | [PNG](../publication_by_field/ffno/Te/interp/p75_case_td030_pp0_1_gamma_004__steady_Te.png) / [PDF](../publication_by_field/ffno/Te/interp/p75_case_td030_pp0_1_gamma_004__steady_Te.pdf) |
| worst | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0221 | [PNG](../publication_by_field/ffno/Te/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_Te.png) / [PDF](../publication_by_field/ffno/Te/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_Te.pdf) |

### phi

[![ffno interp phi median](../publication_by_field/ffno/phi/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_phi.png)](../publication_by_field/ffno/phi/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_phi.png)

| 代表位置 | case | rel.RMSE | 図 |
| --- | --- | ---: | --- |
| best | `case_td003_pp0_5_gamma_007__steady` | 0.0091 | [PNG](../publication_by_field/ffno/phi/interp/best_case_td003_pp0_5_gamma_007__steady_phi.png) / [PDF](../publication_by_field/ffno/phi/interp/best_case_td003_pp0_5_gamma_007__steady_phi.pdf) |
| p25 | `case_td003_pa200_pp0_5_gamma_004__steady` | 0.0124 | [PNG](../publication_by_field/ffno/phi/interp/p25_case_td003_pa200_pp0_5_gamma_004__steady_phi.png) / [PDF](../publication_by_field/ffno/phi/interp/p25_case_td003_pa200_pp0_5_gamma_004__steady_phi.pdf) |
| median | `case_td003_pa200_pp0_1_gamma_010__steady` | 0.0136 | [PNG](../publication_by_field/ffno/phi/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_phi.png) / [PDF](../publication_by_field/ffno/phi/interp/median_case_td003_pa200_pp0_1_gamma_010__steady_phi.pdf) |
| p75 | `case_td030_pp0_1_gamma_004__steady` | 0.0122 | [PNG](../publication_by_field/ffno/phi/interp/p75_case_td030_pp0_1_gamma_004__steady_phi.png) / [PDF](../publication_by_field/ffno/phi/interp/p75_case_td030_pp0_1_gamma_004__steady_phi.pdf) |
| worst | `case_td030_pa050_pp0_1_gamma_007__steady` | 0.0139 | [PNG](../publication_by_field/ffno/phi/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_phi.png) / [PDF](../publication_by_field/ffno/phi/interp/worst_case_td030_pa050_pp0_1_gamma_007__steady_phi.pdf) |

