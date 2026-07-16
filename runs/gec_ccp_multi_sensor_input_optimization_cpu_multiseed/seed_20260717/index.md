# GEC-CCP three-measurement input optimization (CPU FNO)

- 同じ学習済FNOを、条件と構造が異なるheld-out 3ケースへ再利用しました。
- 各センサー点に標準偏差5%の固定Gaussianノイズを付加しました。
- 校正ノイズは0%で、候補評価ごとの再抽選はしていません。
- 目的関数は線形電子密度profileのrelative L2です。loss履歴の表示だけ対数軸です。
- γは探索へ含めていますが、電子密度に対する低感度のため同定成功とは解釈しません。
- budget: 100 CPU FNO evaluations / optimizer / measurement

## Measurements

| ID | held-out case | Td | PP0 [W] | PA [Torr] | gamma | realized noise [%] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M1 | [`case_td003_pa050_pp0_3_gamma_010__steady`](M1/index.md) | 0.03 | 3 | 0.05 | 0.1 | 5.142 |
| M2 | [`case_td016_pp0_1_gamma_007__steady`](M2/index.md) | 0.16 | 1 | 0.1 | 0.07 | 4.555 |
| M3 | [`case_td030_pa200_pp0_5_gamma_004__steady`](M3/index.md) | 0.3 | 5 | 0.2 | 0.04 | 4.568 |

## Baselines

- mean realized sensor noise: 4.7551%
- common initial condition: truth-profile 91.1812%, full-field 87.1215%
- true-input FNO: noisy-sensor 5.0294%, truth-profile 1.2252%, full-field 2.4922%

## Aggregate accuracy

| method | mean sensor loss [%] | mean truth-profile [%] | mean full-field [%] | final regret [%] | R<=1.10 hits | median hit trial | R<=1.00 hits | CPU [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cmaes` | 5.1335 | 1.9703 | 4.0666 | 3.851 | 2/3 | 42.5 | 1/3 | 56.30 |
| `tpe` | 4.9790 | 1.5710 | 2.8736 | 0.963 | 3/3 | 23.0 | 1/3 | 57.61 |
| `random` | 5.3492 | 1.7629 | 3.1403 | 8.520 | 2/3 | 12.0 | 1/3 | 57.34 |

## Cost interpretation

- COMSOL train+validation 65-case acquisition: 4.952 h
- FNO interpolation training: 13.50 min
- Direct COMSOL estimate, 3 measurements x 100 trials: 23.625 h
- Direct COMSOL estimate using Td-wise medians: 10.997 h
- Surrogate from-scratch cost (65 cases + training + CMA search): 5.193 h
- Full 78-case benchmark cost + training + CMA search: 6.386 h
- 65-case build + estimated one COMSOL validation per measurement: 5.429 h
- Existing-FNO marginal CMA search: 56.30 s
- Fixed-budget from-scratch speedup: 4.55x
- Td-wise median-time speedup: 2.12x
- Direct COMSOL oracle-best estimate: 19.676 h
- oracle-bestは最終最良trialを事後に知る下限で、実運用の停止時間ではありません。
- 直接COMSOL時間は既存78ケースのTd別実測平均による推定で、最適候補の新規COMSOL検証は未実施です。

## Interpretation

- clean truth-profile誤差が最小だったのは `tpe` の平均 1.5710%です。
- Randomも3/3ケースを改善しましたが、平均truth-profile誤差は 1.7629%でした。
- 共通品質 R<=1.10 には`tpe` 3/3、Random 2/3ケースが到達しました。
- 従来のbest_trialは各手法自身の異なる最終値の最終更新時刻であり、収束速度の比較から除外しました。数値列はlast_improvement_trialとしてquality_attainment.csvにのみ残しています。
- `tpe` のnoisy-sensor lossは 4.9790%でtrue-input FNOより低い一方、truth/full-field誤差はtrue-inputより高く、ノイズまたはサロゲート誤差の入力補償が含まれます。
- 3ケース・1 seedなので最適化手法の一般順位は主張しません。サロゲートの有効性は、全ケースの真値誤差改善と反復評価時間で判定します。

## Graphs

- [common_quality_convergence.png](common_quality_convergence.png)
- [loss_history_by_case.png](loss_history_by_case.png)
- [budget_comparison.png](budget_comparison.png)
- [truth_error_by_case.png](truth_error_by_case.png)
- [parameter_recovery.png](parameter_recovery.png)
- [cost_and_amortization.png](cost_and_amortization.png)

## Tables

- [case_method_summary.csv](case_method_summary.csv)
- [method_aggregate.csv](method_aggregate.csv)
- [budget_summary.csv](budget_summary.csv)
- [quality_attainment.csv](quality_attainment.csv)
- [common_quality_history.csv](common_quality_history.csv)
- [all_trials.csv](all_trials.csv)
- [all_observations.csv](all_observations.csv)
- [runtime_comparison.csv](runtime_comparison.csv)
- [case_metadata.csv](case_metadata.csv)
- [run_metadata.json](run_metadata.json)

Total three-measurement execution wall time: 184.07 s
