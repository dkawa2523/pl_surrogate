# GEC-CCP optimizer seed study on three fixed measurements

- 仮想計測3件と5%点ノイズdrawは全optimizer seedで完全に同一です。
- optimizer seedのみ5本に変え、各手法15 run（3ケース x 5 seed）を評価しました。
- 主指標Rはbest sensor loss / true-input FNO sensor lossです。共通targetはR<=1.10、厳密targetはR<=1.00です。
- 各手法自身の異なる最終bestの更新時刻はlast improvementであり、速度比較には使用しません。
- optimizer設定はtestケースを見た後に変更せず固定しました。
- 成功率の誤差棒は、同じoptimizer seedの3ケースを一つのclusterとして再標本化した95% bootstrap区間です。

## Common-quality results

| method | R<=1.10 success | ERT [trial] | hit median, successes only [trial] | R<=1.00 success | final R median [IQR] | truth-profile median [%] | full-field median [%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cmaes` | 13/15 (86.7%) | 70.5 | 53.0 | 6/15 | 1.0126 [0.9511, 1.0385] | 1.7455 | 3.8806 |
| `tpe` | 15/15 (100.0%) | 23.5 | 21.0 | 9/15 | 0.9975 [0.9364, 1.0055] | 1.4040 | 3.1353 |
| `random` | 7/15 (46.7%) | 149.3 | 24.0 | 2/15 | 1.1306 [1.0181, 1.1805] | 3.0636 | 3.9546 |

ERT includes the 100-trial censoring cost for runs that did not attain R<=1.10. The hit median is conditional on success and must not be ranked without the success rate.

## Attainment by budget

| method | trial 25 | trial 50 | trial 100 |
| --- | ---: | ---: | ---: |
| `cmaes` | 1/15 | 6/15 | 13/15 |
| `tpe` | 11/15 | 15/15 | 15/15 |
| `random` | 4/15 | 5/15 | 7/15 |

## Graphs

- [multiseed_common_quality.png](multiseed_common_quality.png)
- [multiseed_attainment_rate.png](multiseed_attainment_rate.png)
- [multiseed_final_quality.png](multiseed_final_quality.png)

## Tables

- [method_aggregate.csv](method_aggregate.csv)
- [quality_attainment.csv](quality_attainment.csv)
- [multiseed_case_method_summary.csv](multiseed_case_method_summary.csv)
- [multiseed_trials.csv](multiseed_trials.csv)
- [multiseed_quality_history.csv](multiseed_quality_history.csv)
- [run_metadata.json](run_metadata.json)

Total five-seed three-measurement execution wall time: 923.33 s
