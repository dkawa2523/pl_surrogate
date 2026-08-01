# GEC-CCP sensor-based physical-input optimization (CPU FNO)

- 単一のheld-out COMSOLケースから電子密度センサーprofileを生成しました。
- PP0・PA・gammaを物理単位のまま直接探索し、人工的な補正係数は使用していません。
- lossは線形電子密度profileのrelative L2で、対数変換は使用していません。
- loss_history.pngのみ、収束過程を見やすくするため縦軸を対数表示しています。
- FNOはCUDAを不可視化してCPUのみで実行しました。
- synthetic point noise: 5% (fixed for every trial)
- synthetic calibration noise: 0%
- realized sensor-profile noise: 4.5685%
- case: `case_td030_pa200_pp0_5_gamma_004__steady`
- sensor height: z=12.7 mm
- truth: PP0=5 W, PA=0.2 Torr, gamma=0.04
- common initial point: PP0=3 W, PA=0.125 Torr, gamma=0.07
- budget: 100 CPU FNO evaluations per optimizer
- initial sensor-profile loss: 44.2076%
- initial full-field truth error: 42.3367%
- true-input FNO sensor-profile loss: 5.0309%
- true-input FNO full-field truth error: 1.6967%

## Best conditions

| method | loss [%] | PP0 [W] | PA [Torr] | gamma | profile truth [%] | full-field truth [%] | time [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cmaes` | 5.6479 | 4.93704 | 0.192619 | 0.0582259 | 2.1524 | 2.3567 | 19.00 |
| `tpe` | 5.0710 | 4.98374 | 0.198067 | 0.0913037 | 1.3819 | 1.6499 | 19.20 |
| `random` | 5.8444 | 4.90132 | 0.192279 | 0.0842926 | 2.4969 | 2.6439 | 19.10 |

Total wall time: 61.57 s

## Graphs

- [loss_history.png](loss_history.png)
- [condition_history.png](condition_history.png)
- [search_space.png](search_space.png)
- [best_conditions.png](best_conditions.png)
- [best_profiles.png](best_profiles.png)
- [best_ne_field.png](best_ne_field.png)

## Tables

- [summary.csv](summary.csv)
- [trials.csv](trials.csv)
- [observations.csv](observations.csv)
- [run_metadata.json](run_metadata.json)
