# GEC-CCP sensor-based physical-input optimization (CPU FNO)

- 単一のheld-out COMSOLケースから電子密度センサーprofileを生成しました。
- PP0・PA・gammaを物理単位のまま直接探索し、人工的な補正係数は使用していません。
- lossは線形電子密度profileのrelative L2で、対数変換は使用していません。
- loss_history.pngのみ、収束過程を見やすくするため縦軸を対数表示しています。
- FNOはCUDAを不可視化してCPUのみで実行しました。
- synthetic point noise: 5% (fixed for every trial)
- synthetic calibration noise: 0%
- realized sensor-profile noise: 4.5551%
- case: `case_td016_pp0_1_gamma_007__steady`
- sensor height: z=12.7 mm
- truth: PP0=1 W, PA=0.1 Torr, gamma=0.07
- common initial point: PP0=3 W, PA=0.125 Torr, gamma=0.07
- budget: 100 CPU FNO evaluations per optimizer
- initial sensor-profile loss: 172.7896%
- initial full-field truth error: 170.0203%
- true-input FNO sensor-profile loss: 4.6317%
- true-input FNO full-field truth error: 3.3446%

## Best conditions

| method | loss [%] | PP0 [W] | PA [Torr] | gamma | profile truth [%] | full-field truth [%] | time [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cmaes` | 4.2485 | 1.28 | 0.070648 | 0.0714147 | 1.7912 | 4.5935 | 19.41 |
| `tpe` | 4.2926 | 1.25649 | 0.0722349 | 0.0691323 | 1.6405 | 4.3474 | 19.43 |
| `random` | 5.4780 | 1.27546 | 0.0765595 | 0.0858835 | 4.6751 | 6.4194 | 19.31 |

Total wall time: 62.50 s

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
