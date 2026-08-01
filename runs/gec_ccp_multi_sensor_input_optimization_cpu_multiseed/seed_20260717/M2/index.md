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
| `cmaes` | 4.2588 | 1.33279 | 0.0658401 | 0.0798917 | 1.9099 | 4.9307 | 18.67 |
| `tpe` | 4.3818 | 1.05237 | 0.0958976 | 0.0489929 | 1.2550 | 3.3583 | 19.44 |
| `random` | 4.6240 | 1.15278 | 0.0796247 | 0.0991324 | 1.5105 | 3.6611 | 19.17 |

Total wall time: 61.54 s

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
