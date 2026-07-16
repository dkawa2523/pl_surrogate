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
| `cmaes` | 4.3844 | 1.4278 | 0.0603652 | 0.0697173 | 2.7959 | 6.2094 | 18.63 |
| `tpe` | 4.2735 | 1.19111 | 0.0803965 | 0.0623903 | 1.8814 | 4.1880 | 18.94 |
| `random` | 4.7255 | 1.54224 | 0.0538906 | 0.0469485 | 3.6628 | 7.6590 | 18.69 |

Total wall time: 60.50 s

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
