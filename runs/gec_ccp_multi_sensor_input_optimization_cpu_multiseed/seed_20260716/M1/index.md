# GEC-CCP sensor-based physical-input optimization (CPU FNO)

- 単一のheld-out COMSOLケースから電子密度センサーprofileを生成しました。
- PP0・PA・gammaを物理単位のまま直接探索し、人工的な補正係数は使用していません。
- lossは線形電子密度profileのrelative L2で、対数変換は使用していません。
- loss_history.pngのみ、収束過程を見やすくするため縦軸を対数表示しています。
- FNOはCUDAを不可視化してCPUのみで実行しました。
- synthetic point noise: 5% (fixed for every trial)
- synthetic calibration noise: 0%
- realized sensor-profile noise: 5.1418%
- case: `case_td003_pa050_pp0_3_gamma_010__steady`
- sensor height: z=12.7 mm
- truth: PP0=3 W, PA=0.05 Torr, gamma=0.1
- common initial point: PP0=3 W, PA=0.125 Torr, gamma=0.07
- budget: 100 CPU FNO evaluations per optimizer
- initial sensor-profile loss: 56.2843%
- initial full-field truth error: 49.0076%
- true-input FNO sensor-profile loss: 5.4257%
- true-input FNO full-field truth error: 2.4353%

## Best conditions

| method | loss [%] | PP0 [W] | PA [Torr] | gamma | profile truth [%] | full-field truth [%] | time [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cmaes` | 5.5049 | 2.26774 | 0.085017 | 0.0591266 | 1.4987 | 4.5939 | 19.01 |
| `tpe` | 5.8348 | 2.0499 | 0.103821 | 0.0487786 | 2.2371 | 5.7153 | 18.91 |
| `random` | 6.3928 | 1.69182 | 0.138697 | 0.0949919 | 4.1074 | 9.7629 | 18.62 |

Total wall time: 60.76 s

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
