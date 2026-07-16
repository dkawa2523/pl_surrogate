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
| `cmaes` | 5.4939 | 2.2684 | 0.083585 | 0.0509273 | 1.8486 | 4.9124 | 18.63 |
| `tpe` | 5.4842 | 2.64153 | 0.0621818 | 0.0958605 | 2.0762 | 3.6128 | 18.96 |
| `random` | 5.5792 | 2.5797 | 0.0684867 | 0.0972186 | 1.2813 | 3.1159 | 19.08 |

Total wall time: 60.96 s

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
