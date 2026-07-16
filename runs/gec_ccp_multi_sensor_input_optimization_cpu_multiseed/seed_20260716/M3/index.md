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
| `cmaes` | 5.2585 | 4.9095 | 0.199969 | 0.0550835 | 1.5648 | 1.8495 | 18.70 |
| `tpe` | 5.0185 | 4.99149 | 0.199538 | 0.0656274 | 1.3857 | 1.6768 | 18.93 |
| `random` | 6.3319 | 4.84538 | 0.191382 | 0.0768686 | 3.2461 | 3.3503 | 19.08 |

Total wall time: 60.93 s

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
