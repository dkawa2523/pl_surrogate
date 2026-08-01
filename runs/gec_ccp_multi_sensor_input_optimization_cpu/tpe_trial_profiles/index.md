# TPE trial-by-trial radial electron-density profiles

- optimizer seed: `20260715`
- TPEの各trial条件を学習済FNOへ再入力し、半径方向電子密度profileを再生成しました。
- 線色はtrial番号1から100に対応する連続`viridis`カラーマップです。
- 白抜き黒縁の点は、最適化ターゲットとして使用した固定ノイズ付きセンサー値です。
- 破線はノイズ付加前のheld-out COMSOL真値です。
- 太線は100 trial中の最小センサーlossを持つprofileです。

## Graphs

- [all measurements](tpe_trial_profiles_all_measurements.png)
- [M1](M1_tpe_trial_profiles.png)
- [M2](M2_tpe_trial_profiles.png)
- [M3](M3_tpe_trial_profiles.png)

## Data

- [tpe_trial_profiles.csv](tpe_trial_profiles.csv)
- [profile_summary.csv](profile_summary.csv)
- [run_metadata.json](run_metadata.json)

Profile regeneration and plotting wall time: 57.90 s
