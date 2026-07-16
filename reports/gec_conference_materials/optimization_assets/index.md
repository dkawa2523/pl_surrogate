# GEC-CCP センサー同化・入力最適化 学会用グラフ

[プロジェクト入口](../../../README.md) → [GEC-CCP専用目次](../gec_ccp_index.md) → 最適化結果

学習済みFNOをCPUで反復評価し、固定高さの半径方向電子密度センサーに整合する入力
`PP0`、`PA`、`gamma`を探索した結果です。CMA-ES、TPE、Randomを各100 trialで比較し、
条件の異なるheld-out 3計測を使用しています。センサー値には標準偏差5%の固定Gaussian
ノイズを加え、目的関数は**線形電子密度profileのrelative L2**です。対数変換は目的関数に
使用せず、loss履歴の表示軸だけを対数にしています。

## 学会本編の推奨図

| 順序 | 図 | この図で伝える内容 | ファイル |
|---:|---|---|---|
| 1 | 共通品質の達成率 | 3計測×5 optimizer seedで、trial budgetに対する成功率を比較 | [PNG](ccp_opt_multiseed_attainment_rate.png) |
| 2 | 共通品質への収束 | 15 runの中央値と分布から、TPE・CMA-ES・Randomの収束を比較 | [PNG](ccp_opt_multiseed_common_quality.png) |
| 3 | 最終品質分布 | 100 trial後の最終品質とseed依存性を比較 | [PNG](ccp_opt_multiseed_final_quality.png) |
| 4 | TPEの半径方向分布 | trial番号の連続色、固定ノイズ付きセンサー点、COMSOL真値、best trialを3計測で表示 | [PNG](ccp_opt_tpe_trial_profiles_all_measurements.png) |
| 5 | 計算費用と償却 | 直接COMSOL探索、サロゲート構築、既存FNO再利用の推定時間を比較 | [PNG](ccp_opt_cost_and_amortization.png) |

本編では「達成率 → TPEの半径方向分布 → 計算費用」の3枚を基本とし、手法間の収束差を
詳しく説明する場合だけ共通品質収束と最終品質分布を追加します。

## 空間分布による最適化結果の確認

各図は上段にCOMSOL真値、初期条件のFNO、最適化後FNOを、下段に初期誤差、最適化後誤差、
絶対誤差の減少量を示します。センサーprofileだけで調整した結果が2D電子密度場へどう反映
されたかを確認する図です。

| 仮想計測 | held-out条件 | 電子密度場 |
|---|---|---|
| M1 | `Td=0.03, PP0=3 W, PA=0.05 Torr, gamma=0.10` | [PNG](ccp_opt_best_ne_field_M1.png) |
| M2 | `Td=0.16, PP0=1 W, PA=0.10 Torr, gamma=0.07` | [PNG](ccp_opt_best_ne_field_M2.png) |
| M3 | `Td=0.30, PP0=5 W, PA=0.20 Torr, gamma=0.04` | [PNG](ccp_opt_best_ne_field_M3.png) |

## 補足図

| 図 | 用途 | ファイル |
|---|---|---|
| ケース別loss履歴 | 1 optimizer seedにおける3計測・3手法の推移 | [PNG](ccp_opt_loss_history_by_case.png) |
| trial budget比較 | 所定品質までに必要な評価数 | [PNG](ccp_opt_budget_comparison.png) |
| 真値誤差 | センサーlossだけでなくclean profile・full-field誤差を確認 | [PNG](ccp_opt_truth_error_by_case.png) |
| 入力回復 | `PP0`、`PA`、`gamma`の推定値を真値と比較 | [PNG](ccp_opt_parameter_recovery.png) |
| TPE M1拡大 | M1のtrial別半径方向profile | [PNG](ccp_opt_tpe_trial_profiles_M1.png) |
| TPE M2拡大 | M2のtrial別半径方向profile | [PNG](ccp_opt_tpe_trial_profiles_M2.png) |
| TPE M3拡大 | M3のtrial別半径方向profile | [PNG](ccp_opt_tpe_trial_profiles_M3.png) |

## 主要結果と解釈範囲

共通品質を $R=\text{best sensor loss}/\text{true-input FNO sensor loss}\leq1.10$ とすると、
100 trialまでの達成率はTPE 15/15、CMA-ES 13/15、Random 7/15でした。censoringを含む
ERTはそれぞれ23.5、70.5、149.3 trialです。これは3計測×5 optimizer seedに対する結果で、
一般的な最適化手法の優劣を確定する規模ではありません。

`gamma`は電子密度に対する感度が低いため、電子密度profileだけからの一意な同定成功を
主張しません。また、センサーlossがtrue-input FNOより小さくても、観測ノイズやサロゲート
誤差を入力変数が補償している可能性があります。計算費用図の直接COMSOL時間は既存ケースの
実測時間に基づく推定であり、新しい300回のCOMSOL最適化を実行した結果ではありません。

## 数値データと出典

| 内容 | ファイル |
|---|---|
| multi-seed手法集約 | [CSV](ccp_opt_multiseed_method_aggregate.csv) |
| multi-seedケース別結果 | [CSV](ccp_opt_multiseed_case_method_summary.csv) |
| 品質達成trial | [CSV](ccp_opt_multiseed_quality_attainment.csv) |
| 3計測の条件 | [CSV](ccp_opt_case_metadata.csv) |
| 単一seedのケース×手法結果 | [CSV](ccp_opt_case_method_summary.csv) |
| 実行時間比較 | [CSV](ccp_opt_runtime_comparison.csv) |
| TPE profile要約・全trial | [要約CSV](ccp_opt_tpe_profile_summary.csv) / [全trial CSV](ccp_opt_tpe_trial_profiles.csv) |
| 再現条件 | [multi-seed metadata](ccp_opt_multiseed_run_metadata.json) / [3計測 metadata](ccp_opt_run_metadata.json) / [TPE metadata](ccp_opt_tpe_run_metadata.json) |
| コピー元対応 | [source_manifest.csv](source_manifest.csv) |

正本の詳細レポートは[3計測・単一seed](../../../runs/gec_ccp_multi_sensor_input_optimization_cpu/index.md)、
[3計測・5 seeds](../../../runs/gec_ccp_multi_sensor_input_optimization_cpu_multiseed/index.md)、
[TPE trial profile](../../../runs/gec_ccp_multi_sensor_input_optimization_cpu/tpe_trial_profiles/index.md)です。
このフォルダの図と表は、学会素材から参照しやすいよう正本を名称統一してコピーしたものです。
