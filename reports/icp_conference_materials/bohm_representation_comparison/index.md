# ICP学会本番用：Bohm最適化のSDF・寸法表現比較

このページを本番用アニメーションの固定入口とします。両動画は同じ表示仕様で再生成済みです。

## 問題設定図

- [ICP Bohm最適化の問題設定](icp_bohm_optimization_problem_setup.png)（[PDF](icp_bohm_optimization_problem_setup.pdf) / [SVG](icp_bohm_optimization_problem_setup.svg)）
- [Bohm目的関数の数式図](icp_bohm_objective_definition.png)（[PDF](icp_bohm_objective_definition.pdf) / [SVG](icp_bohm_objective_definition.svg)）
- [図のメタデータ](icp_bohm_optimization_problem_setup_metadata.json)

## 本番用アニメーション

| 表現 | アニメーション | 最終静止画 | 構造探索 |
|---|---|---|---|
| SDF | [GIF](../sdf_bohm_optimization/bohm_optimization_trial_animation.gif) | [PNG](../sdf_bohm_optimization/bohm_optimization_conference_layout.png) / [PDF](../sdf_bohm_optimization/bohm_optimization_conference_layout.pdf) / [SVG](../sdf_bohm_optimization/bohm_optimization_conference_layout.svg) | 不等間隔、一定傾斜の可変高さ |
| 寸法 | [GIF](../dimension_bohm_optimization/bohm_optimization_trial_animation.gif) | [PNG](../dimension_bohm_optimization/bohm_optimization_conference_layout.png) / [PDF](../dimension_bohm_optimization/bohm_optimization_conference_layout.pdf) / [SVG](../dimension_bohm_optimization/bohm_optimization_conference_layout.svg) | 等間隔、同一高さ |

## 共通条件

- 学習済みU-NetをGPU推論に使用
- コイル数：2、3、4、5
- 発表順はコイル数を交互に並べ、各75候補、合計300 Trial
- ウェハ近傍10層、面積等価32半径区間のBohmフラックス
- 密度閾値：`1.0e17 m^-3`
- Bohmプロファイル縦軸：`0～4.5 × 10^20 m^-2 s^-1`で共通固定
- 300フレーム、12 fps、約26.7秒
- 最終フレームは300 Trial内の最小Lossケースを1.8秒保持

## 動画内の最良Lossケース

| 表現 | 発表順Trial | 元trial_id | コイル数 | Loss | 最大局所偏差 | 平均密度 |
|---|---:|---:|---:|---:|---:|---:|
| SDF | 127 | 281 | 4 | 0.086477 | 0.136849 | `1.0267e17 m^-3` |
| 寸法 | 116 | 253 | 5 | 0.123553 | 0.207446 | `1.3168e17 m^-3` |

ここでのTrial番号は、学会用にコイル数を交互配置した表示順です。元データとの対応には`元trial_id`を使用します。

## 発表での推奨順

1. 寸法版で、等間隔・同一高さという探索制約を説明
2. SDF版で、不等間隔・一定傾斜の配置変化を説明
3. 共通縦軸のBohmプロファイルとbest-so-far Lossを比較
4. 最後に、SDF可変高さは学習幾何分布外でありICP再解析前であることを明示

SDF版の低いサロゲートLossだけで物理的優位性を断定しません。高忠実度ICP再解析による検証誤差とregret評価が未完了です。

## 出典と再生成

- SDF run：`runs/icp_stage4_bohm_flux_sdf_500trials_spacing_slope_v4`
- 寸法run：`runs/icp_stage4_bohm_flux_dimension_300trials_v1`
- アニメーション生成：[animate_icp_bohm_sdf_trials.py](../../../scripts/animate_icp_bohm_sdf_trials.py)
- 寸法最適化：[optimize_bohm_flux_dimension_300trials_v1.py](../../../experiments/icp_stage4/scripts/optimize_bohm_flux_dimension_300trials_v1.py)
- 機械可読な固定情報：[manifest.json](manifest.json)
