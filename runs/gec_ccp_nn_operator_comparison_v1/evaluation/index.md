# GEC-CCP NN / Neural-Operator Comparison v1

78条件のGEC-CCPデータを、固定した補間分割（学習54 / Validation 11 / Test 13）で比較した結果です。8モデルをseed 411、412、413で学習し、24/24 runが完了しました。代表seedはTestを使わず、Validation空間目的値だけで選択しています。

## 三seed集計

QualityとValidationは小さいほど良く、plasma R2は大きいほど良い指標です。

| モデル | Quality 平均 ± SD | plasma R2 平均 ± SD | Validation 平均 ± SD | 平均学習時間 [s] |
| --- | ---: | ---: | ---: | ---: |
| `u_no` | 0.000204 ± 0.000009 | 0.999811 ± 0.000007 | 0.02379 ± 0.00020 | 1051.5 |
| `ffno` | 0.000503 ± 0.000009 | 0.999449 ± 0.000017 | 0.03792 ± 0.00054 | 906.1 |
| `fno` | 0.001204 ± 0.000054 | 0.9987 ± 0.0001 | 0.05933 ± 0.00211 | 815.9 |
| `unet` | 0.001336 ± 0.000037 | 0.9985 ± 0.0001 | 0.06936 ± 0.00191 | 651.0 |
| `global_resmlp` | 0.002180 ± 0.000589 | 0.9982 ± 0.0005 | 0.07610 ± 0.01175 | 351.7 |
| `deeponet_pod` | 0.003022 ± 0.000190 | 0.9976 ± 0.0001 | 0.09778 ± 0.00311 | 419.2 |
| `global_densemlp` | 0.003582 ± 0.001122 | 0.9969 ± 0.0012 | 0.09876 ± 0.01490 | 343.6 |
| `global_mlp` | 0.05866 ± 0.01177 | 0.9066 ± 0.0347 | 0.43708 ± 0.03144 | 532.7 |

詳細値は[三seed監査レポート](seed_matrix/report.md)と[集計CSV](seed_matrix/model_seed_aggregate.csv)にあります。

## 空間分布評価

Validationで選んだ代表seedについて、各モデルの13 Testケースを再推論しました。真値・予測値・符号付き相対誤差を、`ne`, `ni`, `Te`, `phi`ごとにbest / p25 / median / p75 / worstの5ケースで出力しています。

| モデル | 13ケース・4物性平均 rel.RMSE |
| --- | ---: |
| `u_no` | 0.0114 |
| `ffno` | 0.0184 |
| `fno` | 0.0277 |
| `global_resmlp` | 0.0296 |
| `unet` | 0.0359 |
| `global_densemlp` | 0.0376 |
| `deeponet_pod` | 0.0452 |
| `global_mlp` | 0.2862 |

- [全モデルの空間分布図一覧](spatial_truth_pred_error/index.md)
- [160図のパスと物性別誤差](spatial_truth_pred_error/publication_field_plots.csv)
- [全104 model/case行の空間誤差](spatial_truth_pred_error/spatial_plot_summary.csv)
- [Validation-only代表seed](../validation_selected/best_by_model.csv)

## 結論

1. **値の空間分布を最も正確に推論したのはU-NO**です。三seedのばらつきも極めて小さく、worstケースでも代表図の相対RMSEは、`ne` 1.57%、`Te` 1.96%でした。
2. **FFNOは次点で、FNOより一貫して高精度**です。U-NOより約14%短時間で、精度と計算時間の折衷候補です。
3. **ResMLPはベクトルモデルとして最も有用**です。U-NOの約1/3の学習時間で、代表seedの4物性平均rel.RMSEは2.96%です。旧GlobalMLPから大幅に改善しました。
4. **POD-DeepONetは安定していますが、総合値分布ではResMLP・Operator系に届きません**。一方、密度・電位の勾配と正値維持は良好です。
5. **旧GlobalMLPは空間分布を過度に平滑化・過大評価するケースが残り、比較基準としては明確に劣ります**。

## 物理的な注意点

- Quality scoreは符号違反を直接罰していません。三seed平均のプラズマ内`ne`負値率は、U-NO 3.09%、FFNO 4.01%、FNO 6.18%、U-Net 6.63%です。一方、ResMLP 0.027%、DenseMLP 0.017%、POD-DeepONet 0.152%でした。
- 値分布ではU-NOが最良ですが、密度勾配の中央値誤差はResMLP/DenseMLPが約7～10%で、U-NOの約16～20%より良好です。シース・境界勾配を重視する用途では総合Qualityだけで選ばないでください。
- 学習損失と主要指標は`plasma_only`です。図でも非対象領域を灰色にマスクしており、プラズマ外の予測値を物理解釈やモデル順位に使用してはいけません。
- `Td`と構造IDが1対1対応するため、table-onlyモデルも`Td`から構造を暗黙に識別できます。本結果は未知構造一般化の証明ではありません。
- ResMLP/DenseMLPの改善には、接続方式だけでなく幅48、GELU、Torch実装、49列ridge headという条件差が含まれます。残差接続単独の効果を主張するには、幅48のplain MLP対照が必要です。
- 本比較は補間専用です。PP0=1だけで学習する旧stress laneや外挿性能とは混在させていません。
