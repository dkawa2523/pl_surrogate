# GEC-CCP single-run model additions

## 結論

`deeponet_plasma`、`cno`、`unetpp`を、n78・train 54 / validation 11 / test 13、
split seed 7、学習seed 412の共通条件で各1回だけ学習・評価した。
ハイパーパラメータ探索や結果を見た再学習は行っていない。

追加3モデルではCNOが最良だった。ただし、同じseed 412の既存FNOより
test quality score、値誤差、勾配誤差の大部分で劣り、現時点でFNOを置き換える
根拠はない。UNet++は既存UNetより悪く、標準設定のまま追加する価値は低い。
通常DeepONetは調整済みPOD-DeepONetより大幅に悪く、空間図にも構造区画に沿う
放射状の誤差が明瞭に残った。

## 単発学習結果

値は13 testケースにおけるphysical relative L2の中央値。勾配も物理スケールの
relative L2中央値であり、すべて小さいほどよい。

| model | epoch（選択/上限） | 時間 | quality score | ne | ni | Te | phi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deeponet_plasma` | 156 / 160 | 191.5 s | 0.031767 | 12.93% | 12.59% | 20.89% | 12.39% |
| `cno` | 76 / 80 | 249.5 s | 0.001837 | 3.84% | 3.66% | 4.94% | 2.38% |
| `unetpp` | 116 / 120 | 352.6 s | 0.002686 | 6.31% | 6.09% | 4.57% | 3.36% |

| model | ne勾配 | ni勾配 | Te勾配 | phi勾配 |
| --- | ---: | ---: | ---: | ---: |
| `deeponet_plasma` | 65.76% | 68.67% | 69.59% | 46.09% |
| `cno` | 28.77% | 33.15% | 26.16% | 11.34% |
| `unetpp` | 37.61% | 40.59% | 26.27% | 14.71% |

負値率は3モデル・4物性すべて0だった。

## seed 412の既存モデルとの比較

| model | quality score | ne | ni | Te | phi | ne勾配 | ni勾配 | Te勾配 | phi勾配 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tuned POD-DeepONet | 0.000984 | 2.11% | 2.11% | 3.64% | 1.43% | 5.73% | 8.13% | 13.74% | 2.52% |
| FNO | 0.001143 | 2.24% | 2.28% | 3.75% | 1.90% | 23.93% | 27.65% | 23.98% | 11.41% |
| UNet | 0.001329 | 4.01% | 3.61% | 3.56% | 2.35% | 31.08% | 32.18% | 22.21% | 11.01% |
| CNO（今回） | 0.001837 | 3.84% | 3.66% | 4.94% | 2.38% | 28.77% | 33.15% | 26.16% | 11.34% |
| UNet++（今回） | 0.002686 | 6.31% | 6.09% | 4.57% | 3.36% | 37.61% | 40.59% | 26.27% | 14.71% |
| plasma DeepONet（今回） | 0.031767 | 12.93% | 12.59% | 20.89% | 12.39% | 65.76% | 68.67% | 69.59% | 46.09% |

この表は同じデータ分割・seedでのスクリーニング比較だが、学習上限は既存設定を
尊重してモデル間で異なる。したがって「無調整の既存レシピとしての有用性」は
比較できるが、各アーキテクチャの到達可能な最高性能を比較した結果ではない。

## 空間分布の所見

- CNOは追加3モデル中で最も滑らかで、密度ピークと下流への広がりを再現した。
  ただし局所勾配はFNOと同等からやや悪く、tuned PODとの差は大きい。
- UNet++は密度ピーク位置を追うが、構造部品の区画境界に沿う放射状誤差と、
  開口部近傍の過大・過小評価が残る。
- plasma DeepONetは大域形状を捉える一方、値・勾配とも不足し、構造区画に沿う
  非物理的な線状誤差が最も強い。

全13ケース・4物性のbest / p25 / median / p75 / worst図は
[`evaluation/spatial_truth_pred_error/index.md`](evaluation/spatial_truth_pred_error/index.md)にある。

## 設定と成果物

- 設定: `configs/experimental/gec_ccp_nn_operator_comparison_v2_single_run_additions/`
- 数値一覧: `comparison_metrics.csv`
- 学習状態: `seed_412/run_status.csv`
- 空間分布指標: `evaluation/spatial_truth_pred_error/model_field_summary.csv`
- 図一覧: `evaluation/spatial_truth_pred_error/publication_field_plots.csv`

## 判断

1. CNOは比較対象として残す価値があるが、標準候補は引き続きFNO・UNO・tuned POD。
2. UNet++は今回の既存設定ではUNetに対する追加価値を示せなかった。
3. plasma DeepONetは現設定のまま採用しない。改善する場合はbranch/trunkの再設計が必要で、
   今回の「調整なし単発評価」の範囲外とする。
