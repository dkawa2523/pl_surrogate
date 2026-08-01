# GEC-CCP POD-DeepONet coefficient-branch tuning v2

## 結論

旧POD-DeepONetの主なボトルネックはPOD基底ではなく、条件・構造記述子からPOD係数を推定するbranchでした。`hidden: [192, 192]`の最後のhidden層に活性化がなく、実質的な非線形変換が1層だけでした。全hidden層後にGELUを適用すると、同一データ・POD基底・損失・optimizerでtest quality scoreが61.3%改善し、FNOと同等以上になりました。

## 変更内容

- branchの活性化関数と、最後のhidden層にも活性化を適用するかを設定可能にした
- descriptorを`raw / drop / compact`から選択可能にした
- descriptorの`condition_dim`と`dim`を明示検証するようにした
- 学習ケースだけでfitする`train_zscore`を追加した
- 標準化統計をcheckpointへ保存し、推論時に同じ統計を再利用するようにした
- `branch`未指定の旧設定・旧checkpointは従来のネットワークを再現する

採用設定は次のとおりです。

```yaml
model_cfg:
  hidden: [192, 192]
  branch:
    activation: gelu
    activate_last_hidden: true
    descriptor:
      mode: raw
      condition_dim: 4
      dim: 114
      normalization: none
```

## Validation-only候補選定

78条件を54 train / 11 validation / 13 testに固定し、seed 412のvalidationだけで選びました。test値は候補確定まで参照していません。

| 候補 | branch | descriptor | validation空間目的値（低いほど良い） |
|---|---|---|---:|
| 採用 | 192×2、各hidden後GELU | raw | 0.057746 |
| 候補2 | 128×2、各hidden後GELU | train-only z-score | 0.060822 |
| 候補3 | 64×3、各hidden後GELU | train-only z-score | 0.110160 |

## 3-seed test結果

値はseed 411/412/413の平均±標準偏差です。quality scoreは誤差指標なので低いほど良く、R²は高いほど良い指標です。

| モデル | validation目的値 | test quality score | test plasma R² | 学習時間/seed |
|---|---:|---:|---:|---:|
| POD-DeepONet（旧branch） | 0.097779 ± 0.003115 | 0.003022 ± 0.000190 | 0.997603 ± 0.000088 | 419.2 s |
| FNO | 0.059329 ± 0.002107 | 0.001204 ± 0.000054 | 0.998707 ± 0.000077 | 815.9 s |
| POD-DeepONet（調整後） | 0.063080 ± 0.005481 | **0.001169 ± 0.000242** | **0.999010 ± 0.000166** | 435.5 s |

調整後PODは旧PODに対しvalidation目的値を35.5%、test quality scoreを61.3%低減しました。FNOに対してvalidationは6.3%高い一方、test quality scoreは2.9%低く、plasma R²も高くなりました。

## 空間分布と連続性

3-seed平均のtest物理診断です。値は各seed内13 testケースのmedian physical relative L2をseed平均したものです。

| モデル | ne値 | ni値 | Te値 | phi値 | ne勾配 | ni勾配 | Te勾配 | phi勾配 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| POD旧branch | 3.49% | 3.37% | 5.21% | 2.53% | 9.40% | 10.32% | 18.41% | 4.33% |
| FNO | 2.42% | **2.39%** | 3.65% | 1.95% | 24.69% | 29.07% | 24.77% | 11.55% |
| POD調整後 | **2.19%** | 2.47% | **3.60%** | **1.41%** | **5.58%** | **8.35%** | **14.80%** | **2.77%** |

調整後PODはniの値誤差だけFNOより0.09 percentage point高いものの、ほかの値誤差と全物性の勾配誤差でFNOを下回りました。POD基底から再構成するため、推論図にはFNOで見られる高周波のざらつきがなく、誤差は主に狭いシース・開口遷移部へ局在しています。

代表seed 412の全13 testケース平均rel.RMSEは、ne 2.94%、ni 2.60%、Te 4.37%、phi 1.52%です。

- [空間分布一覧](evaluation/spatial_truth_pred_error/index.md)
- [ne median図](evaluation/spatial_truth_pred_error/publication_by_field/deeponet_pod/ne/interp/median_case_td003_pa200_pp0_5_gamma_004__steady_ne.png)
- [Te worst図](evaluation/spatial_truth_pred_error/publication_by_field/deeponet_pod/Te/interp/worst_case_td003_pa200_pp0_1_gamma_010__steady_Te.png)
- [3-seed詳細物理指標](evaluation/seed_matrix/model_seed_aggregate.csv)
- [候補選定結果](tuning/validation_results.csv)

## 解釈上の制約

この結果は既知3構造を含むGEC-CCPのmarginal interpolation評価です。`Td`と構造IDが対応しているため、未知構造への幾何一般化を証明するものではありません。また、目的変数は対数変換しておらず、物理値のidentity変換後にtrain-only z-scoreを使う既存契約を維持しています。

候補比較21.9分、最終3-seed学習21.8分、合計43.7分でした。
