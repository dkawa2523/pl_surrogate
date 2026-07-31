# 学会正式採用ガイド：Dimension / SDF学習精度

このページを、Dimension入力とSDF入力の学習精度を説明するときの正式な文章参照先とします。

## 主図

[![Dimension / SDF総合精度比較](dimension_vs_sdf_relative_rmse_comparison.png)](dimension_vs_sdf_relative_rmse_comparison.png)

- スライド: [PNG](dimension_vs_sdf_relative_rmse_comparison.png)
- 原稿・印刷: [PDF](dimension_vs_sdf_relative_rmse_comparison.pdf)
- 編集: [SVG](dimension_vs_sdf_relative_rmse_comparison.svg)

## 図の読み方

- 青棒はDimension入力、橙棒はUnion SDF入力です。
- 横軸は電子密度、イオン密度、電子温度、電位の相対RMSEを単純平均した値です。
- 棒が短いほど、4物性を総合した予測誤差が小さいことを示します。
- 各値は共通36件のstructure-holdout test caseに対するcase-macro評価です。
- FNOは正式採用図から除外しています。

## 発表用説明文

> Dimension入力とUnion SDF入力について、未学習構造36件に対する4物性の平均相対RMSEを比較した。
> FFNOとUNOではSDF入力の誤差がDimension入力より小さく、U-Net系とCNOではDimension入力の誤差が小さい。
> この結果は、SDF表現の効果がモデルアーキテクチャに依存することを示している。

## 数値上の要点

| モデル | Dimension | Union SDF | 精度が高い表現 |
|---|---:|---:|---|
| U-Net | 9.9% | 11.9% | Dimension |
| U-Net++ | 9.7% | 12.6% | Dimension |
| Attention U-Net++ | 9.3% | 13.1% | Dimension |
| FFNO | 7.2% | 6.6% | Union SDF |
| UNO | 5.6% | 4.7% | Union SDF |
| CNO | 4.7% | 6.6% | Dimension |

## 解釈上の注意

- この値は4物性の単純平均であり、特定物性だけの精度を表すものではありません。
- 物性別の根拠は[補助RMSE図](index.md#nn系--no系の色分け比較)と
  [集約CSV](model_family_rmse.csv)を参照してください。
- 絶対RMSEではなく、物性スケールで正規化した相対RMSEです。
- この比較はsurrogateのtest精度であり、構造最適化結果や高忠実度ICP再解析の精度ではありません。

## 再現情報

- [正式採用図カタログ](adopted_figures.csv)
- [metadata](metadata.json)
- [生成スクリプト](../../../experiments/conference/scripts/plot_icp_dimension_sdf_fieldwise_rmse.py)
