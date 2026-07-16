# GEC学会発表用・独立パーツ集

[プロジェクト入口](../../../README.md) → [レポート一覧](../../index.md) → [GEC-CCP専用目次](../gec_ccp_index.md) → 独立パーツ集

合成図から切り出した画像ではなく、現行データとsplitから一つずつ再描画した独立材料です。各図は4:3、PNG 3200×2400、PDF、SVG、metadata JSONを収録しています。

## 第三者向け・最初の選び方

| 発表で答えたい質問 | 最初に選ぶ図 | 必要な場合だけ追加 |
|---|---|---|
| どの装置・領域を解いているか | [CCPジオメトリー](ccp_geometry_conference.svg) / [ICPジオメトリー](icp_geometry_conference.svg) | メッシュ・構造格子 |
| どのようなプラズマ分布か | [CCP電子密度2D](ccp_electron_density_2d.svg) / [ICP電子密度2D](icp_electron_density_2d.svg) | イオン密度、電子温度、電位。斜視図は表紙・概要向け |
| データが応答範囲を覆うか | [CCP応答多様性](ccp_dataset_response_coverage.svg) / [ICP応答カバレッジ](icp_dataset_response_coverage.svg) | ICP形状カバレッジ、運転条件・応答面 |
| モデル間の総合差は何か | [値・空間勾配誤差](ccp_model_accuracy_scatter.svg) | 高精度域拡大、物性別散布図 |
| 物性ごとの予測性能は何か | [電子密度–電子温度 R²](ccp_model_r2_ne_vs_te.svg) / [イオン密度–電位 R²](ccp_model_r2_ni_vs_phi.svg) | 実測値CSV |
| センサー計測から入力を最適化できるか | [multi-seed達成率](../optimization_assets/ccp_opt_multiseed_attainment_rate.png) / [TPE trial profile](../optimization_assets/ccp_opt_tpe_trial_profiles_all_measurements.png) | 収束、費用、最適化後電子密度場は[最適化グラフ集](../optimization_assets/index.md) |
| 入力表現・前処理を説明したい | Z-score、寸法ベクトル、SDF | 本編ではなく手法・付録向け |

形式は、編集する場合はSVG、投稿・配布はPDF、スライドへ直接貼る場合はPNGを推奨します。metadataは出典と再生成条件の確認用です。全素材の機械可読一覧は[asset_catalog.csv](asset_catalog.csv)にあります。

### 推奨最小セット

第三者が最初に確認する本編用セットは、各ケースの「ジオメトリー → 電子密度2D → データ品質」の3段階です。CCPでは応答多様性、ICPでは形状カバレッジと応答カバレッジを品質図として使います。モデル比較が主題の場合だけ、最後に全モデル散布図を1枚追加してください。

同じ物理量の通常2D図と斜視図は同時使用せず、定量説明には通常2D、表紙・概要には斜視図を選びます。モデル比較も総合誤差図とR²図をすべて並べず、発表の主張に対応する1種類を選びます。

### 整理方針

Train / Validation / Test件数、条件tuple数、充足率など、表や本文で一行で示せる情報は画像化していません。データセット品質図として残しているのは、入力分布・応答分布・連続応答面など、グラフでなければ把握しにくい情報だけです。

## ジオメトリーとメッシュ

| 独立材料 | この図だけで伝える内容 | ファイル |
|---|---|---|
| GEC-CCPジオメトリー | base4の計算領域、駆動電極、誘電体接触境界、接地壁 | [PNG](ccp_geometry_conference.png) / [PDF](ccp_geometry_conference.pdf) / [SVG](ccp_geometry_conference.svg) / [metadata](ccp_geometry_conference_metadata.json) |
| GEC-CCPメッシュ | COMSOL出力に保存された1,425四角形要素 | [PNG](ccp_mesh_conference.png) / [PDF](ccp_mesh_conference.pdf) / [SVG](ccp_mesh_conference.svg) / [metadata](ccp_mesh_conference_metadata.json) |
| GEC-ICPジオメトリー | case_g002のplasma、誘電体、substrate、6 coils、上部EM領域 | [PNG](icp_geometry_conference.png) / [PDF](icp_geometry_conference.pdf) / [SVG](icp_geometry_conference.svg) / [metadata](icp_geometry_conference_metadata.json) |
| GEC-ICP格子 | 学習場を保存する600×440構造格子。視認性のため10本ごとに表示 | [PNG](icp_mesh_conference.png) / [PDF](icp_mesh_conference.pdf) / [SVG](icp_mesh_conference.svg) / [metadata](icp_mesh_conference_metadata.json) |

GEC-CCPは元のCOMSOLメッシュを表示しています。GEC-ICPはFEM要素を意味する図ではなく、サロゲート学習に使用する構造格子の図です。

## 代表4物理量の空間分布

各物理量を単独でスライドへ配置できるよう、GEC-CCP（base4代表ケース）とGEC-ICP（case_g002_op01）の通常2D図・斜視図を分離しています。

### GEC-CCP

| 物理量 | 通常2D | 平面斜視 |
|---|---|---|
| 電子密度 $n_e$ | [PNG](ccp_electron_density_2d.png) / [PDF](ccp_electron_density_2d.pdf) / [SVG](ccp_electron_density_2d.svg) / [metadata](ccp_electron_density_2d_metadata.json) | [PNG](ccp_electron_density_perspective.png) / [PDF](ccp_electron_density_perspective.pdf) / [SVG](ccp_electron_density_perspective.svg) / [metadata](ccp_electron_density_perspective_metadata.json) |
| イオン密度 $n_i$ | [PNG](ccp_ion_density_2d.png) / [PDF](ccp_ion_density_2d.pdf) / [SVG](ccp_ion_density_2d.svg) / [metadata](ccp_ion_density_2d_metadata.json) | [PNG](ccp_ion_density_perspective.png) / [PDF](ccp_ion_density_perspective.pdf) / [SVG](ccp_ion_density_perspective.svg) / [metadata](ccp_ion_density_perspective_metadata.json) |
| 電子温度 $T_e$ | [PNG](ccp_electron_temperature_2d.png) / [PDF](ccp_electron_temperature_2d.pdf) / [SVG](ccp_electron_temperature_2d.svg) / [metadata](ccp_electron_temperature_2d_metadata.json) | [PNG](ccp_electron_temperature_perspective.png) / [PDF](ccp_electron_temperature_perspective.pdf) / [SVG](ccp_electron_temperature_perspective.svg) / [metadata](ccp_electron_temperature_perspective_metadata.json) |
| 電位 $\phi$ | [PNG](ccp_electric_potential_2d.png) / [PDF](ccp_electric_potential_2d.pdf) / [SVG](ccp_electric_potential_2d.svg) / [metadata](ccp_electric_potential_2d_metadata.json) | [PNG](ccp_electric_potential_perspective.png) / [PDF](ccp_electric_potential_perspective.pdf) / [SVG](ccp_electric_potential_perspective.svg) / [metadata](ccp_electric_potential_perspective_metadata.json) |

### GEC-ICP

| 物理量 | 通常2D | 平面斜視 |
|---|---|---|
| 電子密度 $n_e$ | [PNG](icp_electron_density_2d.png) / [PDF](icp_electron_density_2d.pdf) / [SVG](icp_electron_density_2d.svg) / [metadata](icp_electron_density_2d_metadata.json) | [PNG](icp_electron_density_perspective.png) / [PDF](icp_electron_density_perspective.pdf) / [SVG](icp_electron_density_perspective.svg) / [metadata](icp_electron_density_perspective_metadata.json) |
| イオン密度 $n_i$ | [PNG](icp_ion_density_2d.png) / [PDF](icp_ion_density_2d.pdf) / [SVG](icp_ion_density_2d.svg) / [metadata](icp_ion_density_2d_metadata.json) | [PNG](icp_ion_density_perspective.png) / [PDF](icp_ion_density_perspective.pdf) / [SVG](icp_ion_density_perspective.svg) / [metadata](icp_ion_density_perspective_metadata.json) |
| 電子温度 $T_e$ | [PNG](icp_electron_temperature_2d.png) / [PDF](icp_electron_temperature_2d.pdf) / [SVG](icp_electron_temperature_2d.svg) / [metadata](icp_electron_temperature_2d_metadata.json) | [PNG](icp_electron_temperature_perspective.png) / [PDF](icp_electron_temperature_perspective.pdf) / [SVG](icp_electron_temperature_perspective.svg) / [metadata](icp_electron_temperature_perspective_metadata.json) |
| 電位 $\phi$ | [PNG](icp_electric_potential_2d.png) / [PDF](icp_electric_potential_2d.pdf) / [SVG](icp_electric_potential_2d.svg) / [metadata](icp_electric_potential_2d_metadata.json) | [PNG](icp_electric_potential_perspective.png) / [PDF](icp_electric_potential_perspective.pdf) / [SVG](icp_electric_potential_perspective.svg) / [metadata](icp_electric_potential_perspective_metadata.json) |

密度は対数色、電子温度と電位は線形色です。CCP電位は0を中心とする発散色、正値のみのICP電位は連続色を使用しています。斜視図は面を変形させず、同じ2D分布を平らな薄板として傾けた表示であり、高さ方向に物理量を割り当てていません。

## データセット品質：GEC-CCP

図にせず表で示す基本情報は、計画81条件中78条件（96%）、Train / Validation / Test = 54 / 11 / 13、各入力レベル24–27ケースです。

| 残す品質図 | この図だけで伝える内容 | ファイル |
|---|---|---|
| 応答多様性 | geometry別の電子密度P95分布と8.7倍の全体幅 | [PNG](ccp_dataset_response_coverage.png) / [PDF](ccp_dataset_response_coverage.pdf) / [SVG](ccp_dataset_response_coverage.svg) / [metadata](ccp_dataset_response_coverage_metadata.json) |

本編では、上記の基本情報を本文または表に記載し、応答の広がりを示す必要がある場合だけ「応答多様性」を使用してください。

## モデル比較：GEC-CCP

横軸は4物性のfield-value relative error、縦軸はspatial-gradient relative errorで、
ともに13テストケースの中央値を物性間で平均した値です。左下ほど、物理量の値と
空間分布形状の両方を正しく再現しています。青丸はNeural Network系、橙菱形は
Neural Operator系で、全点にモデル名を直接表示しています。

### 比較対象モデルと位置づけ

ここでの分類は、モデル比較図の色分けと一致します。全モデルはGEC-CCPの同じ固定格子上で、運転条件から電子密度 $n_e$、イオン密度 $n_i$、電子温度 $T_e$、電位 $\phi$ の空間分布を予測します。「Neural Operator」は本比較におけるアーキテクチャ系列を示す名称であり、未学習の解像度や異なる装置形状への汎化を実証したことを意味しません。

| 表示名（実装ID） | 系列 | 本コードでの構成と特徴 | 比較上の位置づけ | 参考文献 |
|---|---|---|---|---|
| Global MLP (`global_mlp`) | Neural Network | 運転条件ベクトルを多層全結合層へ入力し、4物性の全格子値を一括出力する。空間畳み込みや座標ごとの演算を持たない。 | 最も単純なベクトル→場の基準モデル。固定格子には適用しやすいが、局所性・平滑性はアーキテクチャとして保証しない。 | Hornik, Stinchcombe & White, [*Multilayer feedforward networks are universal approximators* (1989)](https://doi.org/10.1016/0893-6080%2889%2990020-8) |
| ResMLP (`global_resmlp`) | Neural Network | 条件ベクトルから全格子値を出力するMLPに、LayerNorm付き残差ブロックを導入する。 | 深い全結合モデルの最適化安定性を見る比較。画像用ResMLPの移植ではなく、残差結合をベクトル回帰へ適用した本コード独自の簡素な派生。 | 残差結合の原典：He et al., [*Deep Residual Learning for Image Recognition* (2016)](https://arxiv.org/abs/1512.03385) |
| DenseMLP (`global_densemlp`) | Neural Network | 各全結合ブロックの新規特徴を既存特徴へ連結し、最後に圧縮して全格子値を出力する。 | 特徴再利用の効果を見るベクトル回帰モデル。畳み込みDenseNetそのものではなく、dense connectivityをMLPへ移した本コード独自の派生。 | 密結合の原典：Huang et al., [*Densely Connected Convolutional Networks* (2017)](https://arxiv.org/abs/1608.06993) |
| U-Net (`unet`) | Neural Network | 条件・空間特徴マップを2D encoder–decoderへ入力し、同解像度の4物性場を出力する。encoderとdecoderをskip connectionで接続する。 | 局所構造と多尺度情報を明示的に扱うCNN基準モデル。固定格子上の場予測として使用。 | Ronneberger, Fischer & Brox, [*U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)](https://arxiv.org/abs/1505.04597) |
| U-Net++ (`unetpp`) | Neural Network | U-Netのskip経路を入れ子状・密結合にしたdepth-2の2D encoder–decoder。本採用設定ではattentionを使用しない。 | U-Netより細かな特徴融合が有効かを1回のスクリーニング学習で確認する追加比較。 | Zhou et al., [*UNet++: A Nested U-Net Architecture for Medical Image Segmentation* (2018)](https://arxiv.org/abs/1807.10165) |
| FNO (`fno`) | Neural Operator | 2D FFT上の低周波モードに学習可能なspectral convolutionを適用し、pointwise経路と加算する。 | 大域的な空間相関を周波数領域で扱う代表的operator基準モデル。 | Li et al., [*Fourier Neural Operator for Parametric Partial Differential Equations* (2021)](https://arxiv.org/abs/2010.08895) |
| FFNO (`ffno`) | Neural Operator | 2D spectral演算を各空間軸の1D Fourier演算へ因子分解し、局所経路と組み合わせる。 | FNOに対し、スペクトル演算の軽量化・深層化による差を見る比較。 | Tran et al., [*Factorized Fourier Neural Operators* (2023)](https://arxiv.org/abs/2111.13802) |
| U-NO (`u_no`) | Neural Operator | 低周波Fourier大域混合、局所畳み込み、残差結合を組み合わせた本コードの **U-NO-lite**。 | 大域成分と局所成分の併用を見る軽量比較。原著の完全なU字型・多解像度U-NOをそのまま再現したものではない。 | Rahman, Ross & Azizzadenesheli, [*U-NO: U-shaped Neural Operators* (2022)](https://arxiv.org/abs/2204.11127) |
| CNO (`cno`) | Neural Operator | GroupNorm付き局所2D residual convolution blockを積層した本コードの **CNO-lite**。 | 連続場を意識した畳み込み系operatorの軽量スクリーニング。原著CNOの連続–離散等価性を含む全構成の再現ではない。 | Raonić et al., [*Convolutional Neural Operators for robust and accurate learning of PDEs* (2023)](https://arxiv.org/abs/2302.01178) |
| POD-DeepONet (`deeponet_pod`) | Neural Operator / reduced order | 学習データだけから物性別POD基底を作り、branch MLPが運転条件からPOD係数を推定して全場を再構成する。 | 低次元の空間基底と条件回帰を分離でき、少数データで滑らかな場を表しやすい。本比較ではbranchを調整した採用版。R²図では表示名を「DeepONet」とする。 | Lu et al., [*A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data* (2022)](https://arxiv.org/abs/2111.05512) |
| DeepONet (`deeponet_plasma`) | Neural Operator | 運転条件・センサー記述量を処理するbranchと、座標・Fourier特徴を処理するtrunkを組み合わせて各位置の値を出力する軽量plasma版。 | branch–trunk分解を直接使う比較。11モデルの誤差図には含むが、採用版との混同を避けるためR²図からは除外する。 | Lu et al., [*Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators* (2021)](https://doi.org/10.1038/s42256-021-00302-5) |

ResMLP、DenseMLP、U-NO、CNOには、原著の着想を本データの固定格子・少数条件回帰へ合わせた派生実装が含まれます。したがって、表中の文献はアーキテクチャ上の出発点であり、原著結果の直接再現を意味しません。R²散布図は調整済み `deeponet_pod` を「DeepONet」と表示し、未採用の `deeponet_plasma` を除外しています。一方、値・空間勾配誤差の全モデル散布図は両者を分けて掲載しています。

| 独立材料 | この図だけで伝える内容 | ファイル |
|---|---|---|
| 全モデル散布図 | 11モデルの値精度と空間勾配精度を同時比較 | [PNG](ccp_model_accuracy_scatter.png) / [PDF](ccp_model_accuracy_scatter.pdf) / [SVG](ccp_model_accuracy_scatter.svg) |
| 高精度領域拡大 | Global MLPとDeepONetを除いた左下領域のモデル差 | [PNG](ccp_model_accuracy_scatter_zoom.png) / [PDF](ccp_model_accuracy_scatter_zoom.pdf) / [SVG](ccp_model_accuracy_scatter_zoom.svg) |
| 物性別散布図 | 電子密度・イオン密度・電子温度・電位ごとの傾向 | [PNG](ccp_model_accuracy_by_field.png) / [PDF](ccp_model_accuracy_by_field.pdf) / [SVG](ccp_model_accuracy_by_field.svg) |
| 電子密度–電子温度 R² | 横軸：電子温度 R²、縦軸：電子密度 R²。実測テストR²の座標を表示 | [PNG](ccp_model_r2_ne_vs_te.png) / [PDF](ccp_model_r2_ne_vs_te.pdf) / [SVG](ccp_model_r2_ne_vs_te.svg) |
| イオン密度–電位 R² | 横軸：電位 R²、縦軸：イオン密度 R²。実測テストR²の座標を表示 | [PNG](ccp_model_r2_ni_vs_phi.png) / [PDF](ccp_model_r2_ni_vs_phi.pdf) / [SVG](ccp_model_r2_ni_vs_phi.svg) |

数値と再現条件：[CSV](ccp_model_accuracy_scatter_data.csv) / [metadata](ccp_model_accuracy_scatter_metadata.json)

R²散布図の実測値：[CSV](ccp_model_r2_scatter_data.csv)。従来DeepONetを除外し、POD-DeepONetを「DeepONet」と表示しています。主図は高精度域、挿入図は全範囲です。

主図には「全モデル散布図」を使用し、モデルが密集する場合に「高精度領域拡大」を
併記してください。比較条件を揃えるため全モデルseed 412を使用しています。追加3モデルは
既存レシピの単発スクリーニングであり、モデルごとのepoch上限は同一ではありません。

## データセット品質：GEC-ICP

図にせず表で示す基本情報は、60形状×6運転点=360ケース、Train / Validation / Test = 288 / 36 / 36、形状 = 48 / 6 / 6、形状重複0です。

| 残す品質図 | この図だけで伝える内容 | ファイル |
|---|---|---|
| 形状カバレッジ | 5形状パラメータの範囲と中央値 | [PNG](icp_dataset_geometry_coverage.png) / [PDF](icp_dataset_geometry_coverage.pdf) / [SVG](icp_dataset_geometry_coverage.svg) / [metadata](icp_dataset_geometry_coverage_metadata.json) |
| 運転条件・応答面 | 背景に電子密度P95の連続応答面、前景に360個の実シミュレーション点 | [PNG](icp_dataset_operating_coverage.png) / [PDF](icp_dataset_operating_coverage.pdf) / [SVG](icp_dataset_operating_coverage.svg) / [metadata](icp_dataset_operating_coverage_metadata.json) |
| 応答カバレッジ | split別の電子密度P5–P95と中央値 | [PNG](icp_dataset_response_coverage.png) / [PDF](icp_dataset_response_coverage.pdf) / [SVG](icp_dataset_response_coverage.svg) / [metadata](icp_dataset_response_coverage_metadata.json) |

本編では「形状カバレッジ」→「応答カバレッジ」を推奨します。入力空間と応答の関係まで説明するときだけ「運転条件・応答面」を追加してください。応答面はサンプル凸包内だけの区分線形補間で、near-floor 1ケースは補間から外しています。白丸はnear-floorを含む全360ケースです。

## 分離した前処理・形状特徴量

| 独立材料 | 用途 | ファイル |
|---|---|---|
| Z-score前 | 物理スケール上の電子密度分布 | [PNG](ccp_zscore_before.png) / [PDF](ccp_zscore_before.pdf) / [SVG](ccp_zscore_before.svg) / [metadata](ccp_zscore_before_metadata.json) |
| Z-score後 | 同じ分布を中心0・尺度1へ移した結果 | [PNG](ccp_zscore_after.png) / [PDF](ccp_zscore_after.pdf) / [SVG](ccp_zscore_after.svg) / [metadata](ccp_zscore_after_metadata.json) |
| 寸法ベクトル | 5個のglobal geometry値による特徴量化 | [PNG](icp_dimension_vector.png) / [PDF](icp_dimension_vector.pdf) / [SVG](icp_dimension_vector.svg) / [metadata](icp_dimension_vector_metadata.json) |
| 単一coil SDF | SDFの符号と局所的距離表現 | [PNG](icp_single_coil_sdf.png) / [PDF](icp_single_coil_sdf.pdf) / [SVG](icp_single_coil_sdf.svg) / [metadata](icp_single_coil_sdf_metadata.json) |
| union SDF | 全coilを順序非依存でまとめた空間特徴 | [PNG](icp_union_sdf.png) / [PDF](icp_union_sdf.pdf) / [SVG](icp_union_sdf.svg) / [metadata](icp_union_sdf_metadata.json) |

## 再生成

```powershell
.\.venv-test\Scripts\python.exe -W error experiments\conference\scripts\plot_gec_conference_independent_assets.py
.\.venv-test\Scripts\python.exe -W error experiments\conference\scripts\plot_gec_ccp_model_comparison_scatter.py
```

生成元：[plot_gec_conference_independent_assets.py](../../../experiments/conference/scripts/plot_gec_conference_independent_assets.py)

モデル比較図生成元：[plot_gec_ccp_model_comparison_scatter.py](../../../experiments/conference/scripts/plot_gec_ccp_model_comparison_scatter.py)
