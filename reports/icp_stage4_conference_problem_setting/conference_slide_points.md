# ICP_stage4 Conference Slide Points

## Core Message

本研究の主張は、低圧 ICP コイル設計において、コイル数・半径方向配置・寸法を単なる設計パラメータベクトルとして surrogate に入れるのではなく、場が解かれる構造そのものとして特徴量化することにある。

今回の ICP_stage4 dataset では、`pp`, `pp0` を process scalar とし、`llcoil`, `rrc`, `nncoil`, `rrce`, `zzc` は geometry 生成元としてのみ使った。モデル入力には raw coil parameter vector を入れず、coil mask / SDF / part SDF / plasma boundary などの spatial feature map として与えた。

最終的に、8個の2次元場 `ne`, `ni`, `Te`, `phi`, `Br`, `Bz`, `Jelr`, `Jelz` を同時予測する FFNO surrogate で、held-out coil structures に対して mean plasma R2 = `0.960` を得た。

## Recommended Slide Flow

1. 背景と課題
   - ICP 装置では、コイル配置が誘導電磁場、電子加熱、密度ピーク位置、wafer 上均一性を同時に変える。
   - 従来の寸法パラメータ入力では、同じ構造の別表現、整数 coil count、inactive slot、coil order への過敏性が起きやすい。
   - 目的は、process 条件と coil geometry が作る2次元場全体を、構造特徴量ベースで予測・最適化すること。

2. Dataset
   - 360 cases = 60 coil structures x 6 process settings。
   - Process scalar: `pp`, `pp0`。
   - Geometry source: `llcoil`, `rrc`, `nncoil`, `rrce`, `zzc`。
   - Targets: `ne`, `ni`, `Te`, `phi`, `Br`, `Bz`, `Jelr`, `Jelz`。
   - 図: `fig01_dataset_design_space`

3. 構造特徴量化の重要性
   - `part_lite_v1` は order-invariant で保守的だが、ICP_stage4 では coil radial sequence の情報が落ちすぎた。
   - `icp_part_sdf_lite_v1` は固定 slot SDF で coil ごとの位置情報を保持し、精度が大きく回復した。
   - 図: `fig02_feature_ablation_mean_r2`

4. 8-field surrogate の結果
   - FFNO E80 は全 target で高い R2。
   - UNet E80 も物理場は学習できるが、密度予測は FFNO より弱い。
   - 図: `fig03_e80_per_target_r2`

5. 学習過程
   - FFNO は 80 epoch で validation score が安定して上昇。
   - E20 では `ne` がまだ弱く、E80 で density field が大きく改善。
   - loss curve は対数軸で示し、初期急減と後半の収束を同じ図で読めるようにする。
   - 図: `fig04_training_curves_e80`
   - 図: `fig05_e20_to_e80_improvement_ffno`

6. 空間分布としての評価
   - FFNO E80 は boundary / mid plasma / deep plasma で高い R2 を維持。
   - 密度分布の integral, p99, distribution score も実用的な範囲。
   - test case の `truth / prediction / error` を並べ、密度ピーク位置と広がりが視覚的に合っているかを確認する。
   - 空間分布図は対数表示ではなく、密度は `10^18 m^-3`、磁場は `mT` などの線形スケールで示す。
   - 図: `fig06_ffno_e80_regional_r2_heatmap`
   - 図: `fig07_density_distribution_errors_ffno`
   - 図: `fig08_ffno_core_field_triplets`
   - 図: `fig09_ffno_electromagnetic_field_triplets`
   - 図: `fig10_ffno_ne_best_median_worst_maps`
   - 図: `fig11_ffno_ni_best_median_worst_maps`
   - 図: `fig12_ffno_midheight_line_profiles`
   - 図: `fig13_density_spatial_error_components`
   - 図: `fig14_ground_truth_eight_field_test_case`

7. 最適化への接続
   - 主 surrogate: `8-field + icp_part_sdf_lite_v1 + FFNO E80`。
   - UNet は候補設計の cross-check に使う。
   - FFNO だけで改善し UNet で崩れる候補は surrogate risk が高い設計として扱う。
   - 将来 COMSOL 再計算候補は、uniformity だけでなく density level, negative/finite rate, magnetic/current consistency で選ぶ。

## Figure Captions

- `fig01_dataset_design_space`
  - ICP_stage4 の coil count 分布と入出力設定を示す。60構造 x 6 process 条件の構成を最初に説明する図。

- `fig02_feature_ablation_mean_r2`
  - 構造特徴量の違いによる精度差。`part_lite_v1` では情報落ちが大きく、fixed slot SDF により性能が回復することを示す中核図。

- `fig03_e80_per_target_r2`
  - FFNO と UNet の target 別 R2。FFNO が密度を含め全体に高く、UNet は cross-check として妥当であることを示す。

- `fig04_training_curves_e80`
  - 80 epoch 学習の train/validation loss と validation score。loss は対数軸で、初期収束と後半改善を同時に見せる。

- `fig05_e20_to_e80_improvement_ffno`
  - 20 epoch では弱かった density field が 80 epoch で改善することを示す。前回の低精度への説明にも使える。

- `fig06_ffno_e80_regional_r2_heatmap`
  - FFNO の target x region R2。境界近傍を含む spatial fidelity の説明に使う。

- `fig07_density_distribution_errors_ffno`
  - `ne`, `ni` の分布指標。単なる pixel RMSE ではなく、積分値・高密度側・分布形状が保たれていることを示す。

- `fig08_ffno_core_field_triplets`
  - held-out test case について、`ne`, `ni`, `Te`, `phi` の `true / pred / err` をカラーバー付き線形スケールで並べる。密度は `10^18 m^-3` 表示。

- `fig09_ffno_electromagnetic_field_triplets`
  - 同じ case で `Br`, `Bz`, `Jelr`, `Jelz` の `true / pred / err` を示す。磁場は `mT`、電流密度は `A m^-2`。

- `fig10_ffno_ne_best_median_worst_maps`
  - FFNO の `ne` について、held-out case の best / median / worst を並べる。平均的に合う case と崩れる case の違いを説明する図。

- `fig11_ffno_ni_best_median_worst_maps`
  - FFNO の `ni` について、best / median / worst を同形式で示す。

- `fig12_ffno_midheight_line_profiles`
  - mid-height radial line 上の `ne`, `Te` の truth/pred profile。2次元図だけでなく、wafer 方向の分布差を読みやすくする図。

- `fig13_density_spatial_error_components`
  - density の integral / p99 / peak location / shape correlation / distribution score をモデル別に比較する図。空間分布評価を scalar R2 から補強するための図。

- `fig14_ground_truth_eight_field_test_case`
  - held-out test case の8個の目的場そのものを示す図。密度・温度・電位だけでなく、磁場と電子電流密度の空間スケールが異なることを説明する導入図。

## Cautions To State

- 現 dataset の process scalar は `pp`, `pp0` のみ。coil power, pressure, gas flow, gas mixture, RF frequency などを直接振るには、対応列を含む dataset 再生成が必要。
- `Br/Bz` は本来 chamber/valid-field 全体でも評価すべきだが、今回の benchmark 表は plasma-region metric を主に報告している。
- `icp_part_sdf_lite_v1` は固定 slot SDF なので、完全な permutation invariance ではない。ただし ICP_stage4 の 60 構造では coil radial order が物理的に意味を持つため、現時点では有効な表現である。
