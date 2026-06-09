# ICP_stage4 構造特徴量ベース分布場 surrogate 問題設定

## 背景

ICP_stage4 は、低圧 ICP プラズマ装置におけるコイル構造・配置とプロセス条件が、2 次元空間分布場に与える影響を学習するための検証 dataset である。半導体製造装置では、密度や温度の均一性だけでなく、磁場、電流密度、境界近傍の分布まで含めて設計を評価する必要がある。

本問題の主題は、`llcoil`, `rrc`, `nncoil`, `rrce`, `zzc` のような寸法パラメータをそのまま学習入力にすることではない。これらは geometry generator の入力としてのみ使い、学習モデルには coil mask、SDF、boundary feature、part summary feature など、場を解く幾何そのものに近い構造特徴量として渡す。

## Dataset

主な対象 dataset は次である。

| 項目 | 内容 |
| --- | --- |
| 元データ | `data/outputs_icp_stage4_enriched_360` |
| Core4 pack | `data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_sdf_lite_v1` |
| Multi-field pack | `data/outputs_icp_stage4_enriched_360_csv_npz_multifield_structure_v1` |
| 構造数 | 60 groups |
| process 条件 | 各構造 6 条件 |
| 総 case 数 | 360 cases |
| grid | 約 440 x 600 |
| split | structure group 単位の train / val / test |

現行の multi-field pack では、次の 8 変数を 2 次元 target field として扱う。

| target | 物理量 | 推奨評価領域 |
| --- | --- | --- |
| `ne` | electron density | `plasma_only` |
| `ni` | ion density | `plasma_only` |
| `Te` | electron temperature | `plasma_only` |
| `phi` | electric potential | `plasma_only` |
| `Br` | magnetic field radial component | `all_domain` |
| `Bz` | magnetic field axial component | `all_domain` |
| `Jelr` | electron current density radial component | `plasma_only` |
| `Jelz` | electron current density axial component | `plasma_only` |

`Br` と `Bz` は chamber / coil 近傍を含む全領域で意味を持つ。一方、密度・温度・電子電流は主に plasma 領域で評価する。この違いは ICP 固有の mapping であり、本体コードには target 名を直書きしない。

## 入力変数の扱い

学習時の入力は、process scalar と構造特徴量に分ける。

- process scalar: dataset に含まれるプロセス条件列。
- geometry source: `llcoil`, `rrc`, `nncoil`, `rrce`, `zzc`。
- surrogate input: process scalar + structure feature maps。

geometry source は audit、構造生成、最適化候補生成に使うが、surrogate の scalar condition として直接入れない。これにより、coil 数の整数変化、inactive slot、同等 layout の別表現、slot 順序依存といった raw parameter model の弱点を避ける。

## 構造特徴量化の意義

寸法パラメータを直接使う従来型 surrogate では、次の課題が出やすい。

- 同じような coil layout が異なる parameter vector で表現される。
- `nncoil` の変化で topology が不連続になる。
- inactive coil slot や coil order に過敏になる。
- 境界、solid、coil 近傍、plasma 領域との相対配置をモデルが直接見られない。
- 構造数が少ない dataset では、raw vector 外挿に過信しやすい。

構造特徴量ベースでは、モデルは寸法列ではなく、場の支配方程式が解かれる空間構造を入力として見る。推奨 profile は `part_lite_v1` または互換 baseline の `icp_part_sdf_lite_v1` である。

## 学習・評価設定

汎用基盤の設定は、任意 target 名に対して有効領域を指定する。

```yaml
train:
  loss:
    supervised:
      target_region_by_var:
        ne: plasma_only
        ni: plasma_only
        Te: plasma_only
        phi: plasma_only
        Br: all_domain
        Bz: all_domain
        Jelr: plasma_only
        Jelz: plasma_only
      boundary_weight:
        enabled: true
        vars: [ne, ni, Te, phi]
      spatial_consistency:
        enabled: true
        apply_region: target_region
```

評価では R2 だけでなく、分布としての再現性を見る。

- region-aware RMSE / R2
- peak location error
- center-of-mass error
- radial / axial profile RMSE
- gradient RMSE
- shape correlation
- true / pred / error の mask 付き空間分布図

## 推論最適化の位置づけ

ICP 固有の目的関数は外部 runner に置く。例えば z=150 line profile、wafer 上 uniformity、density collapse penalty、coil overlap 制約、COMSOL 再計算候補選定は、本体 optimize ではなく ICP 用の script / config / report で扱う。

本体 optimize は、generic objective parts と geometry provider を呼ぶだけに留める。raw mask optimization や raw SDF optimization は行わず、必ず geometry source から構造特徴量を再生成して surrogate 推論する。

## 参照

本体基盤と ICP 外部処理の責務分離は [icp_stage4_external_vs_core.md](icp_stage4_external_vs_core.md) を参照する。

