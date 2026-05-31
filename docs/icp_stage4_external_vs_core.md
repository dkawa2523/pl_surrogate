# ICP_stage4 外部処理と本体基盤の責務

このメモは、ICP_stage4 を製品基盤の検証ケースとして使うときに、本体コードへ残す汎用機能と、ICP 専用の外部処理を分けるための整理です。

## 本体基盤に残すもの

本体は、特定のプラズマ変数名やコイル設計目的を知らない分布場 surrogate 基盤として保ちます。

- 構造入力は mask / SDF / boundary / part summary などの feature map として扱う。
- `target_region_by_var` で、任意の目的変数ごとに `plasma_only` / `all_domain` を選べる。
- `spatial_consistency.apply_region: target_region` で、分布連続性 loss を変数ごとの有効領域に合わせられる。
- 評価 CSV には region-aware metric と `target_region` を最小限だけ記録する。
- 推論後の raw field は互換性のため変更せず、領域外 mask は評価・可視化側で扱う。
- positive postprocess は、任意の正値変数に対する最終安全処理として扱う。
- optimize は `qoi_ratio`、penalty、reward などの汎用 objective 部品だけを評価する。

本体には ICP 固有の `ne`, `Te`, `Br`, `coil`, `z=150` などの意味を直書きしません。

## ICP_stage4 外部処理に置くもの

ICP の実験・発表・COMSOL 再計算に依存する処理は、`scripts/`、`configs/experimental/`、`reports/`、`external_tools/` 側に置きます。

- COMSOL 元データからの dataset 変換。
- `ne`, `ni`, `Te`, `phi`, `Br`, `Bz`, `Jelr`, `Jelz` の抽出、命名、単位、表示範囲。
- `Br` / `Bz` を `all_domain`、密度・温度・電位・電子電流を必要に応じて `plasma_only` とする ICP 用 mapping。
- `z=150` line、wafer line、mid-height profile などの発表目的に依存する評価。
- coil series、コイル数、COMSOL chamber 制約、再計算候補の出力。
- `uniformity / normalized density` など、ICP 設計に固有の目的関数。
- 学会・論文用の図、カラー設定、ケース選定。

## 推奨設定の形

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

eval:
  spatial_distribution_audit:
    enabled: true
    vars: all
```

coil series の設計変数は surrogate の scalar condition へ直接入れません。外部 runner が layout / mask / SDF / part feature に変換し、その構造特徴量を本体の推論へ渡します。

## 非対象

以下は本体基盤には入れません。

- ICP 固有 objective の直書き。
- raw mask / raw SDF optimization。
- COMSOL chamber 制約や coil count 制約の本体直書き。
- 大量の新規 diagnostics / manifest / registry。
- 発表用図に固有の field label、unit、color scale。
