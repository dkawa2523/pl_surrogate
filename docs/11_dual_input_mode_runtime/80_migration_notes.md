# 80. Migration Notes

## 09 / 10 章からの関係

- `09_same_fidelity_low_data_extensions.*` はモデル拡張の計画
- `10_structural_geometry_feature_system.*` は structure-aware feature 基盤の計画
- 本章 `11_dual_input_mode_runtime.*` は、**それらを train / infer / benchmark の入口で mode に分けて整理し直す** 章

## 今回の追加で変わる考え方

### 以前
- structure feature system を入れる前提で設計しやすかった
- どの model が cond-only baseline なのかが runtime 上あいまいになりやすい

### これから
- まず `table_only` か `table_plus_structure` を選ぶ
- structure-aware 実験は `table_plus_structure` に閉じ込める
- cond-only baseline は `table_only` に閉じ込める

## 既存コードの解釈変更

- fixed geometry は今後 **output context** と **structure input** を区別して扱う
- `coord_feature_pack` は常に存在するものではなく、`table_plus_structure` でのみ first-class
- `OptimizeRunner.space` は cond-only optimization の表現であり、structure mode では `geom_space` を追加する

## 非互換を避ける方針

- mainline の YAML は `runtime.input_mode` を明示しなくても、既存挙動に一致するように migration helper で解決してよい
- ただし新規 experimental config は必ず `runtime.input_mode` を明示する
- silent fallback は warning ではなく error を既定にする
