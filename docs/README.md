# Mainline Specification Index

本ディレクトリは、`plasma_surrogate` の mainline 基盤に対する仕様書です。  
運用入口は `README_mainline.md`、設計と拡張の正本は `docs/` とします。

## この仕様書の前提

- mainline 対象モデル: `global_mlp`, `unet`, `fno`, `deeponet_plasma`
- first-class 対象: 2D periodic
- target 名は固定しない
- target 定義の唯一の真実源は `dataset.targets[].id`
- 学習・推論・評価の target 順序の唯一の真実源は `output_layout.vars`
- 前処理契約の唯一の真実源は `preprocessing.scalers.target_transforms.<var>`
- strict 契約の入口は `src/plasma_surrogate/train/model_dispatch.py`
- m7 の `ne / ni / Te / phi` は現行サンプルであり、基盤契約そのものではない

## 読者別の読み順

### すぐに実行したい

1. [`../README_mainline.md`](../README_mainline.md)
2. [`08_workflows_examples.md`](08_workflows_examples.md)
3. [`06_yaml_reference.md`](06_yaml_reference.md)

### 新しいデータセットを載せたい

1. [`02_data_contract.md`](02_data_contract.md)
2. [`03_preprocess_and_features.md`](03_preprocess_and_features.md)
3. [`06_yaml_reference.md`](06_yaml_reference.md)
4. [`08_workflows_examples.md`](08_workflows_examples.md)

### 前処理や特徴量を追加したい

1. [`01_architecture.md`](01_architecture.md)
2. [`03_preprocess_and_features.md`](03_preprocess_and_features.md)
3. [`07_extension_guide.md`](07_extension_guide.md)

### モデルや physics term を追加したい

1. [`01_architecture.md`](01_architecture.md)
2. [`04_training_models.md`](04_training_models.md)
3. [`05_inference_and_evaluation.md`](05_inference_and_evaluation.md)
4. [`07_extension_guide.md`](07_extension_guide.md)

## 章構成

1. [`01_architecture.md`](01_architecture.md)  
   全体アーキテクチャ、カテゴリ責務、データフロー、truth source の位置

2. [`02_data_contract.md`](02_data_contract.md)  
   `dataset.targets[]` を中心としたデータセット契約

3. [`03_preprocess_and_features.md`](03_preprocess_and_features.md)  
   `cleanse / feature / preprocess` の責務、artifact、変換契約

4. [`04_training_models.md`](04_training_models.md)  
   各 mainline モデルの入出力、strict contract、設定入口

5. [`05_inference_and_evaluation.md`](05_inference_and_evaluation.md)  
   推論、物理項解決、評価、benchmark、compare の読み方

6. [`06_yaml_reference.md`](06_yaml_reference.md)  
   実用単位の YAML リファレンス

7. [`07_extension_guide.md`](07_extension_guide.md)  
   処理追加、モデル追加、physics term 追加時の接続点

8. [`08_workflows_examples.md`](08_workflows_examples.md)  
   benchmark / compare / plot / pipeline の実行例

## コード上の主要入口

- CLI 入口: `src/plasma_surrogate/cli/main.py`
- Task dispatch: `src/plasma_surrogate/pipeline/task_runner.py`
- Workflow 実装: `src/plasma_surrogate/cli/workflows.py`
- Dataset 読み込み: `src/plasma_surrogate/core/dataset_io.py`
- Preprocess 実装: `src/plasma_surrogate/preprocessing/runner.py`
- Scaler 実装: `src/plasma_surrogate/preprocessing/scalers.py`
- Train 契約入口: `src/plasma_surrogate/train/model_dispatch.py`
- Physics registry: `src/plasma_surrogate/train/physics_terms.py`
- Inference 実装: `src/plasma_surrogate/infer/engine.py`
- Benchmark 集計: `src/plasma_surrogate/benchmark/runner.py`
- Compare スクリプト: `scripts/compare_selected_models.py`

## この仕様書で特に強調する設計原則

- target 名や target 数をコードへ再ハードコードしない
- compare / benchmark の列は dynamic target ベースで扱う
- physics は target 名直参照ではなく symbol mapping を使う
- preprocess と infer で feature 契約をずらさない
- `tests/fixtures` はテンプレであり、実運用正本は将来的に `configs/` へ置く
