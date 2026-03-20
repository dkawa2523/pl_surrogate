# Mainline Runbook

mainline の最短運用入口です。  
設計、仕様、YAML 詳細、拡張方法は [`docs/README.md`](docs/README.md) を参照してください。

## Mainline Scope

- Models: `global_mlp`, `unet`, `fno`, `deeponet_plasma`
- Stage categories: `cleanse`, `feature`, `preprocess`, `train`, `infer`, `evaluate`, `pipeline` (internal task: `pipeline.run`)
- Target contract: fixed namesは不要。`dataset.targets[].id` が唯一の target 定義

## Contract Essentials

- `dataset.targets[]` is required
- preprocess contract is `preprocessing.scalers.target_transforms.<var>`
- target columns in benchmark / compare are generated dynamically from active targets
- inference physics uses:
  - `inference.ood.physics.symbols`
  - `inference.ood.boundary_operator.symbols`

## Mainline Fixture Set (m7 Example)

- `tests/fixtures/benchmark_periodic_real_m7_global_frozen_ref.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_unet_isolated_mainline.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_fno_isolated_mainline.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_deeponet_isolated_mainline.yaml`
- `tests/fixtures/benchmark_periodic_real_compare_mainline_fno_deeponet_global.yaml`

## Quick Commands

### Pipeline

```bash
PYTHONPATH=src .venv-torch/bin/python -m plasma_surrogate.cli.main pipeline --config configs/mainline.yaml
```

### Benchmark

```bash
PLASMA_SURROGATE_ENABLE_TORCH=1 PYTHONPATH=src .venv-torch/bin/python -m plasma_surrogate.cli.main benchmark run --config tests/fixtures/benchmark_periodic_real_m7_fno_isolated_mainline.yaml
PLASMA_SURROGATE_ENABLE_TORCH=1 PYTHONPATH=src .venv-torch/bin/python -m plasma_surrogate.cli.main benchmark run --config tests/fixtures/benchmark_periodic_real_m7_deeponet_isolated_mainline.yaml
```

### Compare

```bash
PYTHONPATH=src .venv-torch/bin/python scripts/compare_selected_models.py --config tests/fixtures/benchmark_periodic_real_compare_mainline_fno_deeponet_global.yaml
```

## Outputs to Check

- `runs/.../leaderboard.csv`
- `runs/.../resolved_benchmark.json`
- `runs/.../selected_models_comparison.csv`
- `runs/.../artifacts/<stage>/manifest.json`

## Notes

- `tests/fixtures` are templates. Production configs should live under `configs/`.
- m7 uses `ne`, `ni`, `Te`, `phi`, but mainline itself is target-agnostic.
- 詳細:
- 全体設計: [`docs/01_architecture.md`](docs/01_architecture.md)
- データ契約: [`docs/02_data_contract.md`](docs/02_data_contract.md)
- 前処理と feature: [`docs/03_preprocess_and_features.md`](docs/03_preprocess_and_features.md)
- 学習モデル: [`docs/04_training_models.md`](docs/04_training_models.md)
- 推論と評価: [`docs/05_inference_and_evaluation.md`](docs/05_inference_and_evaluation.md)
- YAML リファレンス: [`docs/06_yaml_reference.md`](docs/06_yaml_reference.md)
- 拡張ガイド: [`docs/07_extension_guide.md`](docs/07_extension_guide.md)
- 実行例: [`docs/08_workflows_examples.md`](docs/08_workflows_examples.md)
