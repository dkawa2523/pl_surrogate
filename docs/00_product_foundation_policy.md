# 00 Product Foundation Policy

This repository is a target-driven plasma surrogate product foundation. The
mainline code should connect dataset targets, preprocessing artifacts, model
capabilities, inference outputs, evaluation metrics, and benchmark selection
without relying on historical experiment shortcuts.

## Keep

- Define all prediction targets through `dataset.targets[]`.
- Treat `preprocessing/schema/output_layout.json` as the canonical target order.
- Treat `preprocessing/schema/target_role_schema.json` as the canonical role and
  positivity metadata.
- Treat `preprocessing.scalers.target_transforms` and
  `preprocessing/scalers/y_scalers.json` as the transform/scaler contract.
- Reuse preprocessing artifacts for train, infer, evaluation, and benchmark.
- Resolve physics symbols from explicit symbols or target roles, not hard-coded
  target names.
- Keep `output_heads.mode: shared` as the first-class default; grouped heads are
  opt-in and must stay target-driven.
- Use `surrogate_quality_score` as the default benchmark selection metric.
- Use `inference.optimize.objective.mode: weighted_sum` for product
  optimization objectives.

## Remove

- Fixed target-name branches in product logic.
- Treating `dataset.targets[].value_transform` as the runtime transform source.
- Describing planned loss or head behavior as implemented product contract.
- Deprecated objective aliases outside a loader or migration shim.
- Generated reports, fixture dumps, and one-off experiment notes from product
  docs.
- Placeholder runtime hashes in train, infer, benchmark, or checkpoint metadata.
- Model-specific policy copied into multiple entrypoints instead of model specs
  and adapters.

## Extension Order

1. Stabilize target role schema and runtime contracts.
2. Keep loss, metrics, and evaluation protocol target-driven.
3. Add model capability in `model_specs.py` before train or infer wiring.
4. Add benchmark support only after train and infer contracts are stable.
5. Keep dataset-specific studies in `configs/experimental/` or `experiments/`.

## Non-Goals

- Rewriting the scientific model families in this cleanup.
- Preserving every historical experiment path as a product interface.
- Treating generated data, runs, reports, or zip files as source code.
