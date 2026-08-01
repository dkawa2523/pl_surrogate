# Responsibility Boundaries Review

## Scope

Reviewed imports and shared helpers around `core`, `preprocessing`, `train`,
`eval`, `infer`, and `benchmark` after the target-group, diagnostics, and head
mode changes.

## Intended Direction

```text
core
  |
  +--> preprocessing --artifacts--> train --checkpoint--> infer
  |          |                         |
  |          +-------------------------+--> eval
  |
  +--> models

benchmark
  +--> orchestrates preprocessing, train, infer, and eval
```

`core` owns reusable contracts and pure numerical helpers. `benchmark` is the
only layer that intentionally imports across the main workflow lanes.

## Changes Made

- Moved pure NumPy physics helpers used by train, eval, infer, and model heads
  to `core/physics_numeric.py`.
- Updated eval payloads, inference diagnostics, benchmark summary calculation,
  and the plasma head to use `core.physics_numeric` instead of importing
  `train.losses`.
- Moved spatial feature construction helpers from `train/spatial_features.py`
  to `preprocessing/spatial_features.py`.
- Updated preprocessing, inference, and training callers to import spatial
  feature helpers from `preprocessing.spatial_features`.
- Removed the old `train/spatial_features.py` module after all in-repo callers
  moved to the new location.

## Boundary Audit

- `core` has no direct imports from `train`, `infer`, or `benchmark`.
- `eval` no longer imports `train.losses`; metric payloads depend on pure core
  helpers only.
- `infer` no longer imports `train.losses` or `train.spatial_features`.
- `preprocessing` no longer imports `train.spatial_features`.
- `target_groups` resolution remains centralized in `core/target_groups.py`.
- `benchmark` still imports train dispatch/protocol code because it is the
  workflow orchestrator; this is intentional.

## Remaining Large Files

- `benchmark/runner.py` still carries broad orchestration for CV, probes,
  metric rows, diagnostics, and artifacts. Keep future changes as small helper
  extractions only when a helper is reused or improves a clear boundary.
- `train/model_dispatch.py` and `train/grid_training.py` still coordinate many
  model-family paths. Existing adapter/contract modules help, but a full plugin
  rewrite remains intentionally out of scope.
- `eval/core_metrics.py` still combines row assembly with quality-score and
  diagnostic column selection. Keep default metrics compact and push expanded
  diagnostics behind `eval.diagnostics.enabled`.
- `infer/engine.py` remains the inference orchestrator. Derived field
  calculations and checkpoint/runtime validation are already split out.

## Guardrails

- Do not add train imports back into `eval`, `infer`, or `preprocessing` for
  shared math or feature helpers.
- Keep `core` free of workflow-layer imports.
- Prefer `core.target_groups` for group resolution and
  `preprocessing.spatial_features` for spatial feature construction.
- Keep `benchmark` as orchestration, not as a source of reusable computation.
