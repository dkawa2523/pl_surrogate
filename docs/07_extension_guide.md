# 07 Extension Guide

This guide describes where to extend the package without spreading new
model-specific branches across the codebase.

## Principles

- `dataset.targets[]` is the source of truth for target variables.
- `output_layout.vars` preserves target order after preprocessing.
- `preprocessing.scalers.target_transforms` owns target transforms.
- Runtime input-mode support is declared in
  `src/plasma_surrogate/core/model_specs.py`.
- Keep strict validation at external boundaries: YAML, dataset loading,
  preprocessing artifacts, checkpoints, and model adapters.
- Avoid hidden fallbacks for production paths. Prefer clear configuration
  errors over silently reusing older m7 assumptions.

## Add Or Change A Model

Start with the stable capability metadata:

1. Add the model id to `src/plasma_surrogate/core/model_specs.py`.
2. Choose the model family and supported input modes.
3. Choose the allowed adapter modes and `auto` resolution.
4. Mark whether the model requires structure packs or belongs to a torch
   execution family.

Then wire the execution path:

1. Add the model implementation under `src/plasma_surrogate/models/`.
2. Add or update the minimal checkpoint build/save/load path.
3. Add the training adapter or dispatch branch.
4. Add the inference adapter only if the generic path cannot handle it.
5. Add benchmark support only after train and infer contracts are stable.

Required tests should be narrow:

- model construction
- checkpoint roundtrip
- train dispatch contract
- inference contract
- one smoke benchmark only if benchmark support is added

## Add Or Change Features

The same feature order must be available in preprocessing and inference.

Preferred flow:

1. Build the feature pack in preprocessing.
2. Persist channel metadata and transform statistics.
3. Load that metadata during inference.
4. Fail fast if a required channel is missing.

Avoid rebuilding equivalent feature lists independently in train, infer, and
benchmark code. If a feature is needed in more than one execution surface,
extract a shared builder before adding more conditionals.

## Add Or Change Target Transforms

Target transforms must be reversible and serialized.

Touch points:

- `src/plasma_surrogate/preprocessing/scalers.py`
- preprocessing validation
- checkpoint metadata, if the transform affects inference
- focused scaler tests

Do not add target-specific exceptions for `ne`, `ni`, `Te`, or `phi` unless the
configuration explicitly declares those targets.

## Add Or Change Physics Terms

Physics terms should resolve target names through symbol mapping instead of
fixed column names.

Touch points:

- `src/plasma_surrogate/train/physics_terms.py`
- loss composer integration
- symbol mapping validation
- one enabled/disabled behavior test

## Add Or Change Metrics Or Compare Outputs

Metrics and compare headers should be target-driven.

Keep dynamic patterns such as:

- `test_rmse_<var>`
- `test_r2_<var>`
- `test_r2_<var>_plasma_*`

Avoid fixed compare headers that assume a particular target set.

## Tests To Keep Small

Normal CI should focus on contracts and small smoke coverage:

- dataset contract
- scaler roundtrip
- input-mode policy
- model adapter contract
- checkpoint roundtrip
- one representative CLI or benchmark smoke per execution mode

Large benchmark sweeps, generated experiment inventories, and report
reproduction checks should live in a separate benchmark/nightly lane or outside
the core package test suite.
