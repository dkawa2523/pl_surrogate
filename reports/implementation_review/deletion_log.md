# Deletion Log

## Deleted

- Removed `configs/experimental/dual_input_modes/`.
  - The directory was not referenced by current code, tests, scripts, or docs.
  - Its configs still used old fixed targets such as `log_ne`, `Te`, and `phi`.
- Removed `configs/experimental/same_fidelity_low_data/`.
  - The directory was not referenced by current code, tests, scripts, or docs.
  - Its configs contained removed product knobs such as log-density source
    transforms and region-balance era experiments.

## Replaced Or Simplified

- Removed the `log_ne` / `log_ni` special-case branch from
  `core/dataset_io.py`.
  - Mainline `csv_npz` loading now rejects non-identity
    `dataset.targets[].value_transform` generically instead of checking fixed
    density source names.
  - The matching unit test now checks the generic transform contract.
- Replaced a fixed `pred["phi"]` write in `train/torch_trainer.py` with a
  potential key resolved through `physics.symbols` / `target_role_schema`.
- Updated `models/heads/plasma_head.py` so the `predict_fields` fallback uses
  the resolved `potential_key` instead of requiring a hard-coded `"phi"` field.
- Cleaned old wording in the UNet and plasma-head comments that described
  current behavior as legacy compatibility.

## Kept Fixed Names And Why

- `phi`, `density`, and `temperature` local variable names remain inside
  numerical physics helpers and tests where they denote mathematical quantities,
  not product target ids.
- `ne`, `ni`, `Te`, and `phi` remain in fixtures and synthetic data because
  they are compact sample target ids used to exercise dynamic target handling.
- `lambda_poisson`, `lambda_bc`, `lambda_rho`, `density_guard`,
  `target_transform_policy`, `output_vars`, `output_key_map`,
  `output_value_transform`, and `allow_mode_fallback` remain only in
  fail-fast rejection guards and tests for removed keys.
- Checkpoint loaders still mention legacy formats in error messages because
  those paths reject old checkpoints instead of accepting compatibility aliases.
- `phi_mode` remains as the existing public configuration key. Internally, the
  field to post-process is resolved through physics symbols or target roles
  where runtime target selection matters.

## Deferred Deletion Candidates

- `infer/contracts.py` still maps some model class names to model ids for
  models that do not all expose `model_type` / `to_meta()` consistently. Remove
  that alias map only after every model class exposes stable metadata.
- `benchmark/runner.py` still contains broad orchestration for metrics,
  diagnostics, probes, and artifacts. Continue reducing it through small helper
  extraction only when the helper is reused or removes a clear boundary issue.
- Removed-key rejection guards can be dropped in a future major cleanup once
  old configs are no longer expected to appear in user workspaces.
