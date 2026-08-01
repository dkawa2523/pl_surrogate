# Final Completion Review

## Verdict

This codebase can be treated as complete for the current objective, with the
optional experiment lanes below frozen rather than expanded in this pass.

The mainline is now target-driven, keeps `shared` allvars as the baseline, and
separates implemented product contracts from planned or research-only lanes.

## Evidence

| Item | Result |
| --- | --- |
| Branch | `model_4-6_conf` |
| Test command | `py -m pytest -q` |
| Test result | `500 passed, 34 skipped, 138 deselected in 3.19s` |
| Diff stat command | `git diff --stat` |
| Diff stat summary | `181 files changed, 3747 insertions(+), 9620 deletions(-)` |

## Complete

| Completion condition | Final status |
| --- | --- |
| Shared allvars baseline is intact | Complete. Full pytest and torch benchmark smoke were passing after the cleanup sequence. |
| No new target-name-fixed product logic | Complete. New runtime paths use `dataset.targets[]`, `target_role_schema`, `output_layout.vars`, and `physics.symbols`; remaining `ne/ni/Te/phi` names are fixtures, synthetic sample ids, or local physics variable names. |
| Target transforms live in preprocessing | Complete. `dataset.targets[].value_transform` is not the runtime transform source; reversible transforms/scalers/clipping are in `preprocessing.scalers.target_transforms` and `TransformBundle`. |
| TargetGroupResolver exists | Complete. `core/target_groups.py` resolves groups from `output_layout.vars` and `target_role_schema`, with strict custom-group validation. |
| Group-wise metrics exist | Complete. Group RMSE/R2 and plasma group RMSE/R2 are emitted as dynamic metric columns. |
| `role_grouped` / `custom_groups` are opt-in | Complete. `shared` remains default; grouped heads are limited to supported grid/operator families. |
| Unsupported model family fail-fast | Complete. Grouped head validation rejects unsupported families. |
| Physics residual follows `physics.terms` | Complete. Removed legacy physics keys are rejected; `LossComposer` composes resolved terms while term registration/config stays in the physics contract path. |
| Diagnostics are not overgrown by default | Complete. Default rows keep target/group metrics; region/positive/detail diagnostics are opt-in. |
| Old docs, unused code, legacy paths reduced | Complete for this pass. Old experiment configs and docs were removed or rewritten; removed-key guards remain only as fail-fast rejection. |
| Tests pass | Complete. Latest full pytest result is recorded above. |
| Docs separate implemented contract from optional lanes | Complete. Product docs and extension guide distinguish baseline, opt-in, planned, and research lanes. |

## Optional And Frozen

| Lane | Frozen decision |
| --- | --- |
| Positive training loss | Keep frozen. Positive diagnostics are report-only; no default training penalty. |
| Poisson hybrid group head | Keep frozen. `poisson_hybrid` group option fails fast until implemented and benchmarked. |
| Full ModelPlugin system | Keep frozen. Current registries/adapters are sufficient; no unused plugin abstraction added. |
| Full FieldLayout/DataAdapter | Keep frozen. `field_layout.json` remains metadata-only; `output_layout.json` remains runtime source. |
| Graph/time operator support | Keep frozen. Current product lane is fixed grid2d. |
| Additional PINN/PINO terms | Keep frozen. Do not register names until validation and loss equations exist. |

## Research Lane

| Lane | Scope |
| --- | --- |
| Role-grouped architecture comparison | Run as opt-in benchmark against shared allvars; do not make default without evidence. |
| Custom target groups | Use for explicit ablations; strict validation is the recommended product behavior. |
| Positive physical-space training penalty | Requires differentiable Torch inverse transforms before loss integration. |
| Potential-specific hybrid heads | Compare only after density/temperature/electrostatic group metrics show where errors concentrate. |
| Additional physics residuals | Implement only with `physics.terms` validation, symbol resolution, and focused tests. |

## Next PR Candidates

| Candidate | Why it is next, not now |
| --- | --- |
| Shrink `benchmark/runner.py` further | Still large, but current behavior is tested; split only around reused orchestration helpers. |
| Normalize `model_type` metadata across all models | Would allow removing the remaining inference class-name alias map. |
| Add differentiable Torch transform inverse module | Needed before positive physical-space training losses. |
| Add benchmark evidence for grouped heads | Needed before changing any default head mode. |
| Reduce removed-key rejection guards in a major cleanup | Safe only when old configs are no longer expected in user workspaces. |

## Deleted

| Deleted item | Reason |
| --- | --- |
| Old `dual_input_modes` experimental configs | Unreferenced and contained obsolete fixed target assumptions. |
| Old `same_fidelity_low_data` experimental configs | Unreferenced and contained log-density/removed-knob experiments. |
| Old product docs for removed experiment lanes | Avoid presenting non-product paths as current contract. |
| Generated/report-like scripts and docs from older studies | Reduced product surface and reader noise. |
| `train/spatial_features.py` implementation module | Moved shared spatial feature helpers to `preprocessing/spatial_features.py`. |
| `target_groups_hash` metadata | Redundant with readable persisted `target_groups` metadata. |

## Known Limits

| Limit | Current behavior |
| --- | --- |
| `phi_mode` key name remains | Public config key is retained, but target resolution uses physics symbols/roles where runtime target selection matters. |
| Some fixed target ids remain in tests | They are sample data ids, not product branching logic. |
| Grouped heads are grid/operator only | POD, global MLP, and plasma-specific lanes stay shared or use their existing path. |
| Torch physics terms are narrower than NumPy terms | Unsupported enabled terms fail fast instead of silently degrading. |
| `field_layout.json` is not consumed by runtime | It is future metadata only; runtime still uses `output_layout.json`. |
| Full plugin architecture is absent | Third-party extension starts from `model_specs.py`, `models/factory.py`, and train adapters. |

## Final Extension Entry Points

| Extension | Start here |
| --- | --- |
| New target | `dataset.targets[]`, then `preprocessing.scalers.target_transforms.<target>` |
| New transform/scaler | `src/plasma_surrogate/preprocessing/scalers.py` |
| New model | `src/plasma_surrogate/core/model_specs.py`, then `models/factory.py` and checkpoint support |
| New train lane | `src/plasma_surrogate/train/model_adapters.py` only if no existing lane fits |
| New output head mode | `src/plasma_surrogate/models/heads/role_grouped.py` |
| New physics term | `core/physics_contract.py`, `train/physics_terms.py`, `train/losses.py` or `train/torch_losses.py`, then `train/loss_composer.py` |

## Completion Statement

No further feature work is required to satisfy the current objective. The repo
is ready to freeze this cleanup as a simple, target-driven baseline with
clearly marked optional research lanes.
