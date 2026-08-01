# 04 Training And Models

Training reads preprocessing artifacts and model capabilities. The first
question for a new model is what its product capability is, not where to add a
special branch.

## Training Contract

- Targets come from `dataset.targets[]` and `output_layout.vars`.
- Roles come from `target_role_schema.json`.
- Features come from feature pack metadata and `channel_map.json`.
- Runtime metadata must include concrete target and feature schema hashes.
- Checkpoints must preserve the effective runtime metadata used for training.

## First-Class Model Registry

`src/plasma_surrogate/core/model_specs.py` is the source of truth for model
status, category, input modes, adapter modes, and feature needs. Train, infer,
and benchmark dispatch should use these specs and local adapters.

Adapter names describe inputs that the runtime actually consumes. FNO/grid
models use `grid_pack`; coordinate MLP and mainline plasma DeepONet use
`coord_pack`; descriptor adapters are limited to the POD/geometry model lanes
that explicitly load a structure-descriptor artifact. Unsupported combinations
fail at the model/input policy gate instead of accepting and ignoring a
descriptor.

## Spatial-Field Mainline

Spatial plasma surrogates use one contract across architectures:

1. the dataset declares target roles and the plasma geometry;
2. the model declares whether it consumes spatial features;
3. the loss composes case-balanced point, boundary, gradient, and multiscale
   terms; and
4. checkpoint selection uses a case-macro spatial validation objective.

This keeps FNO, coordinate operators, POD-residual models, and field-output MLP
baselines on the same supervision and evaluation semantics. Model-specific
branches belong only in the model adapter, not in the loss or inference path.

Model input geometry and objective geometry are separate contracts. For
case-varying structure, preprocessing persists a static spatial pack and a
case-aligned structure pack. Raw `mask_plasma` and configured boundary-distance
channels are resolved independently for scaler fitting, training loss,
checkpoint selection, and evaluation; they are not inferred from a model's
transformed input channels.
Case packs must preserve dataset `case_id` order. If a pack declares
case-varying raw geometry, partial raw geometry is rejected instead of mixing
it with a static fallback.

## Loss Protocol

New spatial-field runs should use the versioned product protocol:

```yaml
train:
  loss:
    protocol: plasma_surrogate_v3
```

Version 3 resolves to Huber supervision on `plasma_only`, `sample_mean`
normalization, equal target-family weighting, and small gradient, multiscale,
and boundary terms. Every case is reduced before the batch is reduced, so a
large plasma mask cannot dominate a small one. Spatial terms share the same
active mask; non-finite values outside that mask are ignored, while non-finite
active values fail fast. The boundary is defined by
`supervised.spatial.boundary_distance_channels` (default: `[distance_any]`)
and uses the minimum absolute distance across all configured raw channels.
This allows a structure-varying run to include part or electrode SDFs without
hard-coding plasma-specific channel names in the loss. Every configured channel
is mandatory and must be finite and case-aligned.

For fields whose spatial derivatives have very different magnitudes, set
`gradient_normalization: target_rms`. Each case's gradient residual is divided
by its masked target-gradient RMS, with `gradient_epsilon` as a positive floor.
The scale depends only on the target, so prediction rescaling cannot reduce the
objective. Use the same settings in `selection.spatial` to keep training and
checkpoint selection aligned.

`plasma_surrogate_v2` remains available unchanged for old-run reproduction. It
uses the earlier point-only defaults and does not silently opt in to spatial
terms.

`train.loss.group_weighting` controls how complete per-target losses are
combined:

- `uniform_by_group` is the version-3 default. It assigns equal share to each
  non-empty `field_family`, then divides that share among the family's targets.
  The authoritative `target_role_schema.json` must cover every trained output.
- `uniform_by_target` averages target losses uniformly.
- `weighted_by_group` uses explicit positive family weights.
- `none` preserves the legacy summed per-target behavior and is the version-2
  default.

Group modes report `loss_supervised_group_<group>` in training history. Missing
or incomplete target role metadata fails fast rather than silently reverting to
per-channel averaging. The resolved loss definition, version, and hash are
stored with the run/checkpoint artifacts.

Positive target metadata is used by evaluation/benchmark sign diagnostics. It
does not currently add a separate positive training loss.

## Checkpoint Selection

For spatial-field runs use `selection.mode: best_val_spatial_objective`. It is
lower-is-better and scores each validation case separately in the same
train-fitted normalized field space. The objective combines point, gradient,
and boundary RMSE, then adds median, p90, and worst-case penalties. If target
roles are available, target families receive equal weight. Pooled validation
R2 modes remain legacy compatibility options and should not be interpreted as
evidence that local field structure is correct.

POD-based models may expose a coefficient-space auxiliary loss. Trainers add
the model-reported `loss_aux_total` to the optimization objective and record
the raw and weighted coefficient terms separately. The field loss remains the
common cross-architecture contract.

## DeepONet Geometry Contract

The mainline `deeponet_plasma` trunk receives coordinates and local spatial
features for each query. For case-varying geometry, set
`branch_mode: set_mlp_pool` and `sensor_pool_mode: set_mlp_pool`. An order-invariant
encoding of the case-level sensor set supplies non-local structure context to
every query. `cond_only` plus `moments` remains a compatibility mode for
fixed-geometry studies, but it cannot represent non-local changes that are
absent from the local trunk row. The mainline runtime consumes `coord_pack`;
it does not consume descriptor adapters.

## Physics Training

Physics residual training follows the `physics.terms` flow. `train.loss`
selects and combines supervised data loss, while `physics.terms` is the source
of truth for physics residual names, enable flags, and weights.

Registered physics terms are currently `poisson`, `boundary`,
`boundary_operator`, and `rho`. Unknown term names fail fast; names for planned
or experimental residuals are not registered until they have code and tests.

Target symbols are resolved from explicit top-level `physics.symbols` first,
then unique role/family metadata from `target_role_schema.json`. Term-local
`symbols` are not supported. Missing or ambiguous symbols fail fast.

## Output Heads

The default and shared allvars baseline is `output_heads.mode: shared`.
Grid/operator models can opt in to grouped heads:

- `role_grouped`: groups targets from `target_role_schema.json` field-family metadata.
- `custom_groups`: uses explicit `output_heads.groups.<name>.targets`.

Grouped heads are limited to grid/operator families (`fno`, `ffno`, `unet`,
`unetpp`, `unetpp_attn`, `u_no`, `cno`). The trunk remains shared and only the
lightweight final heads are split. `custom_groups` is strict by default: every
`output_layout.vars` target must appear in exactly one configured group.

`output_heads.group_options` is reserved for optional experiment lanes on
resolved groups. The only implemented group head is `head: default`; configured
`head: poisson_hybrid` currently fails fast and is left for benchmark-backed
follow-up work. Potential/electrostatic groups must be resolved from roles or
field-family metadata, not from a fixed target name.

## Archive Boundary

Experimental model variants and one-off studies belong in `configs/experimental/`
or `experiments/`. They should not add product contracts until train, infer, and
benchmark behavior is stable.
