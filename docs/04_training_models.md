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

## Loss Protocol

The implemented product protocol is intentionally small:

```yaml
train:
  loss:
    protocol: plasma_surrogate_v2
```

It resolves to supervised `type: mse` and `mask: plasma_only` unless the user
overrides `supervised.type` or `supervised.mask`. `supervised.type: huber` is an
opt-in alternative with a finite positive `huber_delta`; it is supported by the
NumPy and Torch supervised loss composers. Research loss knobs outside that
minimal supervised contract are rejected by the current implementation.

`train.loss.group_weighting` is optional and observationally narrow:

- `none` is the default and preserves the existing summed per-target loss.
- `uniform_by_target` averages target losses uniformly.
- `uniform_by_group` resolves target groups from `target_role_schema.json` and
  gives each non-empty group equal weight. It also reports
  `loss_supervised_group_<group>` in training history. Missing or incomplete
  target role metadata fails fast for this mode.

Positive target metadata is used by evaluation/benchmark sign diagnostics. It
does not currently add a separate positive training loss.

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
