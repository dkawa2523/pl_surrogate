# Model and comparison contract

## Decision

The main comparison contains two separate questions. The formal pair tests the
adopted SDF and Dimension representations. The ABC pair tests actual-structure
SDF against a dimension-only information source while holding the adaptive UNO
architecture, channel count, magnetic carrier and training budget constant.

Cross-pair comparisons are descriptive. The causal representation comparison
is `formal_sdf_500` versus `formal_dimension_500`, and separately
`abc_sdf_500` versus `abc_dimension_p_500`.

## Model 1: formal SDF

- Conditions: `pp`, `pp0`.
- Spatial inputs:
  `x`, `y`, `mask_plasma`, `distance_signed`, `distance_any`,
  `part_sdf_union`, `part_source_mean`.
- Structure provider: actual parametric coil parts.
- Formal feature profile: `icp_coil_sdf_source_mean_v3`.
- UNO: width 64, five layers, 16 Fourier modes, shared output head.
- Operator response adapter: enabled, native scale 4, two padding cells,
  axisymmetric-Neumann radial padding.
- Initial weights:
  `configs/experimental/icp_stage4/conference_continuity_v45/initial_weights/conference_sdf_seed1237.npz`.

The only intended changes from the adopted configuration are the isolated
output path, the epoch budget of 500, recovery checkpoints and descriptive
metadata. Loss, split, optimizer and model semantics remain unchanged.

## Model 2: formal Dimension

- Conditions:
  `llcoil`, `rrc`, `nncoil`, `rrce`, `zzc`, `pp`, `pp0`.
- Spatial inputs:
  `x`, `y`, `mask_plasma`, `distance_signed`, `distance_any`.
- Structure provider: fixed chamber geometry.
- Formal feature profile: `geom_v1_mainline`.
- UNO, response adapter, targets and training objective: matched to formal SDF.
- Initial weights:
  `configs/experimental/icp_stage4/conference_continuity_v45/initial_weights/conference_dimension_seed1237.npz`.

For an irregular structure, the five geometry dimensions are the dataset's
legacy projection. They are not an exact description of independent coil
height, size or spacing.

## Model 3: ABC-SDF

ABC means the cumulative v46 architecture containing:

- A: a true multi-resolution U-shaped UNO.
- B: separate lifting and fusion for process, static geometry, coil geometry
  and electromagnetic inputs.
- C: a learned spatial gate between local convolution and global Fourier
  mixing.

Conditions are `pp` and `pp0`. Spatial input groups are:

| Group | Channels |
|---|---|
| Static | `x`, `y`, `mask_plasma`, `distance_signed`, `distance_any` |
| Actual structure | `part_sdf_union`, `part_source_mean`, `part_second_proximity`, `part_competition`, `solid_proximity` |
| Actual-structure vacuum EM | `vacuum_aphi_unit`, `vacuum_br_unit`, `vacuum_bz_unit`, `vacuum_bmag_unit` |

The vacuum fields are deterministic unit-current fields calculated from coil
geometry. They are not COMSOL plasma responses. The predicted targets remain
`ne`, `ni`, `Te` and `phi`; magnetic fields are not output targets in this
protocol.

The v46 ABC initial-weight file is retained as the reference initialization:
`experiments/icp_stage4/uno_structure_em_v46/configs/initial_weights/ABC_adaptive_mix_seed1237.npz`.

## Model 4: ABC-Dimension-P

`ABC-Dimension-P` is the required dimension-parameter comparator. `P` means
that irregular geometry is projected onto the regular-layout dimension
manifold before constructing the magnetic carrier.

The information source is strictly:

```text
d = (llcoil, rrc, nncoil, rrce, zzc)
p = (pp, pp0)
```

No actual structure SDF, actual individual-coil table, or actual-structure
vacuum field may enter this model.

### Dimension geometry group

The five normalized dimensions are broadcast as five constant spatial maps:

```text
dim_llcoil, dim_rrc, dim_nncoil, dim_rrce, dim_zzc
```

These maps replace the five actual-structure channels in ABC-SDF. This keeps
the geometry lifting tensor at five channels and preserves model capacity.

### Dimension-derived magnetic group

The regular geometry reconstructed from dimensions is:

\[
N=\operatorname{round}(nncoil),\qquad
\Delta r=(rrce-rrc)/N,
\]

\[
r_i=rrc+i\Delta r+\tfrac{1}{2}llcoil,\qquad
z_i=14+zzc+\tfrac{1}{2}llcoil,
\]

with every coil assigned common width and height `llcoil`. From this
reconstructed geometry, calculate the same four deterministic unit-current
vacuum channels:

```text
dimension_vacuum_aphi_unit
dimension_vacuum_br_unit
dimension_vacuum_bz_unit
dimension_vacuum_bmag_unit
```

These fields may differ from the actual-structure vacuum fields for an
irregular case. That difference is intentional: it is the information lost by
the dimension projection.

### Architecture match

ABC-Dimension-P must use the same width, Fourier modes, number of blocks,
multi-resolution path, local/global gate, output heads, response adapter and
dropout as ABC-SDF. Both have two process scalars and fourteen spatial
channels:

| Component | ABC-SDF | ABC-Dimension-P |
|---|---:|---:|
| Process conditions | 2 | 2 |
| Static spatial channels | 5 | 5 |
| Structure/dimension channels | 5 | 5 |
| Vacuum EM channels | 4 | 4 |
| Total spatial channels | 14 | 14 |

The final implementation must assert equal trainable parameter counts. A
failure is a contract error, not an acceptable caveat.

### Initialization match

Use the ABC-SDF initial tensor template for all tensors whose semantic meaning
is unchanged. Reinitialize only the five-channel geometry lifting tensor when
its source changes from actual spatial geometry to broadcast dimensions. Use
seed 1237 and record exact copied keys, element counts and copied fraction.
Do not copy an SDF-geometry lifting tensor into a dimension channel without
reinitialization.

## Conditions that must remain equal

- Dataset and fixed membership file.
- Seed 1237 and one model per variant.
- 500 completed epochs; early stopping disabled.
- Batch size 16 and case shuffling enabled.
- AdamW, learning rate `3e-4`, weight decay `1e-4`, betas `(0.9, 0.999)`,
  epsilon `1e-8`, cosine schedule.
- Targets, target transforms and physical-space inverse transforms.
- Plasma-only Huber supervision and density shape/inventory decomposition.
- Validation checkpoint-selection objective.
- Standard test and frozen unknown-structure evaluator.
- No response leakage.

## Prohibited comparators

- A dimension model receiving vacuum fields calculated from the actual
  irregular structure. This would secretly reintroduce structure information.
- A seven-scalar ABC model with no magnetic carrier. It would confound
  representation with missing physics inputs.
- Expanding the Dimension vector to independent per-coil positions or sizes.
  That is a different representation family and must be a separate study.
- Adding individual-structure COMSOL teacher cases only to one model.

