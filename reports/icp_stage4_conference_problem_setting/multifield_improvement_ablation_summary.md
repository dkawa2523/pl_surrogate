# ICP_stage4 Multi-Field Improvement Ablation

## Purpose

The first 8-field `part_lite_v1` pilot had unexpectedly low density accuracy. I reran targeted ablations to separate three possible causes:

- the reorganized multi-field dataset itself,
- 8-target multi-task interference,
- loss/selection weighting,
- loss of coil-layout information in `part_lite_v1`.

All runs use the same reorganized dataset:

- `data/outputs_icp_stage4_enriched_360_csv_npz_multifield_structure_v1`
- scalar conditions: `pp`, `pp0`
- geometry source variables are not scalar inputs: `llcoil`, `rrc`, `nncoil`, `rrce`, `zzc`
- FFNO, 20 epochs, structure/extrapolation evaluation

## Runs

| Run | Targets | Structure profile | Mean plasma R2 | ne | ni | Te | phi | Br | Bz | Jelr | Jelz |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bad baseline | 8 fields | `part_lite_v1` | 0.158 | -0.227 | -0.229 | -0.043 | 0.378 | 0.548 | 0.264 | 0.204 | 0.367 |
| Core4-only check | 4 fields | `part_lite_v1` | -0.043 | -0.223 | -0.223 | -0.069 | 0.345 |  |  |  |  |
| Fixed-slot SDF | 8 fields | `icp_part_sdf_lite_v1` | 0.773 | 0.193 | 0.608 | 0.949 | 0.961 | 0.949 | 0.739 | 0.895 | 0.888 |
| Density weighted | 8 fields | `part_lite_v1` | 0.072 | -0.228 | -0.229 | -0.045 | 0.339 | 0.450 | 0.080 | -0.013 | 0.220 |

All runs produced finite predictions for all configured targets.

## Diagnosis

The main cause is not the multi-field dataset conversion. Earlier direct checks showed `ne`, `ni`, `Te`, and `phi` match the previous Core4 dataset inside the plasma mask.

The main cause is also not simply the number of target fields. When I removed `Br`, `Bz`, `Jelr`, and `Jelz` but kept `part_lite_v1`, density accuracy stayed poor.

The main cause is the structure representation. `part_lite_v1` compresses coil layout to order-invariant nearest/second/gap/proximity summaries. That is too lossy for this ICP_stage4 dataset. ICP density depends strongly on the radial sequence and per-coil location of the source region. The fixed-slot SDF profile keeps that information, and the model immediately recovers high validation and test accuracy.

Loss reweighting does not solve the issue. Increasing density target weights on `part_lite_v1` still gives negative `ne/ni` R2, so the missing information cannot be recovered by optimization alone.

## Decision

Use `icp_part_sdf_lite_v1` as the current multi-field baseline for this dataset.

For the conference problem setting:

- Keep the principle that coil parameters are not scalar model inputs.
- Use geometry-derived structure feature maps.
- For the current 60-structure ICP_stage4 dataset, fixed per-coil SDF slots are the reliable feature representation.
- Treat `part_lite_v1` as too compressed for density surrogate learning unless it is extended with richer order-invariant radial/layout descriptors.

## Recommended Next Step

Run the accepted `icp_part_sdf_lite_v1` 8-field config for 80 epochs and cross-check with one non-FFNO model. For near-term optimization, use fixed-slot SDF features or `struct_spatial_v1`, not the current `part_lite_v1` summary alone.

Important remaining caveat: `ne` is still weaker than the other fields in the 20-epoch 8-field SDF run (`R2 = 0.193`). `ni`, `Te`, `phi`, `Br`, `Bz`, `Jelr`, and `Jelz` are strong enough to show the representation is working, but density-driven optimization should use an 80-epoch retrain and ideally a Core4 density-focused surrogate for confirmation.

