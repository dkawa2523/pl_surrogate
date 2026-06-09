# ICP_stage4 8-Field `icp_part_sdf_lite_v1` E80 Model Comparison

## Setup

- Dataset: `data/outputs_icp_stage4_enriched_360_csv_npz_multifield_structure_v1`
- Input mode: `table_plus_structure`
- Scalar conditions: `pp`, `pp0`
- Geometry source variables are not scalar model inputs: `llcoil`, `rrc`, `nncoil`, `rrce`, `zzc`
- Structure profile: `icp_part_sdf_lite_v1`
- Targets: `ne`, `ni`, `Te`, `phi`, `Br`, `Bz`, `Jelr`, `Jelz`
- Epochs: 80
- Split/eval: extrap/structure holdout protocol used by the existing ICP_stage4 benchmark runner

UNet uses `output_heads.mode: shared` because the previous density/field split head supports only Core4-style targets and cannot emit `Br`, `Bz`, `Jelr`, `Jelz`.

## Results

| Model | Mean plasma R2 | ne | ni | Te | phi | Br | Bz | Jelr | Jelz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FFNO | 0.960 | 0.964 | 0.983 | 0.975 | 0.986 | 0.978 | 0.923 | 0.946 | 0.929 |
| UNet | 0.830 | 0.488 | 0.683 | 0.969 | 0.962 | 0.929 | 0.885 | 0.861 | 0.862 |

All reported target fields have finite ratio `1.0` on the plasma metric region.

## Interpretation

The 80 epoch FFNO run resolves the earlier density failure. The same multi-field dataset and preprocessing now gives high R2 for all eight target fields once the coil layout is represented by fixed-slot per-coil SDF channels and training is extended to 80 epochs.

UNet also learns the 8-field problem, especially `Te`, `phi`, `Br`, `Bz`, `Jelr`, and `Jelz`, but it is much weaker on density than FFNO. For density-driven coil optimization, FFNO is the primary candidate and UNet should be used as a cross-check rather than the main surrogate.

## Output Paths

- FFNO leaderboard: `runs/icp_stage4_multifield_8field_part_sdf_lite_v1_e80/full/ffno/leaderboard.csv`
- UNet leaderboard: `runs/icp_stage4_multifield_8field_part_sdf_lite_v1_e80/full/unet/leaderboard.csv`
- Status CSV: `runs/icp_stage4_multifield_8field_part_sdf_lite_v1_e80/run_status.csv`

## Decision

Use `8-field + icp_part_sdf_lite_v1 + FFNO E80` as the current multi-field surrogate baseline.

Use UNet E80 as an independent sanity check for candidate designs. A candidate whose density/uniformity improvement appears only in FFNO and not in UNet should be treated as surrogate-risky and prioritized lower for COMSOL recalculation.

