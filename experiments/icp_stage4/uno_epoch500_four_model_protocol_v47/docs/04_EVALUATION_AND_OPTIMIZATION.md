# Evaluation and optimization protocol

## Evaluation populations

Every selected checkpoint is evaluated on three fixed populations:

1. Standard test: 136 cases from the frozen membership file.
2. Unknown structures: 75 G4 cases, 25 from each of
   `unknown_gap_topology`, `unseen_rank_height_size`, and
   `coupled_transform`.
3. Matched anchors: 25 regular G4 cases at the same coil count and operating
   condition as the unknown structures.

The unknown-plus-anchor population must contain exactly 100 cases. Any count
mismatch fails evaluation.

## Primary accuracy metrics

Use physical, inverse-transformed fields and the plasma mask.

- `ni_rel_l2_pct`: plasma-masked two-dimensional ion-density relative L2.
- `bohm_profile_rel_l2_pct`: relative L2 of the wafer-near radial Bohm profile.
- `bohm_uniformity_abs_error_pp`: absolute error in maximum local Bohm-flux
  deviation, in percentage points.
- Structural response: compare `(unknown - matched anchor)` for prediction and
  COMSOL, separately for the 2-D `ni` field and the Bohm profile.

The Bohm flux is:

\[
\Gamma_i=n_i\sqrt{T_e e/m_{Ar}}.
\]

Use ten plasma layers above the wafer and twenty equal-area radial bins. Define
profile nonuniformity as:

\[
U_\Gamma=100\max_r\left|\Gamma_i(r)/\overline{\Gamma_i}-1\right|.
\]

Lower values are better for every error and nonuniformity metric.

Report median, 90th percentile and worst case. Also report paired per-case
differences and a case-bootstrap 95% interval. Do not hide family-specific
regressions inside an overall average.

## Optimization track A: matched regular-layout search

All four models optimize the same representable design family. Enumerate
`nncoil` in `{2,3,4,5,6}` and optimize:

```text
llcoil, rrc, rrce, zzc
```

Use the existing qualified bounds:

| Variable | Lower | Upper |
|---|---:|---:|
| `llcoil` | 0.55344 | 1.44492 |
| `rrc` | 2.49330 | 9.52599 |
| `rrce` | 20.59640 | 29.38050 |
| `zzc` | 0.29864 | 4.69076 |

Require at least `0.20 * llcoil` clearance in addition to the coil width.
Fix the operating condition to the frozen `center` condition unless a separate
operation sweep is declared:

```text
pp  = 1758.995
pp0 = 0.017931
```

Use optimization seed 4099. Give each model exactly the same candidate and
local-refinement budget. A recommended non-naive budget is a deterministic
space-filling global stage followed by local refinement of the same number of
top candidates per coil count. Save every evaluated candidate, not only the
winner.

Primary objective is minimum `U_Gamma`. Reject designs that violate geometry
constraints or the common wafer-near mean-density floor. Define that floor
once from the COMSOL regular anchor, rather than separately from each model, so
models cannot benefit from different feasibility definitions.

For each model report its own optimum and cross-evaluate the union of all
models' top candidates with all four surrogates. This separates optimizer
behavior from evaluator disagreement.

## Optimization track B: expressive independent-coil search

This track tests the additional design freedom that actual structure input can
represent. ABC-SDF and formal SDF may vary each active coil's radial center,
height and size subject to ordering, overlap, chamber and training-support
constraints. Dimension models remain restricted to the regular family and
provide the best regular-layout reference.

Do not pretend that a Dimension model optimized independent coil variables if
those variables are not part of its input. For cross-evaluation only, project
an independent layout back to the five legacy dimensions and label the result
`dimension_projection`, not an exact structure evaluation.

Track B answers a different question from track A and must have separate
tables and figures.

## COMSOL confirmation of optimized designs

Surrogate optimization is not the final result. Deduplicate geometrically
equivalent candidates and run COMSOL for:

- the common baseline regular design;
- the top regular design from each model;
- the top three distinct expressive SDF designs;
- any candidate on which surrogate rankings materially disagree.

Use the existing COMSOL execution code and its convergence checks. Record
initial-condition source, solver status, iteration count and all geometry
parameters. COMSOL outputs are used only after optimization for confirmation;
they must not be fed back into the 500-epoch training dataset in this study.

The optimization conclusion is based on COMSOL-confirmed `U_Gamma`, mean
density and the full radial profile, not surrogate objective alone.

## Acceptance logic

An ABC-SDF accuracy claim requires:

- lower median unknown `ni` error than ABC-Dimension-P;
- lower median unknown Bohm-profile error;
- no material regression in any of the three unknown families;
- direct field plots that agree with the numerical ranking;
- no response-leakage or input-contract failure.

An optimization claim additionally requires COMSOL confirmation. If surrogate
and COMSOL rankings disagree, report the disagreement and do not promote the
surrogate optimum.

