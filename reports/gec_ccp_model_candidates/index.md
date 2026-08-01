# GEC-CCP: implemented model inventory and next comparison candidates

For a concise English table of **Model name**, **Role**, **Current
assessment**, **Surrogate-model application example**, and **Reference**, see the
[GEC-CCP model candidate README](README.md).

## Scope

This inventory compares the canonical model registry with the completed
`gec_ccp_nn_operator_comparison_v1` matrix.  “Not run” means not evaluated in
the current common n78, 54/11/13 interpolation, three-seed protocol.  Some
models have historical leaderboards under other datasets or protocols; those
numbers are not directly comparable and are not used here.

The completed common-protocol models are:

`global_mlp`, `global_resmlp`, `global_densemlp`, `unet`, `fno`, `ffno`,
`u_no`, and `deeponet_pod`.

`deeponet_pod` now means the adopted v2 coefficient branch with GELU after
every hidden layer.  The v1 result remains only as historical evidence.

## Completed single-run screening additions

Seed 412 screening runs have now been completed for `cno`, `deeponet_plasma`,
and `unetpp` under the same n78 54/11/13 data and evaluation contract. These
are one-run screens rather than three-seed results.

| Model | Test quality score | Screening decision |
|---|---:|---|
| `cno` | 0.001837 | Retain as a useful local-operator comparator, but it did not beat FNO |
| `unetpp` | 0.002686 | Existing recipe did not improve on UNet; no immediate promotion |
| `deeponet_plasma` | 0.031767 | Do not adopt without branch/trunk redesign |

Full results: `runs/gec_ccp_nn_operator_comparison_v2_single_run_additions/index.md`.

## First-class models not yet in the common comparison

| Priority | Model | Missing comparison axis | Expected information gain | Recommendation |
|---:|---|---|---|---|
| 1 | `coord_mlp_siren` | implicit coordinate network without POD or spectral convolution | High: isolates continuous-coordinate representation and spatial smoothness | Next orthogonal screening candidate |
| 2 | `unetpp_attn` | nested local skips plus attention versus the current compact UNet | Medium-low after plain UNet++ underperformed UNet | Run only if attention itself is of interest |
| 3 | `geom_deeponet_siren` | descriptor-aware branch plus SIREN trunk | Potentially high for geometry generalization, but current Td/geometry confounding prevents a clean conclusion | Defer until geometry is independently varied or held out |

After the completed screen, `coord_mlp_siren` is the remaining orthogonal
first-class candidate. The CNO result also removes the immediate need to run
the overlapping `cno_operator_unet` hybrid.

## Implemented experimental models not yet in the common comparison

| Model | Role | Current assessment |
|---|---|---|
| `coord_mlp_pod_residual` | POD field plus coordinate residual correction | Scientifically promising for the remaining localized POD error, but keep separate until its experimental training/checkpoint contract is promoted |
| `coord_mlp_fourier` | Fourier-feature coordinate MLP | Useful encoding ablation after `coord_mlp_siren`; not a first addition |
| `unet_operator_v2` | FiLM-conditioned operator-style UNet | Overlaps strongly with UNet/CNO; low immediate information gain |
| `cno_operator_unet` | CNO/UNet hybrid | Run only if the simple `cno` result motivates a multiscale extension |
| `deeponet_plasma_pod` | table-only POD coefficient regression | Weak comparison here because Td already proxies the three geometries |
| `geom_deeponet_pod` | explicitly named geometry-POD lane | Largely overlaps the adopted descriptor-branch `deeponet_pod`; low priority |

## Required comparison contract

Any added model should retain the current contract:

- n78 and split seed 7: train 54 / validation 11 / test 13
- linear physical targets; identity transform and train-only plasma z-score
- `plasma_surrogate_v3` point, boundary, gradient, and multiscale loss
- validation-only checkpoint and recipe selection
- three learning seeds 411/412/413 for a final comparison; one seed is acceptable only for screening
- value error, physical-gradient error, negative ratio, and raw spatial plots
- no comparison of historical leaderboards with this protocol unless they are retrained

## Adoption references

- Canonical future configs: `configs/experimental/gec_ccp_nn_operator_comparison_v2/`
- Adopted POD evidence: `runs/gec_ccp_pod_branch_tuned_v2/review.md`
- Registry: `src/plasma_surrogate/core/model_specs.py`
