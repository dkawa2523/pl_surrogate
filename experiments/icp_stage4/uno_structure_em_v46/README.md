# ICP UNO structure/EM study v46 (isolated evaluation only)

This directory contains the evaluation-only implementation requested after the
v45 conference-continuity study.  It does not edit or replace the adopted
conference models.  The two frozen reference checkpoints are guarded by SHA256
values in `formal_reference_snapshot.json`.

## Question

Can a UNO learn coil-structure response more accurately from the existing 957
COMSOL cases, without adding structure-specific teacher cases and without using
COMSOL response leakage as an input?

## Candidate ladder

All candidates retain the adopted output variables (`ne`, `ni`, `Te`, `phi`),
fixed seed 1237, fixed 685/136/136 split, density shape/inventory head, optimizer,
and physical-space evaluator.

| ID | Isolated change | General hypothesis |
|---|---|---|
| A | True U-shaped multi-resolution UNO | Coarse and fine response scales should be represented explicitly. |
| AB | Separate process, geometry, and EM lifts with multi-level fusion | Weak structural channels should not be suppressed by scalar operating channels. |
| ABC | Learned spatial local/global mixing | The operator should choose local coil-near versus chamber-wide interaction by position. |
| ABD | Shared per-coil SDF encoder with mean aggregation | Coil identity should be permutation-invariant and reusable across count/layout changes. |
| ABE | Vector EM shape/amplitude representation | Deterministic vacuum-field geometry should be separated from global field magnitude. |
| ABF | Matched structural-delta, Sobolev, and multi-scale supervision | The learned mapping should reproduce the *change* caused by structure, not only absolute fields. |
| ABG | Geometry-latent to EM auxiliary reconstruction | The SDF latent should retain electromagnetically meaningful structure. |
| ABH | Target-specific decoders | Density, temperature, and potential gradients should not destructively interfere at the final head. |
| ALL | All compatible changes above | Test whether individually motivated changes compose cleanly. |

Magnetic fields in ABE/ABG/ALL are deterministic unit-current vacuum fields
derived from geometry.  They are either input carriers or auxiliary targets;
plasma COMSOL responses are never used as inputs.

## Comparison protocol

- Screening training: 15 epochs for every candidate, same seed and split,
  validation every 3 epochs, all layers trainable.
- Initialization: label-aware partial warm start from the adopted v45
  `sdf_vacuum_structure` checkpoint; unmatched tensors remain randomly
  initialized.  The copied lower bound is recorded per candidate.
- Standard test: unchanged benchmark evaluator and physical inverse transforms.
- Unknown structures: frozen G4 pool, 75 cases total (25 unknown spacing,
  25 unseen height+size rank, 25 coupled transforms) plus 25 matched regular
  anchors.
- Main direct metrics: plasma-masked `ni` relative L2 and wafer Bohm-profile
  relative L2.
- Structural response metric: relative L2 of
  `(unknown prediction - matched-anchor prediction)` against
  `(unknown COMSOL - matched-anchor COMSOL)`.
- Statistical check: paired case-bootstrap difference versus the formal SDF
  reference.  A lower median alone is not treated as sufficient evidence.

## Reproduction

```powershell
.venv-torch\Scripts\python.exe experiments\icp_stage4\uno_structure_em_v46\smoke_model_contract.py
.venv-torch\Scripts\python.exe experiments\icp_stage4\uno_structure_em_v46\run_study.py --stage generate --epochs 15
.venv-torch\Scripts\python.exe experiments\icp_stage4\uno_structure_em_v46\run_study.py --stage run --epochs 15
.venv-torch\Scripts\python.exe experiments\icp_stage4\uno_structure_em_v46\evaluate_unknown_structures.py
.venv-torch\Scripts\python.exe experiments\icp_stage4\uno_structure_em_v46\summarize_results.py
.venv-torch\Scripts\python.exe experiments\icp_stage4\uno_structure_em_v46\plot_results.py
```

Runs are written only under `runs/icp_uno_structure_em_v46`.  Evaluation and
figures are written only under
`reports/icp_conference_materials/uno_structure_em_v46`.

## Promotion rule

This study cannot overwrite the formal model.  A candidate is only worth a
separate promotion study if it improves both unknown-structure `ni` and Bohm
medians, avoids a material regression in every unknown family, and has paired
bootstrap evidence that does not indicate a clear regression.  Training cost
and standard-test accuracy remain guardrails rather than being folded into an
opaque composite score.
