# ICP conference-continuity retraining v45

This isolated study retrains the adopted ICP UNO specification on the frozen
957-case legacy + v43 dataset without modifying any completed conference run.

The controlled model ladder is:

1. `conference_dimension`: adopted seven-scalar Dimension input.
2. `explicit_dimension`: the same UNO with all six coil rectangles represented
   by active flag, radial/axial center, width and height.
3. `conference_sdf`: adopted union-SDF + count-neutral source input.
4. `sdf_vacuum`: model 3 plus deterministic unit-current vacuum
   `Aphi/Br/Bz/Bmag` fields.
5. `sdf_vacuum_structure`: model 4 plus order-invariant second-nearest,
   coil-competition and solid-proximity structure maps.

All variants use the same frozen case membership, target fields, UNO trunk,
loss, optimizer, seed, epoch budget and checkpoint-selection rule.  The
conference seed-1237 checkpoints initialize the common channels and the whole
network remains trainable.  New channels start with zero lifting weights.

The primary test is the frozen v43 G4 set.  Its non-regular families are
`unknown_gap_topology`, `unseen_rank_height_size` and `coupled_transform`,
which directly exercise unknown combinations of coil spacing, height and size.

## Reproduction

The commands below use the repository's CUDA-enabled virtual environment.

```powershell
.\.venv-torch\Scripts\python.exe experiments/icp_stage4/conference_continuity_v45/prepare_dataset_view.py
.\.venv-torch\Scripts\python.exe experiments/icp_stage4/conference_continuity_v45/audit_comparison_contract.py
.\.venv-torch\Scripts\python.exe experiments/icp_stage4/conference_continuity_v45/run_retraining.py --stage all --epochs 50
.\.venv-torch\Scripts\python.exe experiments/icp_stage4/conference_continuity_v45/evaluate_unknown_structures.py
.\.venv-torch\Scripts\python.exe experiments/icp_stage4/conference_continuity_v45/plot_unknown_structure_results.py
.\.venv-torch\Scripts\python.exe experiments/icp_stage4/conference_continuity_v45/build_report_artifact.py
```

The first unbounded structural-distance smoke test is preserved under
`runs/icp_conference_continuity_v45/smoke`.  The bounded, order-invariant
replacement is preserved under `smoke_bounded_v2`; it reduced the two-epoch
primary score from 0.57348 to 0.07886 without changing the SDF/vacuum trunk.

Evaluation writes case-level data before summaries.  `case_metrics_unknown75.csv`
is the absolute COMSOL comparison; `structural_response_metrics_unknown75.csv`
is the unknown-minus-matched-anchor comparison.  `CHART_MAP.md` records the
question, data and intended use of every publication figure.
