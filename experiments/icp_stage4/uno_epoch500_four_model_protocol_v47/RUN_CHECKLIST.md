# Operator checklist

## Isolation

- [ ] New outputs resolve only below `runs/icp_uno_epoch500_four_model_v47`.
- [ ] New reports resolve only below
      `reports/icp_conference_materials/icp_uno_epoch500_four_model_v47`.
- [ ] v45/v46 formal hashes match before execution.
- [ ] No existing run directory is a writable target.

## Environment and data

- [ ] Portable branch checked out and frozen source files match `SOURCE_INVENTORY.json`.
- [ ] `git lfs pull` and `git lfs fsck` completed for data, weights and references.
- [ ] `SOURCE_INVENTORY.json` hashes pass.
- [ ] Python, torch, CUDA, GPU and dependency versions recorded.
- [ ] CUDA is available and the expected GPU is selected.
- [ ] Dataset contains 957 unique cases.
- [ ] Split contains 685 train, 136 validation and 136 test cases.
- [ ] At least 80 GiB initial free disk and 20 GiB free-space guard configured.

## Model contract

- [ ] Formal SDF and Dimension configs differ from adopted versions only in
      epoch budget, isolated paths, recovery and metadata.
- [ ] ABC-SDF has the declared 14 spatial channels.
- [ ] ABC-Dimension-P uses five broadcast dimensions and dimension-derived,
      not actual-structure, vacuum fields.
- [ ] ABC models have equal trainable parameter counts.
- [ ] Initialization mapping reports exist and semantic mismatches are not copied.
- [ ] No COMSOL response field appears in input channels.

## Smoke and recovery

- [ ] Model instantiation and checkpoint loading pass for all four models.
- [ ] One-epoch smoke passes in isolated directories.
- [ ] Stop/resume equivalence test passes.
- [ ] Recovery checkpoint includes optimizer, scheduler, epoch and RNG states.

## Training

- [ ] Models run sequentially in the fixed order.
- [ ] Early stopping is disabled.
- [ ] Each completion sentinel states 500 completed epochs.
- [ ] Selected checkpoint, final checkpoint and metric histories are hashed.

## Evaluation

- [ ] Standard 136-case evaluation complete for all four models.
- [ ] Frozen evaluation contains exactly 75 unknowns and 25 anchors.
- [ ] Physical inverse transforms and one common plasma mask are used.
- [ ] `ni`, Bohm profile, uniformity and structural-response metrics saved per case.
- [ ] Aggregate, family and paired-bootstrap summaries saved.

## Optimization and conference output

- [ ] Matched regular-layout optimization uses one common budget and seed 4099.
- [ ] Expressive independent-coil optimization is reported separately.
- [ ] Top distinct candidates are confirmed by COMSOL.
- [ ] Figures 00-08 are generated in PNG and SVG.
- [ ] Figure annotations reconcile with source CSV files.
- [ ] `CHART_MAP.md` and `figure_manifest.json` exist.
- [ ] Final formal hashes still match.
