# Recovery, audit, and acceptance

## Formal-asset preservation

Before and after the study, hash at least these frozen formal checkpoints:

```text
runs/icp_conference_continuity_v45/final/conference_sdf/seed_1237/models/u_no/eval_protocol/interp/checkpoints/weights.npz
runs/icp_conference_continuity_v45/final/conference_dimension/seed_1237/models/u_no/eval_protocol/interp/checkpoints/weights.npz
```

Expected SHA256 values at protocol freeze are:

```text
Formal SDF       DB8568A958D7D1A2208C036F27B9B22EB93615008012D0C72E4FF642946FAAC1
Formal Dimension A87AADB88C3FE383BC2B2BB27F306DCC8BBA5735E5C4F1BA2796559C856B04FF
```

Any change is a hard failure. The v47 study has no promotion authority and may
not replace either formal model.

## Sequential state machine

Maintain a top-level `study_status.json` with one state per model:

```text
pending -> smoke_passed -> training -> trained_500 -> evaluated -> accepted
```

On failure use `failed_recoverable` or `failed_contract`. Resume only a
recoverable run whose config, source and dataset hashes match. A contract
failure requires a new isolated run directory; do not patch an existing result
in place.

## Recovery test before long execution

The smoke phase must deliberately stop after a recovery checkpoint, restart in
a new process and prove that:

- epoch numbering continues;
- learning rate continues at the correct cosine position;
- optimizer moments are restored;
- RNG streams are restored;
- the resumed two-step weights match an uninterrupted reference within the
  declared floating-point tolerance.

Without this test, multi-day training must not start.

## Artifact audit

For every model save:

- generated and resolved configuration;
- model implementation hash;
- initial-weight hash and initialization mapping report;
- dataset index and split hashes;
- preprocessing feature names, scaling statistics and pack hashes;
- full epoch metrics;
- final and selected checkpoint hashes;
- standard and unknown-structure case-level metrics;
- elapsed time, GPU name, torch and CUDA versions;
- stdout/stderr log;
- completion sentinel.

For optimization save every candidate and constraint result, not only the best
candidate. For COMSOL confirmation save input geometry, initialization source,
solver status and raw fields.

## Scientific acceptance gates

All four runs must first pass:

1. Exactly 500 completed epochs.
2. Finite losses and predictions.
3. Fixed split and target contract.
4. No response-derived input channel.
5. Formal checkpoint preservation.
6. ABC parameter-count equality.
7. Frozen unknown population count of 75 plus 25 anchors.

Only then interpret accuracy. The default promotion threshold is intentionally
not a composite score. ABC-SDF must improve both unknown `ni` and Bohm-profile
median error against ABC-Dimension-P, avoid a material family regression and
show no clear paired-bootstrap regression. Standard-test accuracy and cost are
guardrails.

Optimization claims require COMSOL confirmation. If COMSOL reverses the
surrogate ranking, the correct conclusion is that the optimization surrogate
is not yet sufficiently reliable in that region.

## Known limitations

- One training seed does not estimate training-seed variance. Treat the
  500-epoch seed-1237 comparison as a high-budget confirmation, not a full
  uncertainty study.
- Warm-start semantics differ where input meanings differ. The copied tensor
  inventory must remain visible.
- ABC-Dimension-P's projected vacuum field is deliberately not the actual field
  of an irregular structure.
- The training dataset is fixed; this protocol does not add optimized COMSOL
  cases back into training.
- Longer training may overfit. Report the selected checkpoint and the full
  500-epoch trajectory.

## Final hand-off contents

The receiving environment should return:

```text
reports/icp_conference_materials/icp_uno_epoch500_four_model_v47/
  TECHNICAL_REPORT.md
  model_summary.csv
  case_metrics_unknown75.csv
  structural_response_metrics_unknown75.csv
  optimization/
  figures/
  CHART_MAP.md
  audit/
  decision_summary.json
```

The report must state both positive and negative results. It must not claim SDF
superiority if the numerical and direct COMSOL comparisons do not support it.

