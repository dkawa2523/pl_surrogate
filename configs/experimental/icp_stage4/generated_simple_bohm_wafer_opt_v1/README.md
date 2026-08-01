# ICP simple Bohm wafer optimization v1

This setting demonstrates a surrogate-model optimization with one physically
interpretable objective and a small joint search space.

## Objective

Minimize the axisymmetric area-weighted coefficient of variation of the Bohm
flux proxy in the two plasma cells nearest the wafer:

`Gamma_B = ni * sqrt(Te)`

`objective = weighted_std(Gamma_B, r) / weighted_mean(Gamma_B, r)`

The omitted constant `sqrt(e / m_i)` cancels from both the CV and the reference
ratio.  A single feasibility constraint keeps the area-weighted mean flux above
80% of `case_g002_op01`, preventing a trivially uniform low-flux solution.

## Variables

- Process: `pp`, `pp0`, restricted to the central 90% of the dataset range.
- Structure: radial shifts `part.coil_01.tx` through `part.coil_06.tx`.

Each structural trial is passed to the parametric-parts geometry provider.  It
rebuilds the coil masks and the trained `part_source_v1` feature channels, so
the optimizer never edits a feature image directly.

## Optimizer and outputs

- Optuna TPE, 192 trials, seed 411.
- One objective term: `bohm_flux_area_cv`.
- One feasibility constraint: `bohm_flux_area_mean >= 6.0249874e16`.
- Save full fields only for the best five trials.
- Run the same problem independently for U-Net and UNO; do not combine their
  predictions during optimization.
