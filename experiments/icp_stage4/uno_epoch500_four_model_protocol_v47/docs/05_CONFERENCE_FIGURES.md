# Conference figure specification

## Figure policy

Every figure must state one question, use physical units, identify COMSOL as
the reference, and write both SVG and PNG. Use one physical color scale within
each field-comparison figure. Do not use latent-space distance or a composite
score as the primary audience-facing evidence.

Write all files only under:

```text
reports/icp_conference_materials/icp_uno_epoch500_four_model_v47/figures/
```

Maintain `CHART_MAP.md` with figure ID, claim, source tables, cases, axes,
units and caveats.

## Required accuracy figures

### Figure 00: direct unknown-structure field comparison

**Message:** which model reproduces the COMSOL ion-density response when coil
spacing, height and size are combined in ways not exactly represented by the
regular dimension vector?

- Columns: the three frozen representative cases
  `v43_te_n4_variant_1__center`, `variant_2`, and `variant_3`.
- Rows: coil geometry, COMSOL, formal Dimension, formal SDF, ABC-Dimension-P,
  ABC-SDF.
- Field: `ni` in `1e17 m^-3` on one shared scale.
- Annotation: plasma relative L2 percentage on each prediction.
- Axes: radius `r [cm]`, height `z [cm]`.

### Figure 01: wafer Bohm-flux profiles

**Message:** does the two-dimensional agreement translate to the wafer-facing
quantity used for process decisions?

- One panel per representative case.
- x-axis: wafer radius `r [cm]`.
- y-axis: `Gamma_i [m^-2 s^-1]`.
- COMSOL: thick black line.
- Four models: distinct, colorblind-safe lines.
- Legend annotation: profile relative L2 percentage.

### Figure 02: error distribution over all 75 unknown structures

**Message:** is improvement general, rather than selected from three favorable
examples?

- Two panels: `ni_rel_l2_pct` and `bohm_profile_rel_l2_pct`.
- x-axis: model.
- y-axis: physical relative L2 percentage, lower is better.
- Show each case as a point plus median and 10th-90th percentile.
- Keep model ordering identical in every aggregate figure.

### Figure 03: accuracy by structural family

**Message:** which type of structural change is learned or missed?

- Rows: four models.
- Columns: unknown spacing, height+size rank, coupled change.
- Two heatmaps: median `ni` error and median Bohm-profile error.
- Every cell contains its numerical percentage.

### Figure 04: recovered structural response

**Message:** does the model predict the change caused by structure, not merely
the average plasma distribution?

- x-axis: COMSOL norm of `(unknown - matched anchor)`.
- y-axis: predicted norm of the same change.
- Normalize both by the matched-anchor norm and show percent.
- Add the `y=x` line.
- Use separate panels for `ni` and the Bohm profile.
- A near-zero predicted change appears directly as a point near the x-axis.

### Figure 05: 500-epoch learning and selection audit

**Message:** did longer training improve validation behavior or only fit the
training set?

- x-axis: epoch 1-500.
- y-axis: training loss and validation selection objective.
- One panel per model; mark selected epoch.
- A second compact panel reports wall time and effective updates.
- Do not compare raw losses across models without confirming identical scaling.

## Required optimization figures

### Figure 06: matched regular-layout optimization

**Message:** under the same representable design freedom, do model optima agree
and survive COMSOL confirmation?

- x-axis: coil count 2-6.
- y-axis: best feasible `U_Gamma [%]`, lower is better.
- Show surrogate result and COMSOL-confirmed result with connected markers.
- A gap between the two is optimization prediction error, not improvement.

### Figure 07: optimized geometry and COMSOL Bohm profile

**Message:** what physical structure was selected and what does COMSOL predict?

- Left: baseline and optimized coil rectangles on physical `r-z` axes.
- Center: COMSOL baseline and optimized `ni` fields on a shared scale.
- Right: COMSOL radial Bohm profiles normalized by their mean.
- Annotate `U_Gamma`, mean density and geometry feasibility.

### Figure 08: regular versus expressive design freedom

**Message:** does actual-structure SDF provide a useful optimum outside the
regular dimension manifold?

- x-axis: COMSOL-confirmed mean Bohm flux or mean density.
- y-axis: COMSOL-confirmed `U_Gamma [%]`.
- Points: regular Dimension-family candidates and independent-coil SDF
  candidates.
- Draw the nondominated frontier; do not connect unrelated trials as a time
  series.

## Minimum conference set

If slide space is limited, use Figures 00, 01, 02 and 07. Figure 00 establishes
the field result, Figure 01 gives the process-facing profile, Figure 02 prevents
case selection, and Figure 07 connects surrogate inference to COMSOL-confirmed
optimization.

## Figure QA

- Numerical annotations must match the source CSV after rounding.
- The same COMSOL truth array must be used for every model in a case.
- Masks, radial bins and units must be identical across models.
- SVG text must remain selectable and no axes may be rasterized.
- Large field maps may be rasterized inside SVG, but labels and annotations
  remain vector.
- Generate a `figure_manifest.json` containing source hashes and script hash.

