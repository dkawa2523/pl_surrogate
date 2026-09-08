# Chart contract: ICP UNO E1 versus Dimension conference package

## Scope

- Frozen models: seed 1237, 200 epochs, fixed 957-case split.
- Validation: the three retained simple COMSOL interventions; no validation case is used for training or checkpoint selection.
- Optimization: surrogate-only search using the established wafer-near Bohm-flux objective and density constraint.
- Formal training/model source is not modified; all new code and outputs are isolated under v49 paths.

## Figure 00 — direct COMSOL comparison

- Question: Across simple spacing, size, and height changes, which frozen representation is closer to COMSOL?
- Takeaway tested: E1 has lower absolute Bohm-profile error than Dimension, while Dimension cannot distinguish per-coil changes that collide in its scalar encoding.
- Family/variant: three-column small-multiple line comparison plus zero-based paired bars.
- Data: 3 interventions x 20 radial bins x COMSOL/Dimension/E1; 6 error bars.
- Renderer: static Matplotlib; PNG and SVG.
- Palette: neutral COMSOL, blue Dimension, orange E1; solid/dashed lines and distinct markers.
- QA: common profile scale, zero-based error bars, explicit units, visible geometry schematic.

## Figure 10 — optimized design comparison

- Question: What best density-feasible design does each representation propose under the same objective and evaluation budget?
- Takeaway tested: E1 can search unequal spacing, bounded height trend, and bounded size trend; Dimension is restricted to its regular-layout encoding.
- Family/variant: paired profile lines, paired layout schematics, and a two-category objective bar.
- Data: one selected candidate per model plus the established 200 uniform-radius radial bins over 0--20 cm.
- Renderer: static Matplotlib; PNG and SVG.
- Palette: blue Dimension, orange E1; non-color line/marker distinction.
- QA: objective bars start at zero; all layout axes share physical scales; surrogate-only caveat is visible.

## Figure 11 — coil-count comparison

- Question: Is the selected result dependent on one coil count?
- Takeaway tested: best feasible maximum deviation and mean density across coil counts 2–6.
- Family/variant: two aligned line charts with a density-threshold reference.
- Data: 5 coil counts x 2 models.
- Renderer: static Matplotlib; PNG and SVG.
- QA: identical coil-count ticks, explicit percent and density units, no dual axis.

## Figure 12 — optimization convergence

- Question: How quickly does each search improve at the same cumulative evaluation budget?
- Takeaway tested: running-best feasible objective, not raw noisy candidate scores.
- Family/variant: highlighted two-series line chart.
- Data: every evaluated candidate in chronological order for both models.
- Renderer: static Matplotlib; PNG and SVG.
- QA: same x/y scale, monotone running-best construction checked independently.

## Animation — synchronized optimization progress

- Question: How do the best-so-far profile and geometry evolve during the two searches?
- Takeaway tested: Dimension updates only its regular scalar layout; E1 can update individual spacing, height, and size within bounded support.
- Variant: synchronized paired-model GIF with layout, normalized profile, and running-best objective.
- Data: representative quantiles of each complete evaluation history plus final selected candidates.
- Renderer: Matplotlib + Pillow GIF.
- QA: fixed axes across frames, synchronized evaluation fraction, final frame held, explicit surrogate-only label.

## v50 requested layout revision

### Analytical question and takeaway

- Revised 00: compare the three retained simple interventions—right-shifted middle coil, size gradient, and mild height gradient—without shrinking the full conference canvas.
- 2D animation: show how the best-so-far surrogate candidate changes the chamber-wide Bohm-flux field, while retaining a fixed comparison frame for Dimension and E1.
- Combined animation: show geometry, normalized 2D Bohm-flux field, wafer-near 1D profile, and optimization convergence together without any panel moving between frames.

### Chart family and renderer

- Revised 00: static small multiples with geometry, three-series radial line comparison, and two-model error bars; Matplotlib PNG and SVG.
- 2D animation: fixed-grid spatial heatmap small multiples plus geometry and convergence; Matplotlib/Pillow GIF and final-frame PNG.
- Combined animation: fixed four-row small multiples (geometry, heatmap, radial line, convergence); Matplotlib/Pillow GIF and final-frame PNG.

### Data sufficiency and metric definition

- Revised 00 uses the existing COMSOL-backed profile bins and relative-L2 metrics for three explicitly retained interventions.
- Animations use all 7,040 saved candidates per model and the synchronized best-so-far indices from the validated v49 search history.
- Spatial color encodes predicted local Bohm flux divided by that candidate's area-weighted wafer mean. A single robust display range is computed jointly across both models and all displayed best-so-far states; clipped display limits are stated on the figure.
- The 1D profile and Dmax retain the formal v49 definition: 200 uniform-radius bins over 0–20 cm and a ten-layer wafer-near band.

### Palette and non-color distinction

- Hard two-root cap: blue for Formal Dimension and orange for E1; plasma-field magnitude uses one sequential blue root with a shared colorbar.
- Model identity is also carried by column title, line style, and marker shape.

### Fixed-layout contract

- The figure canvas, GridSpec, axis bounds, color limits, text anchors, colorbar position, and legend position are created once and never recomputed per frame.
- Artists are updated in place; no animated axis is cleared and neither `tight_layout` nor `bbox_inches='tight'` is used for GIF frames.
- Revised 00 keeps the original 16.8 x 8.7 inch canvas while changing from four to three equal-width columns.

### Output and QA surface

- Outputs remain isolated below `reports/icp_conference_materials/icp_uno_e1_dimension_conference_v49`.
- Validate source row counts, running-best monotonicity, common color limits, PNG dimensions, GIF frame count/duration, and visual readability of the first/middle/final frames.

## v51 restoration of the established GEC/ICP animation language

- Reference layout: the prior ICP production animation uses one landscape 1 x 3 GIF per representation—left Bohm evidence, center optimization history, right chamber/wafer/coil layout. The prior GEC animation uses a landscape spatial-field/history pair.
- Revised 2D animation: one 13.4 x 4.2 inch landscape GIF per model with `2D field | optimization history | geometry`.
- Revised combined animation: one 17.4 x 4.2 inch landscape GIF per model with `1D profile | 2D field | optimization history | geometry`.
- Models are not stacked. Dimension and E1 receive separate GIFs with identical axes and synchronized 90-state sampling.
- Restored palette: Dimension `#0072B2`, E1 `#D55E00`, spatial `viridis`, feasible candidates `#0072B2`, best-so-far `#D55E00`, current candidate `#CC79A7`, coils `#D62728`, wafer `#737373`, and wafer band `#FFB000`.
- Optimization scatter uses 300 representative candidates, matching the density of the earlier ICP conference animation, while the best-so-far line is still computed from all 7,040 candidates.
- Axes, color limits, artist positions, legend anchors, and canvas size are fixed across frames. Final states are held without changing the displayed frame counter.

## v51 restoration of the established GEC/ICP animation grammar

### Evidence from previous conference assets

- The prior ICP production animation uses one landscape GIF per representation with a single 1 x 3 row: Bohm profile, optimization history, and chamber/coil layout (`13.4 x 4.2` inches; encoded `1419 x 453` px).
- The prior GEC production animation uses a landscape 1 x 2 row: viridis spatial field and optimization history (`1334 x 529` px).
- Established colors are Dimension `#0072B2`, SDF/E1 `#D55E00`, feasible/completed points `#0072B2`, future/infeasible points grey, best-so-far `#D55E00`, current candidate `#CC79A7`, coils `#D62728`, wafer `#737373`, wafer band `#FFB000`, plasma background `Blues`, and spatial physical fields `viridis`.

### Revised analytical question and layout

- 1D restoration: for each model separately, show mean-zero wafer-near Bohm profile, optimization history, and chamber/coil layout in the established landscape 1 x 3 row.
- 2D replacement: for each model separately, show normalized 2D Bohm field, optimization history, and chamber/coil layout in a landscape 1 x 3 row.
- 1D + 2D: for each model separately, show mean-zero 1D wafer profile, normalized 2D Bohm field, optimization history, and chamber/coil layout in a landscape 1 x 4 row.
- Dimension and E1 are delivered as separate synchronized GIFs, matching the prior ICP convention and avoiding any vertical model stacking.

### Fixed-layout and comparison contract

- All axes, color limits, profile limits, logarithmic objective limits, legend anchors, and annotation boxes are created once and updated in place.
- Dimension and E1 use identical spatial color limits, 1D profile limits, objective limits, chamber bounds, frame progress, and final hold duration.
- The 2D field remains local predicted Bohm flux divided by the candidate's wafer-mean Bohm flux; the common viridis scale is clipped to the joint 1st–99th percentiles and explicitly labelled.
- Outputs are surrogate-only and must not be described as COMSOL-validated optimized designs.

### v51 1D-only animation contract

- Analytical question: how does the best-so-far wafer-near Bohm-flux profile become more uniform while the coil design changes?
- Layout: one 13.4 x 4.2 inch landscape GIF per model with `1D Bohm profile | optimization history | chamber/wafer/coil layout`.
- Metric: the left axis shows percentage deviation from each candidate's area-weighted wafer mean over the established 200 radial bins and ten-layer wafer-near band; the center axis shows the saved Dmax objective for all 7,040 evaluations.
- Comparison: Dimension and E1 use identical axis limits, frame indices, final hold, geometry bounds, and non-color encodings; no panel is reflowed between frames.
- Claim boundary: the animation visualizes frozen-surrogate optimization behavior only; the optimized final candidate remains pending COMSOL confirmation.

## Current-trial coherence revision

- Analytical question: for each displayed optimization trial, what geometry was evaluated and what 1D/2D Bohm response did that exact geometry predict?
- Frame grain: 90 uniformly spaced evaluated trials are sampled from each complete 7,040-trial history; the final hold repeats trial 7,040 without creating a new analytical state.
- Index invariant: profile, spatial field, coil geometry, trial Dmax annotation, and pink current-trial marker must all use the same saved history index. Only the orange convergence line represents the running best.
- Comparison: Dimension and E1 retain identical sampled progress fractions, axis limits, palette, canvas dimensions, and final hold.
- Visible wording: panels and annotations say `current trial` or `trial Dmax`; they must not call the displayed geometry or field `best`.
- QA blocker: publication is blocked if any frame's profile, field, geometry, and current marker do not resolve to one identical history index.

## Formal Dimension–SDF truth/prediction/error spatial figures

- Analytical question: on the same three frozen unknown-structure cases, where do the latest Formal Dimension and Formal SDF ion-density predictions agree with or deviate from COMSOL?
- Models: seed 1237, 200 training epochs, identical 957-case dataset and fixed split; validation-selected epochs are 36 for Formal Dimension and 152 for Formal SDF.
- Data grain: three representative unknown-structure cases, each on the same 440 x 600 axisymmetric grid. Population claims remain based on all 75 unknown-structure cases and are not inferred from the three displayed cases alone.
- Figures: one 3 x 3 `COMSOL truth | model prediction | signed error` figure per model, plus one 3 x 5 direct-comparison figure with a single COMSOL column and prediction/error pairs for both models.
- Field and units: physical ion density `ni` in `10^17 m^-3`; signed error is `prediction - COMSOL` in the same units.
- Scale policy: every truth and prediction panel uses one joint physical scale; every error panel uses one joint symmetric signed-error scale across both models and all three cases. Both limits use the joint 99.5th percentile and the clipping policy is printed in the subtitle.
- Palette: `viridis` for non-negative physical density and `RdBu_r` centered at zero for signed error; model identity is carried by panel titles and row labels, not by changing physical-field colors.
- Outputs: PNG, SVG, and PDF under `formal_dimension_sdf_spatial_v52`; source metrics and a machine-readable validation summary are colocated.
- QA blocker: masks, coordinates, and COMSOL truth must match between model records; recomputed plasma-domain relative L2 must match the saved case metrics within numerical tolerance.
