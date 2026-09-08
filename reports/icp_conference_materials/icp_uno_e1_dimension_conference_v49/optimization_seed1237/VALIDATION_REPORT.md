# ICP UNO E1 versus Dimension conference optimization

Overall assessment: **Share with caveats**.

Both frozen seed-1237, 200-epoch models use the same 957-case dataset and the same derivative-free evaluation budget. The optimization minimizes the maximum local deviation of the wafer-near Bohm-flux profile subject to mean density >= 1e17 m^-3.

| Model | Selected coils | Dmax | Mean density [m^-3] | Spacing std [cm] | Height span [cm] | Size span [cm] |
|---|---:|---:|---:|---:|---:|---:|
| Formal Dimension | 4 | 11.546% | 1.2067e+17 | 0.000 | 0.000 | 0.000 |
| E1 causal EM | 4 | 11.178% | 1.3432e+17 | 0.281 | 0.740 | 0.013 |

Predicted E1 Dmax improvement relative to Dimension: **3.19%**.

## Interpretation

- Dimension searches only layouts that its seven scalar inputs can distinguish: regular spacing, one height, and one size.
- E1 searches unequal ordered spacing and bounded height/size trends because those structures are explicitly supplied through union/per-coil SDF and quadrature-order-3 vacuum-field channels.
- Equal objective-evaluation counts make the numerical search budget comparable, although E1 has a higher-dimensional design space.
- This is a surrogate-stage optimization. The selected E1 design must be recalculated in COMSOL before making a physical-superiority or realized-regret claim.

## Validation

- E1 generated input versus training feature check passed: True.
- Equal search budget: True (7040 evaluations/model).
- Selected candidates satisfy density and geometry constraints: True.
- Running-best histories are monotone by construction and independently checked: True.

## Assets

- [Optimized design comparison](10_optimized_design_comparison.png)
- [Coil-count comparison](11_coil_count_comparison.png)
- [Optimization convergence](12_optimization_convergence.png)
- [Dimension 1D animation](horizontal_animation_v51/dimension/dimension_optimization_1d_horizontal.gif)
- [E1 1D animation](horizontal_animation_v51/e1/e1_optimization_1d_horizontal.gif)
- [Dimension 2D animation](horizontal_animation_v51/dimension/dimension_optimization_2d_horizontal.gif)
- [E1 2D animation](horizontal_animation_v51/e1/e1_optimization_2d_horizontal.gif)
- [Dimension combined 1D + 2D animation](horizontal_animation_v51/dimension/dimension_optimization_1d_2d_horizontal.gif)
- [E1 combined 1D + 2D animation](horizontal_animation_v51/e1/e1_optimization_1d_2d_horizontal.gif)
- [Horizontal-animation QA summary](horizontal_animation_v51/horizontal_animation_v51_summary.json)
- [Machine-readable summary](summary.json)
- [Dimension history](dimension_optimization_history.csv)
- [E1 history](e1_optimization_history.csv)

## v50 animation revision validation

- The optimization was not rerun; the two animations reconstruct only the synchronized best-so-far states from the complete 7,040-candidate histories.
- Both models use the same chamber grid, plasma mask, color limits, spatial axes, profile axes, and objective axes for every frame.
- The 2D color encodes local predicted Bohm flux divided by the candidate's wafer-mean Bohm flux; the shared display range is clipped to the joint 1st–99th percentiles and is labelled in the figure.
- Recomputed spatial states agree with the saved optimization metrics within the declared numerical tolerances; `animation_revision_v50_summary.json` records the exact deltas and encoded GIF metadata.
- These remain surrogate predictions pending COMSOL validation.

## v51 horizontal conference-animation restoration

- Review of the prior ICP production animation confirmed a landscape 1 x 3 structure: Bohm evidence, optimization history, and chamber/coil layout. The prior GEC field animation also uses a landscape field/history composition.
- Dimension and E1 are therefore exported as separate model-specific GIFs rather than stacked vertically.
- The 1D-only variant uses `1D Bohm profile | optimization history | geometry` in the established 1 x 3 layout. The 2D-only variant also uses a 1 x 3 layout; the combined 1D + 2D variant uses a 1 x 4 layout. All retain the established 4.2-inch figure height.
- The earlier palette is restored: viridis field, blue feasible candidates, orange best-so-far, pink current candidate, red coils, grey wafer, and yellow wafer-near band. Dimension and E1 profile lines retain blue and orange, respectively.
- Each of the six model-specific GIFs encodes 90 frames at 12 fps with a 1.57-second final hold. Final frames were visually inspected; encoded frame counts, dimensions, fixed limits, panel positions, legends, and annotations were checked from the generated metadata.
- The optimization history uses 300 representative scatter points for conference readability, while the orange best-so-far line is recomputed from all 7,040 candidates.
- Machine-readable QA is saved in `horizontal_animation_v51/horizontal_animation_v51_summary.json`; validation passed.
