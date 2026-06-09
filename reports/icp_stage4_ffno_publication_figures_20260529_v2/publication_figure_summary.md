# ICP_stage4 FFNO Publication Figure Summary

## Runs
- Plasma 4-field FFNO: `runs\icp_stage4_multifield_plasma_ffno_modes24_e80\full\ffno_plasma_modes24_grad005_gpu80_20260528_211257`
- 8-field FFNO for magnetic/current fields: `runs\icp_stage4_multifield_8field_part_sdf_lite_v1_e80\full\ffno_gpu80_20260528_101020`

## Key Metrics
- Plasma 4-field primary: `test_r2_plasma_mean_extrap` = 0.9826
- 8-field primary: `test_r2_plasma_mean_extrap` = 0.9520
- Plasma FFNO continuity grad/lap ratios: 0.865 / 1.051

## Interpretation
- `phi` is replotted with a jet field colormap for visual readability; residual panels use a diverging colormap.
- `Br` and `Bz` are evaluated and displayed on the full domain, because magnetic fields extend outside the plasma region.
- `Jelr` and `Jelz` are displayed with the plasma mask, matching the current target region used in the dataset evaluation.
- The figure set separates scalar R2, distribution-aware metrics, spatial true/pred/error maps, and mid-height profiles.

## Figures
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig01_learning_curve_plasma_ffno_log.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig02_target_r2_plasma_and_em.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig03_distribution_metrics_by_target.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig04_case_quality_distribution.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig05_plasma_fields_worst_case_triplet_phi_jet.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig06_ne_best_median_worst.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig06_ni_best_median_worst.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig06_Te_best_median_worst.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig06_phi_best_median_worst.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig07_phi_best_median_worst_jet_row_scales.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig08_phi_shared_jet_scale_best_median_worst.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig09_plasma_midheight_profiles.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig10_em_current_fields_worst_case_triplet.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig11_Br_best_median_worst.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig11_Bz_best_median_worst.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig11_Jelr_best_median_worst.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig11_Jelz_best_median_worst.png`
- `reports\icp_stage4_ffno_publication_figures_20260529_v2\fig12_em_current_midheight_profiles.png`
