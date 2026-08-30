# ICP conference continuity v45: chart map

| Figure | Audience question | Data | Encoding | Decision use |
|---|---|---|---|---|
| 00 unknown-combination field comparison | Does each input representation reproduce the COMSOL ion-density distribution for visibly different unknown geometries? | Three frozen n=4 center-operation G4 cases; coil-layout CSV; COMSOL and model `ni` fields | Columns are spacing, height+size, and coupled-transform structures. Rows are geometry, COMSOL, formal Dimension, explicit Dimension, formal SDF, SDF+vacuum field, and SDF+vacuum+structure. Each prediction is annotated with plasma relative L2. | Primary conference figure; prevents a scalar average from hiding a wrong spatial distribution. |
| 01 Bohm-flux profile comparison | Does field accuracy translate to the wafer-facing quantity of interest? | Same three cases; `ni` and `Te`; 10-layer wafer band and 20 equal-area radial bins | COMSOL is a thick black line; every model is a colored line; x is radius [cm], y is Bohm flux [m^-2 s^-1]. | Confirms whether a visually plausible 2-D field is useful for process prediction. |
| 02 family accuracy | Which representation generalizes across spacing, height+size, and coupled unknown combinations? | 75 frozen G4 non-regular cases | Grouped point/bar summaries of case median with bootstrap 95% CI; field and Bohm errors in separate panels | Main quantitative ranking; no mixing of structural families. |
| 03 incremental improvement | What is gained by magnetic and structural inputs beyond the conference SDF baseline? | Same 75 cases, paired by case | Bars are aggregate medians; labels are per-case win rates for SDF -> SDF+vacuum -> SDF+vacuum+structure | Attributes improvement to each controlled addition without confusing aggregate medians with paired effects. |
| 04 structural-response fidelity | Does the model predict the direction and magnitude of the change from the matching regular layout? | Each unknown case minus same-count/same-operation regular anchor | x is response cosine (1 is correct direction); y is response gain (1 is correct magnitude); family faceting | Separates output memorization from actual sensitivity to geometry. |
| 05 dimension collision audit | Is any SDF advantage an artifact of withholding coil variables from the Dimension model? | Formal 7-variable collisions, explicit 6-coil Dimension, SDF models | Collision count plus grouped median field/Bohm errors on the same 75 cases | Makes the fairness caveat explicit and identifies the correct Dimension control. |

All accuracy panels use physical-space COMSOL targets.  No response-derived
magnetic field is an input: magnetic variants use only deterministic
unit-current vacuum fields calculated from the coil geometry.
