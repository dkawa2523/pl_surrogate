# GEC-CCP NN / Neural-Operator Comparison v1

n=78; interpolation split 54/11/13; 3 independent training seeds.
All requested 1 x 3 artifacts passed the common-contract audit before aggregation.
Representative plotting seeds use only the minimum finite validation objective.

## Three-seed aggregate

| Model | Quality mean ± SD | Plasma R2 mean ± SD | Validation mean ± SD | Plot seed | Runtime mean [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| deeponet_pod | 0.0012 ± 0.0002 | 0.9990 ± 0.0002 | 0.0631 ± 0.0055 | 412 | 435.5 |

## Physical diagnostics (three-seed means)

Negative ratios are evaluated only inside the plasma target. Gradient values are median physical relative L2 errors.

| Model | ne negative [%] | ni negative [%] | Te negative [%] | ne gradient [%] | ni gradient [%] | Te gradient [%] | phi gradient [%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deeponet_pod | 0.0003 | 0.0021 | 0.0190 | 5.58 | 8.35 | 14.80 | 2.77 |

## Seed-level audit

| Model | Seed | Validation | Quality | Plasma R2 | Runtime [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| deeponet_pod | 411 | 0.0628 | 0.0011 | 0.9991 | 434.0 |
| deeponet_pod | 412 | 0.0577 | 0.0010 | 0.9991 | 437.3 |
| deeponet_pod | 413 | 0.0687 | 0.0014 | 0.9988 | 435.3 |

## Interpretation

The quality ranking is a test-set report, not a model-selection rule.
The configured quality score does not penalize sign violations; use the physical-diagnostics table alongside it.
Use the validation-selected manifest for publication plots and keep the PP0 stress lane separate.
