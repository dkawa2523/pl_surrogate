# GEC-CCP NN / Neural-Operator Comparison v1

n=78; interpolation split 54/11/13; three independent training seeds.
All 8 x 3 artifacts passed the common-contract audit before aggregation.
Representative plotting seeds use only the minimum finite validation objective.

## Three-seed aggregate

| Model | Quality mean ± SD | Plasma R2 mean ± SD | Validation mean ± SD | Plot seed | Runtime mean [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| u_no | 0.0002 ± 0.0000 | 0.9998 ± 0.0000 | 0.0238 ± 0.0002 | 413 | 1051.5 |
| ffno | 0.0005 ± 0.0000 | 0.9994 ± 0.0000 | 0.0379 ± 0.0005 | 412 | 906.1 |
| fno | 0.0012 ± 0.0001 | 0.9987 ± 0.0001 | 0.0593 ± 0.0021 | 412 | 815.9 |
| unet | 0.0013 ± 0.0000 | 0.9985 ± 0.0001 | 0.0694 ± 0.0019 | 412 | 651.0 |
| global_resmlp | 0.0022 ± 0.0006 | 0.9982 ± 0.0005 | 0.0761 ± 0.0117 | 413 | 351.7 |
| deeponet_pod | 0.0030 ± 0.0002 | 0.9976 ± 0.0001 | 0.0978 ± 0.0031 | 411 | 419.2 |
| global_densemlp | 0.0036 ± 0.0011 | 0.9969 ± 0.0012 | 0.0988 ± 0.0149 | 413 | 343.6 |
| global_mlp | 0.0587 ± 0.0118 | 0.9066 ± 0.0347 | 0.4371 ± 0.0314 | 413 | 532.7 |

## Physical diagnostics (three-seed means)

Negative ratios are evaluated only inside the plasma target. Gradient values are median physical relative L2 errors.

| Model | ne negative [%] | ni negative [%] | Te negative [%] | ne gradient [%] | ni gradient [%] | Te gradient [%] | phi gradient [%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| u_no | 3.0905 | 0.1359 | 0.0000 | 16.41 | 20.12 | 14.72 | 5.72 |
| ffno | 4.0106 | 0.2823 | 0.0000 | 21.09 | 24.26 | 19.20 | 7.11 |
| fno | 6.1764 | 1.1005 | 0.0000 | 24.69 | 29.07 | 24.77 | 11.55 |
| unet | 6.6269 | 2.5463 | 0.0000 | 32.82 | 36.30 | 21.55 | 11.02 |
| global_resmlp | 0.0272 | 0.0013 | 0.0433 | 7.04 | 9.76 | 18.49 | 4.74 |
| deeponet_pod | 0.1517 | 0.0020 | 0.0558 | 9.40 | 10.32 | 18.41 | 4.33 |
| global_densemlp | 0.0174 | 0.0015 | 0.0433 | 7.23 | 9.79 | 18.03 | 6.10 |
| global_mlp | 3.6035 | 0.5031 | 0.2192 | 38.29 | 39.46 | 73.03 | 31.18 |

## Seed-level audit

| Model | Seed | Validation | Quality | Plasma R2 | Runtime [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| deeponet_pod | 411 | 0.0944 | 0.0028 | 0.9977 | 435.5 |
| deeponet_pod | 412 | 0.1006 | 0.0031 | 0.9975 | 387.1 |
| deeponet_pod | 413 | 0.0984 | 0.0031 | 0.9976 | 434.9 |
| ffno | 411 | 0.0382 | 0.0005 | 0.9994 | 907.0 |
| ffno | 412 | 0.0373 | 0.0005 | 0.9995 | 906.6 |
| ffno | 413 | 0.0383 | 0.0005 | 0.9994 | 904.8 |
| fno | 411 | 0.0615 | 0.0012 | 0.9986 | 819.1 |
| fno | 412 | 0.0573 | 0.0011 | 0.9988 | 814.5 |
| fno | 413 | 0.0592 | 0.0012 | 0.9987 | 814.1 |
| global_densemlp | 411 | 0.1073 | 0.0041 | 0.9963 | 340.2 |
| global_densemlp | 412 | 0.1075 | 0.0043 | 0.9961 | 341.1 |
| global_densemlp | 413 | 0.0816 | 0.0023 | 0.9983 | 349.6 |
| global_mlp | 411 | 0.4713 | 0.0702 | 0.8676 | 535.7 |
| global_mlp | 412 | 0.4306 | 0.0467 | 0.9342 | 532.4 |
| global_mlp | 413 | 0.4094 | 0.0592 | 0.9181 | 530.1 |
| global_resmlp | 411 | 0.0773 | 0.0025 | 0.9980 | 347.0 |
| global_resmlp | 412 | 0.0872 | 0.0025 | 0.9978 | 350.4 |
| global_resmlp | 413 | 0.0638 | 0.0015 | 0.9988 | 357.6 |
| u_no | 411 | 0.0238 | 0.0002 | 0.9998 | 1052.9 |
| u_no | 412 | 0.0240 | 0.0002 | 0.9998 | 1050.3 |
| u_no | 413 | 0.0236 | 0.0002 | 0.9998 | 1051.4 |
| unet | 411 | 0.0714 | 0.0014 | 0.9984 | 652.4 |
| unet | 412 | 0.0676 | 0.0013 | 0.9985 | 653.8 |
| unet | 413 | 0.0692 | 0.0013 | 0.9984 | 646.8 |

## Interpretation

The quality ranking is a test-set report, not a model-selection rule.
The configured quality score does not penalize sign violations; use the physical-diagnostics table alongside it.
Use the validation-selected manifest for publication plots and keep the PP0 stress lane separate.
