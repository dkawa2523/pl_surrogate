# GEC-CCP model candidates and references

This page is the concise English reference for GEC-CCP model candidates.
`Current assessment` describes the model's role in the common GEC-CCP
comparison; it is not a claim that the cited paper was reproduced exactly.

## Primary candidates

| Model name | Role | Current assessment | Surrogate-model application example | Reference |
|---|---|---|---|---|
| `coord_mlp_siren` | Implicit coordinate network with sinusoidal activations | The strongest remaining orthogonal first-class screening candidate. It isolates continuous-coordinate representation and spatial smoothness without POD or spectral convolution. | Continuous reconstruction of high-gradient plasma fields at queried spatial coordinates; more generally, compact representation of signals and PDE solutions. | Sitzmann et al., [*Implicit Neural Representations with Periodic Activation Functions* (2020)](https://arxiv.org/abs/2006.09661) |
| `unetpp_attn` | Nested U-Net skip paths with attention gating | Medium-to-low priority because plain U-Net++ underperformed U-Net in the existing screen. Run only when the incremental value of attention is itself a research question. | Full-field plasma prediction when localized sheaths or boundary layers should receive selective multiscale emphasis; the cited architectures were originally demonstrated for image segmentation. | Zhou et al., [*UNet++* (2018)](https://arxiv.org/abs/1807.10165); Oktay et al., [*Attention U-Net* (2018)](https://arxiv.org/abs/1804.03999) |
| `geom_deeponet_siren` | Geometry-aware branch with a SIREN coordinate trunk | Potentially informative for geometry generalization, but the present correlation between gas temperature and geometry prevents a clean attribution. Defer until geometry is independently varied or held out. | Geometry-conditioned field surrogate evaluated at arbitrary coordinates, suitable in principle for inverse design across independently varied reactor geometries. | Lu et al., [*Learning nonlinear operators via DeepONet* (2021)](https://doi.org/10.1038/s42256-021-00302-5); Sitzmann et al., [*SIREN* (2020)](https://arxiv.org/abs/2006.09661) |

## Implemented experimental candidates

| Model name | Role | Current assessment | Surrogate-model application example | Reference |
|---|---|---|---|---|
| `coord_mlp_pod_residual` | POD field reconstruction plus coordinate-wise residual correction | Scientifically promising for localized residual errors left by POD, but keep it outside the common comparison until its experimental training and checkpoint contract is promoted. | Fast reduced-order sweeps in which POD captures the global plasma field and a coordinate residual restores localized sheath or edge errors. | Lu et al., [*A comprehensive and fair comparison of two neural operators* (2022)](https://arxiv.org/abs/2111.05512) |
| `coord_mlp_fourier` | Coordinate MLP using Fourier-feature positional encoding | Useful as an encoding ablation after `coord_mlp_siren`; it is not the first additional model to prioritize. | Coordinate-query surrogate for spatial fields containing fine radial or axial variation that a plain MLP tends to smooth. | Tancik et al., [*Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains* (2020)](https://arxiv.org/abs/2006.10739) |
| `unet_operator_v2` | FiLM-conditioned operator-style U-Net | Strongly overlaps the existing U-Net and CNO comparisons, so its immediate information gain is low. | Operating-condition-controlled generation of full 2D plasma fields on a fixed reactor grid, including rapid parameter sweeps. | Ronneberger, Fischer & Brox, [*U-Net* (2015)](https://arxiv.org/abs/1505.04597); Perez et al., [*FiLM: Visual Reasoning with a General Conditioning Layer* (2018)](https://arxiv.org/abs/1709.07871) |
| `cno_operator_unet` | Multiscale CNO/U-Net hybrid | Consider only if the simple CNO result motivates a multiscale extension; otherwise it adds substantial architectural overlap. | Resolution-consistent multiscale PDE-field surrogate for reconstructing both global discharge structure and localized gradients. | Raonic et al., [*Convolutional Neural Operators for robust and accurate learning of PDEs* (2023)](https://arxiv.org/abs/2302.01178); Ronneberger, Fischer & Brox, [*U-Net* (2015)](https://arxiv.org/abs/1505.04597) |
| `deeponet_plasma_pod` | Table-based POD coefficient regression | A weak discriminator for the present GEC-CCP dataset because gas temperature already acts as a proxy for the three geometries. | Low-cost reconstruction of electron density, ion density, electron temperature, and potential during process-condition optimization. | Lu et al., [*A comprehensive and fair comparison of two neural operators* (2022)](https://arxiv.org/abs/2111.05512) |
| `geom_deeponet_pod` | Explicit geometry-conditioned POD-DeepONet lane | Largely overlaps the adopted descriptor-branch `deeponet_pod`, so it currently has low priority. | Reduced-order reactor-geometry screening when geometry descriptors vary independently from operating conditions. | Lu et al., [*A comprehensive and fair comparison of two neural operators* (2022)](https://arxiv.org/abs/2111.05512) |

## Completed screening additions

These models are retained here because they were previously candidates and
have since received a single-seed common-contract screen.

| Model name | Role | Current assessment | Surrogate-model application example | Reference |
|---|---|---|---|---|
| `cno` | Local convolutional neural-operator comparator | Retain as a useful local-operator comparator. Its seed-412 screen did not beat FNO, so it is not promoted as the leading model. | Grid-based PDE surrogate for repeated full-field evaluations under changing boundary or operating parameters. | Raonic et al., [*Convolutional Neural Operators for robust and accurate learning of PDEs* (2023)](https://arxiv.org/abs/2302.01178) |
| `unetpp` | Nested-skip multiscale convolutional network | The existing recipe did not improve on U-Net; no immediate promotion is recommended. | Fixed-grid reconstruction of plasma fields with multiscale local structures; the cited model was originally developed for segmentation. | Zhou et al., [*UNet++* (2018)](https://arxiv.org/abs/1807.10165) |
| `deeponet_plasma` | Direct branch-trunk DeepONet for plasma fields | Do not adopt without redesigning the branch/trunk representation; its single-run screen was substantially weaker than the leading models. | Sensor- or condition-to-field operator surrogate that predicts plasma quantities at requested coordinates for monitoring or inverse problems. | Lu et al., [*Learning nonlinear operators via DeepONet* (2021)](https://doi.org/10.1038/s42256-021-00302-5) |

## Interpretation notes

- The references identify the architectural starting point, not an exact
  reproduction of every method in the cited paper.
- Hybrid and `-lite` implementations may intentionally differ from their
  source architectures to satisfy the fixed-grid, small-data GEC-CCP contract.
- Final promotion requires the common split, validation-only selection, and
  three learning seeds. A single seed is screening evidence only.
- Detailed status, evaluation rules, and implementation pointers remain in
  the [full model inventory](index.md).
