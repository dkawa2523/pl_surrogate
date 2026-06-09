# ICDDPS invited talk repo scan summary

???????????????????? artifact ?????????????????????????

## ?????? benchmark
| run root | leaderboard | selected comparison | resolved benchmark | preprocessing | inference |
|---|---:|---:|---:|---:|---:|
| `runs/ext0520` | 53 | 12 | 53 | 1592 | 9701 |
| `runs/benchmarkrun_ext0520` | 59 | 0 | 63 | 1891 | 12306 |
| `runs/icp_stage4_core4` | 36 | 0 | 48 | 1443 | 2459 |
| `runs/periodic_real_tuned_v83` | 21 | 3 | 22 | 655 | 3866 |
| `runs/improvement_cycle_20260401_r2` | 12 | 0 | 24 | 832 | 2136 |
| `runs/report_20260402` | 12 | 1 | 12 | 363 | 2136 |
| `runs/improvement_cycle_20260401` | 12 | 0 | 12 | 450 | 2136 |
| `runs/periodic_real_tuned_v66` | 9 | 1 | 9 | 261 | 1602 |
| `runs/periodic_real_tuned_v51` | 6 | 1 | 6 | 174 | 1068 |
| `runs/periodic_real_tuned_v59` | 4 | 2 | 4 | 116 | 712 |
| `runs/periodic_real_tuned_v50` | 3 | 3 | 3 | 87 | 534 |
| `runs/icp_stage4_struct_spatial_v1_distribution_v2_probe_e20_primary` | 5 | 0 | 6 | 192 | 0 |
| `runs/periodic_real_tuned_v76` | 4 | 1 | 5 | 230 | 712 |
| `runs/icp_stage4_multifield_8field_part_sdf_lite_v1_e80` | 5 | 0 | 5 | 160 | 0 |
| `runs/periodic_real_tuned_v49` | 4 | 1 | 4 | 116 | 712 |
| `runs/periodic_real_tuned_v35` | 4 | 1 | 4 | 116 | 712 |
| `runs/periodic_real_tuned_v33` | 4 | 1 | 4 | 116 | 712 |
| `runs/periodic_real_tuned_v27` | 4 | 1 | 4 | 116 | 712 |
| `runs/periodic_real_tuned_v64` | 3 | 2 | 3 | 87 | 534 |
| `runs/periodic_real_tuned_v45` | 3 | 2 | 3 | 87 | 534 |
| `runs/periodic_real_tuned_v36` | 3 | 2 | 3 | 87 | 534 |
| `runs/icp_stage4_part_sdf_lite_v1_e80_primary` | 4 | 0 | 4 | 128 | 0 |
| `runs/periodic_real_tuned_v78` | 3 | 1 | 3 | 138 | 534 |
| `runs/periodic_real_tuned_v75` | 3 | 1 | 3 | 138 | 534 |
| `runs/periodic_real_tuned_v74` | 3 | 1 | 3 | 138 | 534 |
| `runs/periodic_real_tuned_v73` | 3 | 1 | 3 | 138 | 534 |
| `runs/periodic_real_tuned_v72` | 3 | 1 | 3 | 138 | 534 |
| `runs/periodic_real_tuned_v71` | 3 | 1 | 3 | 138 | 534 |
| `runs/periodic_real_tuned_v70` | 3 | 1 | 3 | 138 | 534 |
| `runs/periodic_real_tuned_v68` | 3 | 1 | 3 | 138 | 534 |

## ??????
- `deeponet`: evidence files 28095
- `unet`: evidence files 22305
- `deeponet_plasma`: evidence files 17299
- `coord_mlp`: evidence files 15830
- `fno`: evidence files 11911
- `global_mlp`: evidence files 9625
- `unetpp`: evidence files 6395
- `ffno`: evidence files 5912
- `deeponet_pod`: evidence files 5313
- `coord_mlp_siren`: evidence files 3654
- `cno`: evidence files 3480
- `unetpp_attn`: evidence files 3437
- `coord_mlp_fourier`: evidence files 1677
- `u_no`: evidence files 1554
- `cno_operator_unet`: evidence files 701

## ??? target
- `ne`: config occurrences 129
- `ni`: config occurrences 129
- `Te`: config occurrences 129
- `phi`: config occurrences 129
- `Br`: config occurrences 12
- `Bz`: config occurrences 12
- `Jelr`: config occurrences 12
- `Jelz`: config occurrences 12

## ??? spatial prediction
- `configs/experimental/icp_stage4/conference_structure_feature_study.yaml`
- `configs/experimental/icp_stage4/icp_stage4_core4_cno_smoke.yaml`
- `runs/benchmark_m2/models/coord_mlp/inference/batch/summary.csv`
- `runs/benchmark_m2/models/coord_mlp/inference/optimize/best.json`
- `runs/benchmark_m2/models/coord_mlp/inference/optimize/summary.json`
- `runs/benchmark_m2/models/coord_mlp/inference/optimize/trials.csv`
- `runs/benchmark_m2/models/coord_mlp/inference/single/096d4c6c22680f9f5c941b65177cbbfc47f0578e/derived.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/096d4c6c22680f9f5c941b65177cbbfc47f0578e/diagnostics.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/096d4c6c22680f9f5c941b65177cbbfc47f0578e/diagnostics_maps.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/096d4c6c22680f9f5c941b65177cbbfc47f0578e/fields_model.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/096d4c6c22680f9f5c941b65177cbbfc47f0578e/fields_phys.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/096d4c6c22680f9f5c941b65177cbbfc47f0578e/ood_report.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/096d4c6c22680f9f5c941b65177cbbfc47f0578e/qoi.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/13ccd020131452a05cf5e2e94b7e21f760b8e76a/derived.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/13ccd020131452a05cf5e2e94b7e21f760b8e76a/diagnostics.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/13ccd020131452a05cf5e2e94b7e21f760b8e76a/diagnostics_maps.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/13ccd020131452a05cf5e2e94b7e21f760b8e76a/fields_model.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/13ccd020131452a05cf5e2e94b7e21f760b8e76a/fields_phys.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/13ccd020131452a05cf5e2e94b7e21f760b8e76a/ood_report.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/13ccd020131452a05cf5e2e94b7e21f760b8e76a/qoi.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/13ccd020131452a05cf5e2e94b7e21f760b8e76a/warnings.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/4ce955ae0dd8a76902f1c584367f9da294e59705/derived.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/4ce955ae0dd8a76902f1c584367f9da294e59705/diagnostics.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/4ce955ae0dd8a76902f1c584367f9da294e59705/diagnostics_maps.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/4ce955ae0dd8a76902f1c584367f9da294e59705/fields_model.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/4ce955ae0dd8a76902f1c584367f9da294e59705/fields_phys.npz`
- `runs/benchmark_m2/models/coord_mlp/inference/single/4ce955ae0dd8a76902f1c584367f9da294e59705/ood_report.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/4ce955ae0dd8a76902f1c584367f9da294e59705/qoi.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/4ce955ae0dd8a76902f1c584367f9da294e59705/warnings.json`
- `runs/benchmark_m2/models/coord_mlp/inference/single/4f7f332ab9bf3744afd1494922a8d06c7f16e43f/derived.npz`
- ... first 30 of 80 candidate paths

## ??? geometry / coord feature
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_cno.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_cno_operator_unet.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_coord_mlp_fourier.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_coord_mlp_pod_residual.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_coord_mlp_siren.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_deeponet_plasma.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_deeponet_plasma_pod.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_deeponet_pod.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_ffno.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_fno.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_geom_deeponet_pod.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_geom_deeponet_siren.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_global_mlp.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_u_no.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_unet.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_unet_operator_v2.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_unetpp.yaml`
- `configs/benchmarkrun_ext0520/templates/benchmark_ext0520_unetpp_attn.yaml`
- `configs/experimental/dual_input_modes/table_only_deeponet_pod.yaml`
- `configs/experimental/dual_input_modes/table_only_global_mlp.yaml`
- `configs/experimental/dual_input_modes/table_plus_structure_cno_part_lite.yaml`
- `configs/experimental/dual_input_modes/table_plus_structure_coord_mlp_fourier_part_lite.yaml`
- `configs/experimental/dual_input_modes/table_plus_structure_deeponet_pod_descriptor.yaml`
- `configs/experimental/dual_input_modes/table_plus_structure_ffno_part_lite.yaml`
- `configs/experimental/dual_input_modes/table_plus_structure_geom_deeponet_siren_part_lite.yaml`
- `configs/experimental/dual_input_modes/table_plus_structure_u_no_part_lite.yaml`
- `configs/experimental/icp_stage4/conference_structure_feature_study.yaml`
- `configs/experimental/icp_stage4/generated/full/benchmark_icp_stage4_core4_full_cno.yaml`
- `configs/experimental/icp_stage4/generated/full/benchmark_icp_stage4_core4_full_ffno.yaml`
- `configs/experimental/icp_stage4/generated/full/benchmark_icp_stage4_core4_full_global_mlp.yaml`
- ... first 30 of 80 candidate paths

## GEC-CCP ????????????
- `GEC`, `CCP`, `reference cell` ????? evidence ? data/config artifact ???????

## ICP / coil geometry ???????
- ICP / coil / geometry source ??? evidence ???????
  - `configs/experimental/icp_stage4/conference_structure_feature_study.yaml`
  - `configs/experimental/icp_stage4/generated/full/benchmark_icp_stage4_core4_full_cno.yaml`
  - `configs/experimental/icp_stage4/generated/full/benchmark_icp_stage4_core4_full_ffno.yaml`
  - `configs/experimental/icp_stage4/generated/full/benchmark_icp_stage4_core4_full_global_mlp.yaml`
  - `configs/experimental/icp_stage4/generated/full/benchmark_icp_stage4_core4_full_unet.yaml`
  - `configs/experimental/icp_stage4/generated/smoke/benchmark_icp_stage4_core4_smoke_cno.yaml`
  - `configs/experimental/icp_stage4/generated/smoke/benchmark_icp_stage4_core4_smoke_ffno.yaml`
  - `configs/experimental/icp_stage4/generated/smoke/benchmark_icp_stage4_core4_smoke_global_mlp.yaml`
  - `configs/experimental/icp_stage4/generated/smoke/benchmark_icp_stage4_core4_smoke_unet.yaml`
  - `configs/experimental/icp_stage4/generated_conference_part_lite_v1_e80/full/benchmark_icp_stage4_core4_full_cno_operator_unet.yaml`
  - `configs/experimental/icp_stage4/generated_conference_part_lite_v1_e80/full/benchmark_icp_stage4_core4_full_ffno.yaml`
  - `configs/experimental/icp_stage4/generated_conference_part_lite_v1_e80/full/benchmark_icp_stage4_core4_full_unet.yaml`
  - `configs/experimental/icp_stage4/generated_conference_part_lite_v1_e80/smoke/benchmark_icp_stage4_core4_smoke_cno_operator_unet.yaml`
  - `configs/experimental/icp_stage4/generated_conference_part_lite_v1_e80/smoke/benchmark_icp_stage4_core4_smoke_ffno.yaml`
  - `configs/experimental/icp_stage4/generated_conference_part_lite_v1_e80/smoke/benchmark_icp_stage4_core4_smoke_unet.yaml`
  - `configs/experimental/icp_stage4/generated_conference_part_sdf_lite_v1_e80_baseline/full/benchmark_icp_stage4_core4_full_ffno.yaml`
  - `configs/experimental/icp_stage4/generated_conference_part_sdf_lite_v1_e80_baseline/smoke/benchmark_icp_stage4_core4_smoke_ffno.yaml`
  - `configs/experimental/icp_stage4/generated_conference_struct_spatial_v1_e80_baseline/full/benchmark_icp_stage4_core4_full_ffno.yaml`
  - `configs/experimental/icp_stage4/generated_conference_struct_spatial_v1_e80_baseline/smoke/benchmark_icp_stage4_core4_smoke_ffno.yaml`
  - `configs/experimental/icp_stage4/generated_full_e80_extrap_primary/full/benchmark_icp_stage4_core4_full_cno.yaml`
  - `configs/experimental/icp_stage4/generated_full_e80_extrap_primary/full/benchmark_icp_stage4_core4_full_ffno.yaml`
  - `configs/experimental/icp_stage4/generated_full_e80_extrap_primary/full/benchmark_icp_stage4_core4_full_unet.yaml`
  - `configs/experimental/icp_stage4/generated_full_e80_extrap_primary_normal/full/benchmark_icp_stage4_core4_full_cno.yaml`
  - `configs/experimental/icp_stage4/generated_full_e80_extrap_primary_normal/full/benchmark_icp_stage4_core4_full_ffno.yaml`
  - `configs/experimental/icp_stage4/generated_full_e80_extrap_primary_normal/full/benchmark_icp_stage4_core4_full_unet.yaml`
  - `configs/experimental/icp_stage4/generated_full_e80_extrap_primary_unrestricted/full/benchmark_icp_stage4_core4_full_cno.yaml`
  - `configs/experimental/icp_stage4/generated_full_e80_extrap_primary_unrestricted/full/benchmark_icp_stage4_core4_full_ffno.yaml`
  - `configs/experimental/icp_stage4/generated_full_e80_extrap_primary_unrestricted/full/benchmark_icp_stage4_core4_full_unet.yaml`
  - `configs/experimental/icp_stage4/generated_full_pilot_linear/full/benchmark_icp_stage4_core4_full_cno.yaml`
  - `configs/experimental/icp_stage4/generated_full_pilot_linear/full/benchmark_icp_stage4_core4_full_ffno.yaml`

## ?????????? schematic ????????
- GEC/ICP chamber schematic: ?????????????????????????
- Coil-to-structure-feature workflow: coil layout ?? mask / SDF / boundary / part summary ????????
- Surrogate workflow: process condition ???????? 2D field ???? QoI ?????????

## Inventory counts
- `inference_artifact`: 83796
- `preprocessing_artifact`: 15144
- `dataset_npz`: 5029
- `resolved_benchmark_json`: 473
- `leaderboard_csv`: 440
- `config_yaml`: 147
- `selected_models_comparison_csv`: 103
- `report_markdown`: 36
- `dataset_index_csv`: 9
- `artifact_manifest_json`: 4

## Config input modes
- `table_plus_structure`: 114
- `table_only`: 7

??? `../tables/artifact_inventory.csv` ? `../tables/config_contract_audit.csv` ????
