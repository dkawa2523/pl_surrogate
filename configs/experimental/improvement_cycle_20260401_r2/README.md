# Improvement Cycle 20260401 R2

This directory stores the fixed 12-run matrix for the R2 improvement cycle:

1. `C3_coord_siren_control`
2. `C4_coord_siren_gated_affine`
3. `C5_coord_siren_gated_affine_sel`
4. `C6_coord_siren_gated_affine_sel_mt`
5. `C7_coord_siren_gated_affine_w0_res`
6. `D7_deeponet_control`
7. `D8_deeponet_tephi_sel`
8. `D9_deeponet_tephi_sel_mt`
9. `D10_deeponet_fused_learned_mix`
10. `U5_unetpp_attn_control`
11. `U6_unetpp_attn_density_bias`
12. `U7_unetpp_density_bias_port`

Materialize fixtures:

```powershell
python scripts/materialize_improvement_cycle_20260401_r2.py
```

Generated fixtures are emitted to:

`tests/fixtures/generated/improvement_cycle_20260401_r2/`
