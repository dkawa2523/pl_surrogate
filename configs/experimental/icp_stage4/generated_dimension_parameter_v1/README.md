# ICP dimension-parameter surrogate v1

Controlled counterpart of the case-varying coil-SDF study.

- Cases: `icp_stage4_dimension_parameter_v1_{unet,uno,coord_mlp}`
- Conditions: `llcoil, rrc, nncoil, rrce, zzc, pp, pp0`
- Spatial input: fixed chamber coordinates, plasma mask, and chamber distance
  only (`geom_v1_mainline`)
- No case-varying coil SDF or coil source map is provided.
- Targets remain in physical linear units and are train-split plasma z-scored.
- Loss, split, seed, and 70 epochs match the SDF-based physical-loss cases.

`coord_mlp` is a coordinate MLP: it predicts each field value from the seven
conditions and the fixed spatial coordinate features.  This avoids a very large
flattened-grid output layer while preserving a direct MLP baseline.

Generate/update the YAML files with:

```powershell
.venv-test\Scripts\python.exe experiments/icp_stage4/scripts/generate_icp_stage4_dimension_parameter_v1.py
```
