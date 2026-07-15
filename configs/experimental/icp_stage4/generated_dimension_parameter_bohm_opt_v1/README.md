# ICP dimension-parameter Bohm-flux optimization v1

- Cases: `icp_stage4_dimension_parameter_bohm_opt_v1_{unet,uno,coord_mlp}`
- Models: the corresponding trained models under
  `runs/icp_stage4_dimension_parameter_v1/{unet,uno,coord_mlp}`
- Objective: minimize the axisymmetric area-weighted CV of
  `ni * sqrt(Te)` in the two plasma cells nearest the wafer.
- Constraint: retain at least 80% of the reference area-mean Bohm-flux proxy.
- Optimizer: independent Optuna TPE runs, 192 trials per model.

Unlike the SDF case, all structure variables are ordinary condition scalars;
no geometry feature image is regenerated during optimization.  Continuous
dimensions use the dataset 5--95 percentile range. `nncoil=4` is fixed because
the current generic optimizer samples continuous variables and fractional coil
counts are not physical.
