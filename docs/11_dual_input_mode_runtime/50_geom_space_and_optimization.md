# 50. Geom Space and Optimization

## 目的

`table_plus_structure` では cond だけでなく geometry も最適化したい。  
ただし raw mask / raw SDF の画素値を直接最適化してはいけない。

## v1 の原則

- `geom_space` は **意味のある低次元パラメータ**だけにする
- raw pixel optimization は禁止
- `table_only` では `geom_space` は存在してはいけない

## 推奨する `geom_space`

- `part.<id>.tx`
- `part.<id>.ty`
- `part.<id>.scale_x`
- `part.<id>.scale_y`
- `part.<id>.rotation_deg`
- `part.<id>.fillet`
- `gap.<name>`
- `offset.<name>`

必要なら bounded latent を少数だけ使う:
- `z_geom[k]`
- `z_part.<id>[k]`

ただし latent は `descriptor / latent lane` が安定してから有効化する。

## provider の mode

### `fixed`
- 既存 fixed geometry provider 互換
- table_only の既定

### `parametric_parts`
- base geometry + `parts_manifest` + `geom_param` から Geometry Core を再構成する
- table_plus_structure 専用

## optimize API の変更

現行:
```python
run(space, n_trials, geom_ref, axis, ...)
```

変更後:
```python
run(
    space,
    n_trials,
    geom_ref,
    axis,
    geom_space=None,
    joint_objective_cfg=None,
    ...
)
```

### `table_only`
- `geom_space is None` が必須
- `geom_ref.geom_param` が来たら reject

### `table_plus_structure`
- `geom_space` を許可
- backend は cond と geom を同時に提案してよい
- ただし CSV backend も random / optuna と同じ field contract を守る

## objective metadata

推論 summary / benchmark row には少なくとも以下を保存する。

- `input_mode_effective`
- `geom_space_enabled_effective`
- `geom_param_keys_effective`
- `best_cond`
- `best_geom_param`
- `objective_key`
- `n_trials`
