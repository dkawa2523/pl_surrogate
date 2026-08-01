# Contract Minimization Review

## Scope

Reviewed newly added contract, metadata, schema hash, and validation surfaces
around target groups, output heads, runtime metadata, and preprocessing layout
artifacts.

## Decisions

### Kept Contracts

- `output_layout.json` remains the canonical target order contract.
- `target_role_schema.json` remains the canonical role/positive/family metadata
  contract.
- `runtime_schema_hashes.json` is kept because train, inference, benchmark, and
  checkpoint metadata use it to detect mismatched preprocessing schemas.
- `output_heads.mode` is kept, with `shared` as the default. Shared mode does
  not require target group metadata.
- Grouped output heads keep checkpoint `target_groups` metadata. This is the
  minimal contract needed to reconstruct and validate grouped heads.

### Removed Contracts

- Removed `target_groups_hash` from model attributes, checkpoint metadata, and
  train contract artifacts. It duplicated the persisted `target_groups`
  metadata and made inference errors harder to read.

### Optional Or Metadata-Only Contracts

- `field_layout.json` remains metadata-only for future grid layout readers.
  Train, infer, and benchmark continue to use `output_layout.json`; field
  layout is not a required runtime contract.
- Runtime `target_role_schema` validation for grouped heads is now conditional:
  if runtime schema metadata is available, it is checked against checkpoint
  `target_groups`; if it is absent, checkpoint `target_groups` is treated as the
  grouped-head contract.
- `role_grouped` and `custom_groups` target group validation applies only when
  those modes are selected. Shared allvars baseline does not require target
  groups.

## Validation Notes

- Fail-fast behavior remains for malformed grouped head configs, unknown
  custom targets, duplicate targets, missing custom targets in strict mode, and
  checkpoint/model grouped metadata mismatch.
- Error messages now refer to target group metadata differences rather than
  internal hash values.
- No additional JSON artifact was promoted to a required runtime contract in
  this pass.

## Follow-Up Candidates

- Keep runtime schema hashes focused on preprocessing/runtime compatibility.
  Do not add new hashes unless a real cross-stage mismatch cannot be detected
  from existing metadata.
- Keep `field_layout.json` out of model dispatch until graph, time-series, or
  variable-mesh lanes are actually implemented.
