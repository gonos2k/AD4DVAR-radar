# PR #138 scientific spatial-grid closure checklist

## Authority snapshot

- Review source: user-provided additional review of merged PR #137 / latest `main`
- Review timestamp/time zone: 2026-08-23 / Asia-Tokyo
- Reviewer-stated base/head: `main@612598e623ee735690a42224c5dbbf67ec800fe5`, PR #137 head `6b2152e52c9ba05509b3f0636f477438921696a0`
- Verified repository base: `origin/main@612598e623ee735690a42224c5dbbf67ec800fe5`
- Verified PR/head/tree: PR #137 `MERGED`; exact-head CI run `32573374484` succeeded
- Worktree state: branch `agent/pr138-spatial-grid-closure`; user-owned untracked `.omx/` preserved
- CI snapshot: PR #137 Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI smoke all `SUCCESS`
- Project boundary: reproducible offline scientific validation; operational deployment is out of scope

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R138-001 | P1-HIGH | Forecast grid and verification geometry can claim one grid identity while using different axis direction, origin, affine transform, or CRS. | `RadarGridTimeContract` → verification geometry → source selection/FSO | REPRODUCED: current helper creates positive row-y coordinates while the forecast default affine has negative row-y; `_resolve_verification()` compares only the opaque grid digest. | repository-actionable | Introduce one content-addressed projected-grid contract and derive verification coordinates only from its affine transform. | Reject y-axis reversal, half-cell translation, rotation, anisotropy mismatch, CRS mismatch; verify nonzero-radar-y source selection. | ✅ shared affine v2 and product-derived geometry v3 | ✅ focused + full CPU PASS | ☐ |
| R138-002 | P1-NUMERICAL | Fixed 1 mm/1e-6 tolerances accept arbitrary floating dtypes and are incompatible with large-magnitude float32 projected coordinates. | geometry construction and uniform-grid validation | REPRODUCED: v2 accepts every floating dtype while applying fixed `1e-3 m` and `1e-6` checks that are absent from the algorithm preimage. | repository-actionable | Require float64 current coordinates, derive them from the shared affine, and bind exact dtype/tolerance/cell-center policy into the algorithm identity. | Reject float32/float16/bfloat16 and post-quantized coordinates; pass rotated large-origin float64 affine deterministically. | ✅ float64-only coordinates and algorithm-bound numeric policy | ✅ focused + full CPU PASS | ☐ |
| R138-003 | P1-PROVENANCE | PR #137 changed the historical v9 observation-error payload under the same contract name. | legacy observation-error digest and v12 durable audit | REPRODUCED: comparison with `main@7144d925` shows v9 was added retroactively to mask/source and detection/time payload sets. | repository-actionable | Preserve the exact historical v9 payload branch and add a golden legacy digest fixture; corrected lineage remains v10+ only. | PR #136-format v9 payload/digest remains byte- and digest-identical; legacy v12 audit loads without current re-entry. | ✅ historical v9 payload branch restored | ✅ golden + audit-only replay PASS | ☐ |
| R138-004 | P2-GOVERNANCE | `PR137_CHECKLIST.md` still records merge HOLD although PR #137 is merged and exact-head CI succeeded. | post-merge scientific audit record | REPRODUCED: GitHub reports merged commit `612598e...` and successful run `32573374484`, while the document still says merge HOLD and CI pending. | repository-actionable | Record exact PR head, merge commit, CI run/result, and merge time while retaining scientific HOLD gates. | Documentation assertions match GitHub metadata and do not claim confirmatory/publication GO. | ✅ exact post-merge evidence recorded | ✅ documentation review PASS | ☐ |
| R138-005 | P3-MAINTAINABILITY | Verification contract sets repeat `radar-verification-bundle-v13`. | generation capability declarations | REPRODUCED: one accepted-contract set contains v13 twice and the observation-error capability set contains it three times. | repository-actionable | Replace duplicated inline entries with small explicit capability constants. | Static regression asserts current bundle membership once and generation mappings remain exact. | ✅ explicit supported/error capability sets | ✅ focused + full CPU PASS | ☐ |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S138-001 | Geometry v2 binds `grid_contract_digest` into its digest. | `RadarObservationGeometryContract.payload` | Grid-identity mutation changes the current geometry digest. |
| S138-002 | Registry v5 binds its CRS to geometry and states the representative-tilt baseline. | current registry/geometry validation | CRS mismatch, geometry-model mutation, and altitude-role mutation remain fail-closed. |
| S138-003 | Current verification/FSO generations are exact. | v14 ↔ FSO v20 ↔ FSOI v16 | Current/legacy cross-generation combinations remain rejected. |
| S138-004 | Confirmed clear is excluded from quantitative intensity/FSO point metrics. | verification weighting and FSO validation | Clear floor changes cannot change the current point-intensity score. |
| S138-005 | Spatial-age and scalar temporal uncertainty gates remain distinct. | plan and derivation contracts | Stale spatial observations cannot regain FSO eligibility through scalar error inflation. |

## External scientific actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X138-001 | Scientific investigators | Run an independent multi-site real-radar cohort using the shared affine-grid contract and report source-selection/metric sensitivity. | Fixed legal native radar cases and independent verification observations. | confirmatory claim / external publication | OPEN |
| X138-002 | Scientific investigators | Validate representative-tilt and spatial-age assumptions against beam-aware geometry and asynchronous scans. | Range/elevation/motion-regime stratified empirical analysis. | confirmatory claim / external publication | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P1/P2/P3 is fixed or evidence-disproved.
- [x] Every fix has a targeted regression test.
- [x] Adjacent and broad CPU suites pass.
- [x] Every changed digest preimage and generation has synchronized producers and consumers.
- [x] Historical v9/v12 audit identity is preserved by golden evidence.
- [x] PR #137 post-merge evidence is synchronized.
- [ ] PR #138 head equals the reported pushed commit.
- [ ] Exact-head CI failures, if any, are classified as code or infrastructure/policy.
- [x] Offline research, confirmatory claims, publication, and deployment remain separate decisions.
- [x] External scientific evidence items remain visible with owner and required action.
- [x] Merge remains HOLD unless explicitly authorized.

## Implemented scientific generation boundary

```text
RadarSpatialGridIdentity              v1 -> v2
RadarObservationGeometryContract      v2 -> v3
VerificationObservationErrorPlan      v8 -> v9
VerificationObservationMaskEvidence   v6 -> v7
MaskDerivationArtifact                v6 -> v7
ObservationDerivationInputs           v7 -> v8
ObservationErrorDerivationArtifact    v7 -> v8
ObservationErrorContract             v10 -> v11
VerificationBundle                   v13 -> v14
Variational FSO                      v19 -> v20
Variational FSOI                     v15 -> v16
Semantic scoring replay              v18 -> v19
Semantic generation                  v16 -> v17
Scoring case                         v17 -> v18
NeuralPriorHoldoutPlan               v28 -> v29
Package                            0.102 -> 0.103
```

Historical v9 payload semantics and legacy replay v18 remain audit-only. No
deployment generation is advanced because deployment is outside this scientific PR.

## Local validation evidence

```text
Focused spatial-grid, generation, and cold-replay regressions:
  PASS

Adjacent durable ledger/run-artifact modules:
  92 passed
  55 subtests passed

Sensitivity module:
  82 passed
  98 subtests passed

Full CPU suite:
  796 passed
  423 subtests passed
  18 torch.jit deprecation warnings
  3228.77 seconds

Changed scientific source type check:
  0 errors

Repository CI wrapper in this local non-locked environment:
  one pre-existing out-of-scope importlib metadata stub error in
  src/advar/runtime_closure.py:326; the file is unchanged by this PR.
  Exact-head locked CI remains authoritative.
```

Historical PR #136 (`7144d925f43d415133a2defe08d32ee6d0ff1e78`) golden
artifacts were regenerated from that exact tree:

```text
verification-observation-error-contract-v9
  f949f768b5fb9f5b63d157d45df8ab64fde23e9a26314b2ae47b955a6ae69db9

radar-verification-bundle-v12
  c9db72cf16a38f185f49aeedc313c2fc69263a12a0196e182bc327348e63b9b5
```

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Axis-aligned synthetic/offline research | GO | PR #137 exact-head evidence remains valid. |
| Bounded exploratory single-/multi-radar research | CONDITIONAL GO | Use only explicitly validated projected-grid contracts; no confirmatory claim. |
| Multi-radar confirmatory claim | HOLD | R138 repository items and X138 independent evidence must close. |
| External publication | HOLD | Independent cohort and empirical geometry evidence are required. |
| Operational deployment | NO-GO / OUT OF SCOPE | This project is optimized for scientific validation, not deployment authorization. |
| PR #138 merge | HOLD | Independent review and exact-head CI required. |
