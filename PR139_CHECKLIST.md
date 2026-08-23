# PR #139 scientific affine-domain closure checklist

## Authority snapshot

- Review source: user-provided additional review of merged PR #138 / `main@e530e53`
- Review timestamp/time zone: 2026-08-23 / Asia-Tokyo
- Reviewer-stated base/head: `main@e530e53a797f801078d6a97befc6fc0df21af265`, PR #138 head `509f50391e87b194735663110c5c936897829a96`
- Verified repository base: `origin/main@e530e53a797f801078d6a97befc6fc0df21af265`
- Verified PR/head/tree: PR #138 `MERGED`; exact-head CI run `32588174999` succeeded
- Worktree state: branch `agent/pr139-affine-scientific-closure`; user-owned untracked `.omx/` preserved
- CI snapshot: PR #138 Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI smoke all `SUCCESS`
- Project boundary: reproducible offline scientific validation; operational deployment is out of scope

## Post-merge evidence

- PR head: `ad11cd819581cd6425edbf4faa1bcdefb6c4847b`
- Merge commit: `74f198361e666e8ac54c68d5d8227359dc2443ef`
- Merged at: `2026-08-23T06:53:25Z`
- Exact-head CI run: `32610382798`
- Required jobs: Python 3.10 CPU `SUCCESS`; Python 3.12 CPU `SUCCESS`; Wheel/CLI smoke `SUCCESS`
- PR status: `MERGED / PASS`

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R139-001 | P1-SCIENTIFIC | Spatial-age gating uses the minimum affine column norm even when shear makes the minimum physical cell displacement smaller. | projected-grid affine → geometry spacing → spatial FSO eligibility | REPRODUCED: reviewer shear has 1000 m minimum axis norm but 447.2136 m minimum singular displacement; the plan uses the former. | repository-actionable | Centralize affine metrics and define `fraction_cells` by the L2 index norm, using the minimum singular value for supported shear. | Rotated/anisotropic orthogonal grids retain their result; supported shears use the contracted minimum displacement; equal axis norms with different shear produce different gates. | ✅ minimum-singular-value spacing and spatial-age algorithm v2 | ✅ targeted + full CPU PASS | ✅ exact-head CI PASS |
| R139-002 | P1-CONTRACT | Direct `RadarSpatialGridIdentity` construction bypasses the condition-number and normalized-determinant policy enforced by `RadarGridTimeContract`. | spatial identity constructor → geometry/evidence production | REPRODUCED: one nearly parallel affine is accepted by direct identity construction and rejected by the grid/time constructor as ill-conditioned. | repository-actionable | Use one affine validation/measurement function in both constructors and bind its thresholds to the current algorithm identity. | An ill-conditioned affine is rejected identically by both constructors; valid affine metrics are identical. | ✅ shared strict affine validator and metrics | ✅ targeted + full CPU PASS | ✅ exact-head CI PASS |
| R139-003 | P1-PROVENANCE | `grid_shape_yx` is not bound to forecast radar tensor dimensions when the run is created or reloaded. | forecast tensors → grid-time contract → serialized run identity | REPRODUCED: a 4×4 forecast frame with a valid 2×2 current projected-grid identity is accepted by `ForecastRunContract.from_inputs()`. | repository-actionable | Add exact spatial-shape validation to current grid contracts and invoke it on creation and durable integrity validation. | H+1/W+1/swapped/degenerate/tampered shapes fail before nowcast; correct shapes retain byte-identical output. | ✅ creation/load shape binding | ✅ targeted + full CPU PASS | ✅ exact-head CI PASS |
| R139-004 | P2-PROVENANCE | Any trimmed string, including geographic or invalid CRS labels, is accepted as a projected metre CRS authority. | projection string → CRS digest → grid/registry/geometry identity | REPRODUCED: `EPSG:4326` and `not-a-crs` both receive valid-looking projected CRS digests. | repository-actionable | Add a small explicit projected-metre CRS registry and canonical semantic payload; reject unsupported/geographic labels. | EPSG:5179 passes; EPSG:4326, unknown strings, non-metre/axis-order mismatches fail. | ✅ current v3 semantic projected-CRS allowlist | ✅ targeted + full CPU PASS | ✅ exact-head CI PASS |
| R139-005 | P2-NUMERICAL | Float64 geometry is cast to any reflectivity floating dtype, allowing half/bfloat16 spatial selection and threshold calculations. | product-derived geometry → source evidence dtype → selection/detection/error | REPRODUCED: product-owned geometry emits float16 and bfloat16 range fields when requested, and current evidence validation accepts every floating dtype. | repository-actionable | Restrict current scientific source evidence to float32/float64 and retain geometry/source-score calculations in an explicitly contracted dtype. | float16/bfloat16 fail; float32/float64 preserve deterministic selection near range boundaries. | ✅ current evidence restricted to float32/float64 | ✅ targeted + full CPU PASS | ✅ exact-head CI PASS |
| R139-006 | P2-GOVERNANCE | PR #138 checklist still records pre-merge HOLD and pending exact-head evidence after merge. | post-merge scientific audit record | REPRODUCED | repository-actionable | Record exact PR head, merge commit/time, CI run/result, and merged status while retaining scientific HOLD gates. | Documentation matches GitHub metadata without claiming confirmatory/publication GO. | ✅ exact post-merge record synchronized | ✅ documentation review PASS | ✅ exact-head CI PASS |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S139-001 | Forecast and verification share one affine identity including origin, axes, rotation, CRS token, and cell-center convention. | `RadarSpatialGridIdentity-v2` and geometry v3 | Current geometry remains product-derived from the shared authority. |
| S139-002 | Current projected coordinates are float64 and replayed with exact equality. | geometry v3 validation and durable geometry shards | Float32 coordinate ingestion remains rejected. |
| S139-003 | Current verification/FSO generations are exact. | verification v14 ↔ FSO v20 ↔ FSOI v16 | Current/legacy cross-generation combinations remain rejected. |
| S139-004 | Historical observation-error v9 identity and replay v18 audit boundary are preserved. | golden v9 digest and legacy replay decoder | Historical golden artifacts remain byte/digest stable. |

## External scientific actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X139-001 | Scientific investigators | Run independent single-/multi-site real-radar cohorts using the closed affine-domain contract. | Fixed legal native radar cases and independent verification observations. | confirmatory claim / external publication | OPEN |
| X139-002 | Scientific investigators | Empirically validate shear/orthogonality choice, representative-tilt geometry, and scan-age thresholds. | Grid-regime, range/elevation, and motion-regime sensitivity analysis. | confirmatory claim / external publication | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P1/P2 is fixed or evidence-disproved.
- [x] Every fix has a targeted regression test.
- [x] Adjacent and broad CPU suites pass.
- [x] Every changed digest preimage and generation has synchronized producers and consumers.
- [x] Historical audit identities remain stable.
- [x] PR #138 post-merge evidence is synchronized.
- [x] PR #139 head equals the reported pushed commit.
- [x] Exact-head CI failures, if any, are classified as code or infrastructure/policy.
- [x] Offline research, confirmatory claims, publication, and deployment remain separate decisions.
- [x] External scientific evidence items remain visible with owner and required action.
- [x] PR #139 merge status is recorded separately from scientific HOLD gates.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Axis-aligned/rotated orthogonal offline research | GO | PR #138 exact-head evidence remains valid while this checklist is resolved. |
| Sheared-grid bounded offline spatial skill/FSO | GO | L2 index displacement is bound to the minimum affine singular value; conditioning is identical at both constructors. |
| Bounded exploratory single-/multi-radar research | CONDITIONAL GO | Use only explicitly validated grid/CRS/dtype contracts; no confirmatory claim. |
| Multi-radar confirmatory claim | HOLD | Repository findings and X139 independent evidence must close. |
| External publication | HOLD | Independent cohort and empirical geometry evidence are required. |
| Operational deployment | NO-GO / OUT OF SCOPE | This project is optimized for scientific validation, not deployment authorization. |
| PR #139 merge | MERGED / PASS | Exact head `ad11cd8…`; merge `74f1983…`; CI `32610382798` succeeded. |

## Implemented scientific generation boundary

```text
RadarSpatialGridIdentity             v2 -> v3
RadarObservationGeometryContract     v3 -> v4
ObservationMaskAlgorithm             v7 -> v8
ObservationErrorAlgorithm            v8 -> v9
ObservationSpatialAgeGate            v1 -> v2
VerificationObservationErrorPlan     v9 -> v10
VerificationMaskEvidence             v7 -> v8
MaskDerivationArtifact               v7 -> v8
ObservationDerivationInputs          v8 -> v9
ObservationErrorDerivationArtifact   v8 -> v9
ObservationErrorContract            v11 -> v12
VerificationBundle                  v14 -> v15
Variational FSO                     v20 -> v21
Variational FSOI                    v16 -> v17
Semantic scoring replay             v19 -> v20
Semantic generation                 v17 -> v18
Scoring case                        v18 -> v19
NeuralPriorHoldoutPlan              v29 -> v30
Package                           0.103 -> 0.104
```

Historical projected-CRS v1 string identity, spatial-grid v2, verification
v14, replay v19, and holdout v29 remain byte-audit-only. Deployment contracts
are unchanged because deployment is outside this scientific PR.

## Local validation evidence

```text
Targeted affine/shape/CRS/dtype and legacy-audit regressions:
  8 passed
  6 subtests passed

Adjacent nowcast, sensitivity, and run-artifact modules:
  287 passed after two stale generation expectations were synchronized
  267 subtests passed

Full CPU FP32/FP64 suite (clean final run):
  802 passed
  427 subtests passed
  18 torch.jit deprecation warnings
  3493.06 seconds

Full product source type check:
  0 errors

Diff and changed-file syntax validation:
  PASS
```
