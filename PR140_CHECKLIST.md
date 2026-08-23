# PR #140 metric-CRS and affine hardening checklist

## Authority snapshot

- Review source: user-provided additional review of merged PR #139 / `main@74f1983`
- Review timestamp/time zone: 2026-08-23 / Asia-Tokyo
- Reviewer-stated base/head: `main@74f198361e666e8ac54c68d5d8227359dc2443ef`, PR #139 head `ad11cd819581cd6425edbf4faa1bcdefb6c4847b`
- Verified repository base: `origin/main@74f198361e666e8ac54c68d5d8227359dc2443ef`
- Verified PR/head/tree: PR #139 `MERGED`; merge commit reachable from `origin/main`; exact-head CI run `32610382798` succeeded
- Worktree state: branch `agent/pr140-metric-crs-affine-hardening`; user-owned untracked `.omx/` preserved
- CI snapshot: PR #139 Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI smoke all `SUCCESS`
- Final PR #140 head: `27b7f60a3f824e58a29e5ea2ec01d42ac6853b4c`
- Merge commit/time: `cf114ebd8b7ee6fb9eb72d09672bbf0aad12f211` / `2026-08-23T10:19:21Z`
- Exact-head CI: run `32629076703`, all required jobs `SUCCESS`
- PR status: `MERGED / PASS`
- Project boundary: reproducible offline scientific validation; operational deployment is out of scope

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R140-001 | P1-SCIENTIFIC | EPSG:3857 projected metres are accepted as physical ground-distance metres in current verification/FSO. | CRS semantics → radar range/source selection/detection limit/spatial age | REPRODUCED: `radar_projected_crs_semantic_digest("EPSG:3857")` returns a current identity. | repository-actionable | Restrict the current metric scientific CRS authority to EPSG:5179; retain historical CRS identities as audit-only. | EPSG:5179 current identity passes; EPSG:3857, EPSG:4326, and unknown labels fail current construction. | ✅ current CRS v3 permits EPSG:5179 only | ✅ targeted + full CPU PASS | ✅ merged / exact-head CI PASS |
| R140-002 | P1-NUMERICAL/CONTRACT | Finite extreme affine entries can overflow derived determinant/SVD metrics to NaN/Inf and bypass comparison-based conditioning checks. | affine input → derived metrics → spatial identity/grid-time acceptance | REPRODUCED: the finite `1e308` cancellation matrix is accepted and both exposed displacement spacings are `NaN`. | repository-actionable | Normalize before affine metric calculations, require every derived metric finite, and reject unrepresentable physical scales. | Cancellation/overflow matrices fail; ordinary 500 m–10 km affine metrics remain unchanged; explicitly supported large finite diagonal behaves deterministically. | ✅ scaled affine metrics and finite-result gate | ✅ targeted + full CPU PASS | ✅ merged / exact-head CI PASS |
| R140-003 | P2-SCIENTIFIC | A missing axis in 1×N/N×1 grids still contributes to minimum-singular-value spatial-age spacing. | grid shape → physical cell displacement → spatial metric age support | REPRODUCED: a 1×100 grid with 1000 m columns and an unused 100 m row vector reports 100 m spatial spacing. | repository-actionable | Derive active-axis spacing from shape; make 1×1 spatial metrics explicitly unsupported. | 1×N uses column spacing, N×1 uses row spacing, 2-D uses minimum singular spacing, and 1×1 rejects spatial metric use. | ✅ shape-aware spatial spacing and age algorithm v3 | ✅ targeted + full CPU PASS | ✅ merged / exact-head CI PASS |
| R140-004 | P2-GOVERNANCE | PR #139 checklist still records pre-merge HOLD/pending evidence after merge. | post-merge scientific audit record | REPRODUCED from committed checklist and GitHub metadata | repository-actionable | Record exact PR head, merge commit/time, exact-head CI success, and merged status while retaining scientific HOLD gates. | Documentation matches GitHub metadata and does not claim confirmatory/publication/deployment GO. | ✅ PR #139 post-merge evidence synchronized | ✅ documentation review PASS | ✅ merged / exact-head CI PASS |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S140-001 | Sheared-grid spatial age uses the affine minimum singular displacement. | current v15 verification/FSO v21 path | Ordinary orthogonal, rotated, anisotropic, and supported shear cases retain the intended spacing. |
| S140-002 | Forecast and verification share exact shape/origin/affine/CRS identity. | `RadarSpatialGridIdentity-v3` and geometry v4 | Shape mismatch and independent geometry construction remain fail-closed. |
| S140-003 | Current scientific evidence rejects half precision and preserves historical audit generations. | source evidence v8 and legacy audit decoders | FP32/FP64 current paths pass; old artifact identities remain unchanged. |

## External scientific actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X140-001 | Scientific investigators | Run independent single-/multi-site real-radar cohorts under the closed metric-CRS and affine contract. | Fixed legal native radar cases and independent verification observations. | confirmatory claim / external publication | OPEN |
| X140-002 | Scientific investigators | Quantify map-projection, range/elevation, scan-age, and source-selection sensitivity over the study area. | Preregistered CRS/grid/range-regime sensitivity report. | confirmatory claim / external publication | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P1/P2 is fixed or evidence-disproved.
- [x] Every fix has a targeted regression test.
- [x] Adjacent and broad CPU suites pass.
- [x] Changed scientific digest preimages and generations have synchronized producers and consumers.
- [x] Historical audit identities remain stable.
- [x] PR #139 post-merge evidence is synchronized.
- [x] PR #140 head equals `27b7f60a3f824e58a29e5ea2ec01d42ac6853b4c`.
- [x] Exact-head CI run `32629076703` passed all required jobs.
- [x] Offline research, confirmatory claims, publication, and deployment remain separate decisions.
- [x] External scientific evidence items remain visible with owner and required action.
- [x] PR #140 is merged; scientific claim gates remain separately HOLD.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| EPSG:5179 bounded offline scientific research | GO | R140-001 through R140-003 are closed with targeted, adjacent, and full CPU evidence. |
| EPSG:3857 physical-distance research | HOLD | Not a current metric-scientific CRS; any future support requires explicit scale correction and sensitivity evidence. |
| Multi-radar confirmatory claim | HOLD | Repository findings plus X140 independent evidence must close. |
| External publication | HOLD | Independent cohort and projection sensitivity evidence are required. |
| Operational deployment | NO-GO / OUT OF SCOPE | This project is optimized for scientific validation, not deployment authorization. |
| PR #140 merge | MERGED / PASS | Exact head and required CI are recorded above. |

## Implemented scientific generation boundary

```text
RadarProjectedCRSIdentity             v2 -> v3
RadarSpatialGridIdentity              v3 -> v4
RadarObservationGeometryContract      v4 -> v5
ObservationMaskAlgorithm              v8 -> v9
ObservationErrorAlgorithm             v9 -> v10
ObservationSpatialAgeGate             v2 -> v3
VerificationObservationErrorPlan     v10 -> v11
VerificationMaskEvidence              v8 -> v9
MaskDerivationArtifact                v8 -> v9
ObservationDerivationInputs           v9 -> v10
ObservationErrorDerivationArtifact    v9 -> v10
ObservationErrorContract             v12 -> v13
VerificationBundle                   v15 -> v16
Variational FSO                      v21 -> v22
Variational FSOI                     v17 -> v18
Semantic scoring replay              v20 -> v21
Semantic generation                  v18 -> v19
Scoring case                         v19 -> v20
NeuralPriorHoldoutPlan               v30 -> v31
Package                            0.104 -> 0.105
```

Historical CRS v2, spatial-grid v3, replay v20, and holdout v30 remain
byte-audit-only. Deployment contracts are unchanged because deployment is
outside this scientific PR.

## Local validation evidence

```text
Targeted CRS/affine/degenerate-axis and legacy-audit regressions:
  9 passed

Adjacent nowcast, sensitivity, and run-artifact modules:
  289 passed
  96.805 seconds

Full CPU suite:
  805 passed
  3449.025 seconds

Product source type check:
  0 errors
  warnings retained under the repository's current basedpyright policy

Diff and changed-file syntax validation:
  PASS
```
