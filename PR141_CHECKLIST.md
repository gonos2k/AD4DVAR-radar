# PR #141 metric-domain scientific hardening checklist

## Authority snapshot

- Review source: user-provided additional review of merged PR #140 / `main@cf114eb`
- Review timestamp/time zone: 2026-08-23 / Asia-Tokyo
- Reviewer-stated base/head: `main@cf114ebd8b7ee6fb9eb72d09672bbf0aad12f211`, PR #140 head `27b7f60a3f824e58a29e5ea2ec01d42ac6853b4c`
- Verified repository base: `origin/main@cf114ebd8b7ee6fb9eb72d09672bbf0aad12f211`
- Verified prior PR/head/tree: PR #140 `MERGED`; merge commit reachable from `origin/main`; exact-head CI run `32629076703` succeeded
- Worktree state: branch `agent/pr141-metric-domain-hardening`; user-owned untracked `.omx/` preserved
- CI snapshot: PR #140 Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI smoke all `SUCCESS`
- Project boundary: reproducible offline scientific validation; operational deployment is out of scope

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R141-001 | P1-SCIENTIFIC | The current EPSG:5179 identity declares bounded Korean ground-metre semantics without binding the grid and radar coordinates to an allowed study domain or scale-error budget. | CRS identity → grid/radar geometry → physical range, source selection, FSS, and spatial-age interpretation | REPRODUCED: current v4 accepts origin `(0,0)` and radar positions outside the EPSG:5179 area-of-use envelope. | repository-actionable | Introduce one content-addressed metric-domain contract that binds the current CRS to explicit projected bounds, radar/grid inclusion, and a preregistered linear-scale-error authority. | An in-domain grid and every radar pass; origin `(0,0)`, out-of-domain corners/radars, wrong domain digest, and excessive scale error fail before scientific verification. | ✅ metric-domain v1, grid v5, geometry/registry v6 | ✅ targeted PASS | ☐ PR/CI |
| R141-002 | P2-NUMERICAL | An extremely small finite affine can preserve normalized singular metrics while the reconstructed determinant underflows to `0.0`, yielding a valid grid with zero cell area. | affine normalization → physical determinant/cell area → scientific grid identity | REPRODUCED: a `1e-200 m` diagonal v4 grid is accepted with positive spacing and `cell_area_m2 == 0.0`. | repository-actionable | Require the physical determinant and cell area to remain finite and strictly positive, and bind an explicit supported scientific-spacing interval into the current algorithm identity. | `1e-200` diagonal and zero-area reconstruction fail; supported radar spacings pass without changing ordinary affine metrics. | ✅ determinant/area representability plus 1 m–100 km axis range | ✅ targeted PASS | ☐ PR/CI |
| R141-003 | P2-SCIENTIFIC/LEARNING | Metre-configured automated-learning tiles on a sheared affine are basis-aligned parallelograms, while the current contract can be read as a physically equivalent square-tile guardrail. | affine grid → tile shape → whitened-gradient/perturbation norm → candidate ranking | REPRODUCED: the supported shear returns a `(16,16)` tile for 16 km but its projected area is `153.6 km²`, not `256 km²`. | repository-actionable | Permit metre-based automated-learning tile guardrails only for orthogonal affines while preserving shear-aware FSS. | Rotated orthogonal and anisotropic grids pass; sheared FSS remains valid; sheared metre-based learning tiles fail before learning scores or candidate validation. | ✅ metre learning tiles orthogonal-only; sheared FSS retained | ✅ targeted PASS | ☐ PR/CI |
| R141-004 | P2-GOVERNANCE | PR #140 checklist still records pre-merge HOLD/pending evidence after merge. | post-merge scientific audit record | REPRODUCED from the committed checklist and GitHub PR/CI metadata. | repository-actionable | Record exact PR head, merge commit/time, exact-head CI success, and merged status while retaining scientific HOLD gates. | Documentation matches GitHub metadata and does not claim multi-radar confirmatory/publication/deployment GO. | ✅ PR #140 post-merge record synchronized | ✅ documentation review PASS | ☐ PR/CI |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S141-001 | PR #140 restricts the current metric CRS to EPSG:5179 and keeps EPSG:3857 historical-only. | current CRS v3 and legacy v2 identity mapping | EPSG:3857 must remain rejected by current scientific constructors. |
| S141-002 | Affine metrics are scale-normalized and derived NaN/Inf values fail closed. | shared affine validator and extreme-value regressions | Ordinary affine metrics remain unchanged while overflow/cancellation inputs fail. |
| S141-003 | Spatial-age spacing is shape-aware for 2-D, 1×N, N×1, and 1×1 grids. | spatial grid v4 and spatial-age algorithm v3 | Active-axis and unsupported-point behavior remain explicit. |
| S141-004 | Shear-aware soft FSS uses projected-distance support rather than axis-only tile counts. | metric-window construction in the current sensitivity path | Scientific hardening must not disable valid sheared-grid FSS. |

## External scientific actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X141-001 | Scientific investigators | Supply the legally usable fixed study-domain definition and independent real-radar cohort needed for confirmatory claims. | Preregistered domain artifact, native radar cases, and independent verification observations. | multi-radar confirmatory claim / external publication | OPEN |
| X141-002 | Scientific investigators | Quantify EPSG:5179 linear-scale sensitivity over the registered study domain against an independent geodetic reference. | Reproducible corner/interior scale-error report and accepted error budget. | external publication | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P1/P2 is fixed or evidence-disproved.
- [x] Every fix has a targeted regression test.
- [x] Broad scientific CPU suites pass (`570 passed`, `405 subtests`).
- [x] Every changed scientific digest preimage and generation has synchronized producers and consumers.
- [x] Historical v31 holdout and v21 replay identities load as audit-only.
- [x] PR #140 post-merge evidence is synchronized.
- [ ] PR #141 head equals the reported pushed commit.
- [ ] Exact-head CI failures, if any, are classified as code or infrastructure/policy.
- [x] Offline research, confirmatory claims, publication, and deployment remain separate decisions.
- [x] External scientific evidence items remain visible with owner and required action.
- [x] Merge remains HOLD unless explicitly authorized.

## Current decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Externally bounded EPSG:5179 offline research | GO | PR #140 closure remains valid; R141 review concerns repository claims about internally enforced scope. |
| Repository-enforced metric-domain offline research | GO | R141-001 is closed for the registered EPSG:5179 envelope; independent legal/scientific domain approval remains X141-001. |
| Orthogonal/rotated physical metrics | GO | Existing affine and spatial-age closure remains applicable. |
| Sheared-grid FSS | GO | Projected-distance FSS support must be preserved. |
| Sheared-grid metre-based automated-learning tiles | FAIL-CLOSED / HOLD | Current scientific generation rejects them; pixel exploratory tiles remain available without a metre-square claim. |
| Multi-radar confirmatory claim | HOLD | Repository closure plus X141 independent evidence are required. |
| External publication | HOLD | Independent cohort and scale-error evidence are required. |
| Operational deployment | NO-GO / OUT OF SCOPE | The project is optimized for scientific validation, not deployment authorization. |
| PR #141 merge | HOLD | Independent review and exact-head CI are required. |

## Implemented scientific generation boundary

```text
RadarMetricDomainContract             new v1
RadarProjectedCRSIdentity             v3 -> v4
RadarSpatialGridIdentity              v4 -> v5
RadarObservationGeometryContract      v5 -> v6
MosaicObservationSourceRegistry       v5 -> v6
ObservationMaskAlgorithm              v9 -> v10
ObservationErrorAlgorithm            v10 -> v11
VerificationObservationErrorPlan     v11 -> v12
VerificationMaskEvidence              v9 -> v10
MaskDerivationArtifact                v9 -> v10
ObservationDerivationInputs          v10 -> v11
ObservationErrorDerivationArtifact   v10 -> v11
ObservationErrorContract             v13 -> v14
VerificationBundle                   v16 -> v17
Variational FSO                      v22 -> v23
Variational FSOI                     v18 -> v19
Semantic scoring replay              v21 -> v22
Semantic generation                  v19 -> v20
Scoring case                         v20 -> v21
NeuralPriorHoldoutPlan               v31 -> v32
Package                            0.105 -> 0.106
```

Historical grid v4, verification v16, replay v21, and holdout plan v31 remain
audit-only. Deployment generations are unchanged because deployment is outside
this scientific hardening PR.

## Local validation evidence

```text
Current scientific suites (deployment bundle and long promotion suite excluded):
  tests/test_nowcast.py
  tests/test_sensitivity.py
  tests/test_ledger.py
  tests/test_run_artifact.py
  plus acceptance/calibration/CLI/ensemble/matrix-free/metrics/PCG/
  runtime/runtime-closure/variational suites
  570 passed, 405 subtests passed

Current generation and semantic replay focus:
  5 passed

Changed-file syntax and diff validation:
  python -m py_compile src/advar/*.py  PASS
  git diff --check                  PASS

Scientific package evidence:
  4 hashed Linux CPU dependency locks synchronized
  isolated sdist/wheel build PASS (advar-radar-nowcast 0.106.0)

Product source type comparison:
  working tree: 1 error in unchanged runtime_closure.py, 7327 warnings
  origin/main:  1 identical error, 7295 warnings
  classification: pre-existing baseline issue outside this scientific PR
```
