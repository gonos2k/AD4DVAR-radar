# PR #137 scientific geometry identity closure checklist

## Scope boundary

This repository is an offline scientific-validation system, not an operational
deployment product. This checklist closes repository-actionable geometry,
coordinate-frame, and spatial-age semantics only. It does not authorize
shadow, canary, LIVE operation, or an external scientific claim.

## Authority snapshot

- Review source: user-supplied additional review of merged PR #136
- Review timestamp/time zone: 2026-08-22 / Asia-Tokyo
- Reviewer-stated PR/head: PR #136 / `033cc5274173adfc2718c325a51aaa14a9e91ebe`
- Verified repository: `/Users/yhlee/ADVAR`, `gonos2k/AD4DVAR-radar`
- Verified base: `origin/main@7144d925f43d415133a2defe08d32ee6d0ff1e78`
- Verified merge: PR #136 merge commit is reachable from `origin/main`
- Working branch: `agent/pr137-geometry-identity`
- Worktree state: tracked files clean; user-owned untracked `.omx/` preserved
- Verified PR #136 CI: run `32555563042`; Python 3.10 CPU, Python 3.12 CPU,
  and Wheel/CLI smoke all successful

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R137-001 | P1-HIGH | Geometry digest omits `grid_contract_digest` | Geometry content-addressed identity and preregistered observation plan | REPRODUCED: changing only the grid digest retained the same geometry digest | repository-actionable | Add grid identity to a v2 geometry preimage; make v1 audit-only | Changing only grid identity changes geometry digest; v1 cannot create current evidence | ✅ geometry v2 and current-generation rejection | ✅ focused PASS | ☐ |
| R137-002 | P1-SCIENTIFIC | Declared scalar `grid_spacing_m` is not derived from `grid_x_m/grid_y_m` | Spatial scan-age gate | REPRODUCED: a 1 km coordinate grid accepted a declared 10 km spacing | repository-actionable | Restrict current geometry to a uniform rectilinear grid and verify exact derived spacing within a documented absolute tolerance | Inflated spacing, nonuniform grid, and coordinate permutation fail; valid grid passes | ✅ coordinate-derived uniform/isotropic spacing | ✅ focused PASS | ☐ |
| R137-003 | P1-PROVENANCE | Radar projected coordinates are not bound to the grid CRS | Multi-radar source registry to geometry replay | REPRODUCED: registry payload has no CRS identity and geometry arithmetic assumes a shared frame | repository-actionable | Bind one common projected CRS digest to the ordered registry and require exact registry/geometry equality | CRS mismatch and CRS-only mutation fail; geometry replay rejects cross-frame coordinates | ✅ registry v5 exact CRS binding | ✅ focused PASS | ☐ |
| R137-004 | P2-SCIENTIFIC | `radar_altitude_m` is retained but unused while the model may appear to claim full beam geometry | Scientific model interpretation | REPRODUCED: altitude appears only in source validation and payload | repository-actionable | Name and document the current model as projected-horizontal representative-tilt baseline; preserve altitude as provenance-only until a preregistered beam model exists | Contract/payload explicitly encode baseline role; replay preserves the statement | ✅ explicit model/altitude role | ✅ focused PASS | ☐ |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S137-001 | Current v13 verification is exactly bound to FSO v19 and FSOI v15 | Current-generation mapping and focused regression | Current/legacy cross-generation tests remain fail-closed |
| S137-002 | Multi-radar source selection is product-owned and deterministic | Registered physical inputs, eligibility-first selection, ordered tie break | Score-tampering and source-order tests remain collected |
| S137-003 | Confirmed clear is excluded from quantitative intensity and FSO point metrics | Echo-only intensity weight and spatial-age FSO support | Clear-floor invariance and support-score tests remain collected |
| S137-004 | Scan age is represented separately for scalar and spatial metrics | Temporal quality/error model plus spatial displacement gate | Age sensitivity and spatial-support replay tests remain collected |

## External scientific actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X137-001 | Scientific study owner | Supply independent real-radar multi-site cases with authoritative grid/CRS and radar-location metadata | Fixed native inputs and independent verification cohort | Multi-radar confirmatory claim and external publication | OPEN |
| X137-002 | Scientific study owner | Empirically validate representative-tilt approximation and spatial-age parameters by range and storm regime | Preregistered sensitivity analysis | Multi-radar confirmatory claim and external publication | OPEN |

## Implemented generation boundary

The geometry-sensitive current chain is:

```text
RadarObservationGeometryContract   v1 -> v2
MosaicObservationSourceRegistry    v4 -> v5
VerificationObservationErrorPlan   v7 -> v8
VerificationBundle                 v12 -> v13
Variational FSO                     v18 -> v19
Semantic scoring replay             v17 -> v18
NeuralPriorHoldoutPlan               v27 -> v28
Package                              0.101.0 -> 0.102.0
```

Prior generations must remain typed audit-only and must not re-enter current
confirmatory derivation or FSO computation.

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P1/P2 is fixed or evidence-disproved.
- [x] Every fix has a targeted regression test.
- [x] Adjacent and broad CPU suites pass: `792 passed`, `418 subtests passed`
  in `3423.83s` on the local Python 3.10 CPU environment.
- [x] Every changed digest preimage and nested scientific generation is synchronized.
- [x] Package exports and durable replay reconstruction are synchronized.
- [ ] PR head equals the reported pushed commit.
- [ ] CI failures are classified as code or external infrastructure/policy.
- [x] Research, suppressed shadow, canary, LIVE, and external publication have separate decisions.
- [x] External scientific evidence requirements remain visible.
- [x] Merge remains HOLD unless explicitly authorized.

## Local validation evidence

```text
Focused geometry/current-generation regressions:
  4 passed

Full CPU suite:
  792 passed
  418 subtests passed
  18 torch.jit deprecation warnings
  3423.83 seconds

Changed product source type check:
  python -m basedpyright --level error \
    src/advar/sensitivity.py src/advar/promotion.py \
    src/advar/ledger.py src/advar/__init__.py
  0 errors

Repository CI wrapper in the local, non-locked environment:
  one pre-existing out-of-scope error in src/advar/runtime_closure.py:326
  (`PackageMetadata.get` typing); that file is unchanged by this PR.
  Exact-head locked CI remains the authority for the PR/CI column.
```

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Development and offline single-radar research | GO | Full local CPU suite and current deterministic contracts |
| Multi-radar exploratory research | GO for bounded offline studies | Geometry identity closure implemented; representative-tilt limitation remains explicit |
| Multi-radar confirmatory claim | HOLD | X137 independent cohort and empirical parameter evidence |
| Publication-suppressed shadow | NO-GO / out of scope | Project is not developed for deployment |
| Canary | NO-GO / out of scope | Project is not developed for deployment |
| State-advancing LIVE | NO-GO / out of scope | Project is not developed for deployment |
| External publication | HOLD | Independent cohort and scientific calibration required |
| PR merge | HOLD | Review and exact-head CI required |
