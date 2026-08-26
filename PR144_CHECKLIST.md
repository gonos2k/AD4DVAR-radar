# PR #144 set-valued metric geometry closure checklist

## Authority snapshot

- Cycle authority key: `main@b2771c0f513847544ff9a8d91644e75a18eeed4a` + pasted post-merge review of PR #143 + `2026-08-25 Asia/Tokyo`
- Predecessor checklist/PR/merge commit: `PR143_CHECKLIST.md` / PR #143 / `b2771c0f513847544ff9a8d91644e75a18eeed4a`
- Review source: user-provided additional review of `main@b2771c0`
- Reviewer-stated base/head: `main@b2771c0`
- Verified repository base: `origin/main@b2771c0f513847544ff9a8d91644e75a18eeed4a`, tree `6b81aec1237165f483978448cc133357f6b4e7a9`
- Verified predecessor head: PR #143 head `262e9e8a872a04383fd5b1232d22ecbab0d35792`; reachable from `origin/main`
- Worktree: `/Users/yhlee/ADVAR`; branch `agent/pr144-set-valued-metric-geometry`; pre-existing untracked user files inventoried and preserved
- CI snapshot: PR #143 run `32795528631`; Python 3.10 CPU, Python 3.12 CPU, Wheel/CLI smoke all `SUCCESS`; each CPU job `813 passed`, `2 MPS-only skipped`, `448 subtests passed`
- Fresh branch start: exactly `origin/main@b2771c0f513847544ff9a8d91644e75a18eeed4a`

## Deduplication and non-regression ledger

| Finding ID | Semantic fingerprint (`boundary + invariant + acceptance intent`) | Prior finding/status | Current disposition | Existing guard to preserve |
|---|---|---|---|---|
| R144-001 | physical-radius uncertainty → pair echo completeness → every possibly required cell must be present | R143-001 CLOSED; new counterexample to consumer policy | refined repository P1 | certainly-inside support for causal/amplitude consumers |
| R144-002 | physical sidelobe exclusion radius → set-valued annulus → PSR eligibility must not treat a non-monotone subset statistic as certified | R143-001 CLOSED; new non-monotone counterexample | refined repository P1 | exact projected PSR behavior for legacy/non-current grids |
| R144-003 | physical FSS radius → set-valued annulus → confirmatory metric unavailable when footprint membership is unresolved | R143-001 CLOSED; new non-monotone counterexample | refined repository P1 | projected/pixel exploratory FSS and exact affine footprint construction |
| R144-004 | radar projected range interval → max-range/detection/source ranking/spatial-age → current verification must fail closed | R143-001 scope omission | refined repository P1 | product-owned geometry, time, QC, and deterministic source ordering |
| R144-005 | empty operational capability → current forecast-run lineage → nonempty deployment claims must be rejected | R143-002 selector CLOSED; lineage boundary not covered | refined repository P1 | legacy lineage audit decoding and scientific neural-prior lineage |
| R144-006 | interval value object → threshold decisions → invalid direct construction must fail and decision names must be unambiguous | none | new repository P2 | exact endpoint semantics from PR #143 |
| R144-007 | capability registry metadata → construct/serialize/decode/action reachability → every declared capability must be exercised | R143-004 metadata registry CLOSED | refined repository P2 | small explicit registry, no general framework |
| R144-008 | geodetic process bytes → native runtime/tool closure → report reproduction claims must enumerate remaining executable inputs | R143-003 partial runtime closure CLOSED | refined repository P2 | canonical locale/timezone, loader rejection, PROJ/projinfo/proj.db and library hashes |
| X144-001 | governed study domain → polygon authority → publication claim | X143-001 OPEN | external carry-forward | bbox-only claim remains explicit |
| X144-002 | independent geodetic engine/database → scale revalidation → certified ground metric | X143-002 OPEN | external carry-forward | repository evidence remains sampled internal evidence |
| X144-003 | preregistered independent real-radar cohort → generalized skill/publication | X143-003 OPEN | external carry-forward | synthetic/bounded offline evidence only |

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R144-001 | P1-SCIENTIFIC | Pair echo completeness checks only certainly-inside offsets and can accept a missing cell that is possibly inside the physical radius. | metric evidence → pair footprint → common-mask completeness | REPRODUCED on base; CLOSED in branch. | repository-actionable | Add one typed footprint partition and use `possibly_inside` only for completeness while preserving `certainly_inside` for positive support. | At projected 1005 m, radius 1000 m, epsilon 0.006, a missing neighbor makes pair completeness false. | CLOSED: `GroundDistanceFootprint` + consumer-owned `FootprintUse` | PASS: `test_current_metric_uncertainty_reaches_pair_psr_and_speed_gates` | PENDING exact-head CI |
| R144-002 | P1-SCIENTIFIC | Excluding the uncertain annulus from the sidelobe set does not bound PSR because the statistic is non-monotone in membership. | metric evidence → exclusion footprint → phase-correlation pair eligibility | REPRODUCED on base; CLOSED in branch. | repository-actionable | Return an explicit geodetically-uncertain PSR status and make current scientific pair selection unavailable whenever the annulus is nonempty. | The numeric counterexample cannot pass current scientific pair eligibility; legacy projected behavior remains unchanged. | CLOSED: current metre PSR raises `GeodeticMetricUncertaintyError` on nonempty annulus | PASS: targeted PSR annulus regression | PENDING exact-head CI |
| R144-003 | P1-SCIENTIFIC | Core-only physical FSS is not a conservative ground-radius FSS when uncertain offsets exist. | metric evidence → affine FSS footprint → metric/promotion eligibility | REPRODUCED on base; CLOSED in branch. | repository-actionable | For current metric grids, reject confirmatory metre-based FSS when the footprint annulus is nonempty; retain pixel/projected exploratory paths explicitly. | The supplied core `0.600` versus possible `0.795` boundary is unavailable rather than accepted. | CLOSED: current metre FSS rejects set-valued footprint; pixel legacy path preserved | PASS: non-monotone FSS counterexample regression | PENDING exact-head CI |
| R144-004 | P1-SCIENTIFIC | Current verification uses nominal projected range for maximum range, source score, detection limit, range error, and spatial-age spacing. | source geometry → range/detection/score/age → observation state and FSO lineage | REPRODUCED on base; CLOSED in branch. | repository-actionable | Derive lower/upper ground-range tensors, use upper bounds for max range/detection/error, interval source scores with strict dominance, and ground-spacing lower bound for the age gate; bump affected generations. | 100 km projected at epsilon 0.006 is not max-range eligible at 100 km; overlapping source-score intervals yield unassigned; age gate uses projected spacing divided by `1+epsilon`. | CLOSED: upper-bound eligibility/detection/error, strict interval source dominance, lower-bound spacing | PASS: `test_ground_range_bounds_control_verification_selection_and_age` + current v19→FSO v25→FSOI v21 chain | PENDING exact-head CI |
| R144-005 | P1-CONTRACT | Current forecast runs can carry internally consistent deployment-lineage v19 despite an empty operational capability set. | deployment registry → forecast-run constructor/integrity → artifact claim | REPRODUCED on base; CLOSED in branch. | repository-actionable | Reject any nonempty deployment lineage in current run construction/integrity when operational capability is empty; retain v19 as audit-only decoding. | Self-signed/canonical internally valid v19 input is rejected by current run creation and reload; no-lineage scientific prior still passes. | CLOSED: dedicated unsupported error for current nonempty lineage; v17/v19 historical mappings audit-only | PASS: constructor/reload attack + durable v66/v69 audit fixtures | PENDING exact-head CI |
| R144-006 | P2-CONTRACT | Frozen interval dataclasses accept NaN, infinity, negatives, booleans, and reversed bounds; relation names are context-dependent. | interval construction → threshold decision API | REPRODUCED on base; CLOSED in branch. | repository-actionable | Validate interval invariants and return `CERTAINLY_SATISFIES/UNCERTAIN/CERTAINLY_VIOLATES` from explicit maximum/minimum decision methods while retaining a narrow legacy relation adapter if required. | Every invalid direct interval is rejected; endpoint decisions preserve PR #143 mathematics. | CLOSED: shared invariant validator + `ThresholdDecision` API | PASS: invalid constructor and endpoint regressions plus manual epsilon edge audit | PENDING exact-head CI |
| R144-007 | P2-GOVERNANCE | Registry tests compare metadata with hand-created current objects but do not exercise full lifecycle reachability. | capability registry → factory/codec/scientific/operational action | REPRODUCED on base; CLOSED in branch. | repository-actionable | Add a small lifecycle-probe registry for affected families and tests for factory, canonical serialization, audit decode, scientific action, and explicit operational rejection; extend to verification/FSO/replay current graph without adding a general framework. | Every issuable entry constructs/round-trips; scientific entries reach their validator; empty operational sets reject current lineage. | CLOSED: explicit lifecycle probe per family with test discovery/no-skip enforcement | PASS: registry metadata, probe reachability, verification/FSO/replay and operational rejection probes | PENDING exact-head CI |
| R144-008 | P2-PROVENANCE | Metric report does not bind the inspection tool, Python executable/native extensions, or platform shared-cache limitation. | generator process/runtime → geodetic report identity | REPRODUCED on base; CLOSED to host-verifiable scope in branch. | repository-actionable within host limits; full independent/container closure remains X144-002 | Record/hash the dependency-inspection tool, Python executable, relevant native extensions and an explicit shared-cache/container-identity status; reject unsupported unverifiable closure claims; regenerate evidence. | Source/full report checks are byte-identical on the sealed host; field/hash tampering and loader/locale changes fail. | CLOSED: report v3/generator v4 records executable, inspector, native-extension and closure-limit identity | PASS: source check, canonical report digest `9658c906…`, tamper/environment tests | PENDING exact-head CI |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S144-001 | Projected-to-ground distance and speed interval equations and maximum-threshold direction are correct. | `projected_ground_distance_interval()` / speed equivalent and PR #143 boundary tests | New set-valued consumers must reuse the same epsilon and strict endpoint convention. |
| S144-002 | Area interval bounds and threshold-crossing fail-close behavior are mathematically correct. | metric-domain area validators | No distance work may weaken area behavior. |
| S144-003 | Direct operational selector is explicitly unsupported and operational capability sets are empty. | `OperationalDeploymentUnsupportedError` and registry | Current lineage closure must extend this boundary, not reactivate selection. |
| S144-004 | Metric evidence v2 substantially improves locale, loader and dynamic-library provenance. | committed report v2 and generator v3 | New closure fields must be additive and content-addressed. |
| S144-005 | PR #143 exact head passed all required CPU and Wheel/CLI jobs. | run `32795528631` | PR #144 is a forward fix from the reachable merge, not a rollback. |

## External actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X144-001 | scientific governance owner | Supply an approved study polygon, source authority, boundary convention and version. | Repository has bbox evidence only. | polygon-based confirmatory and publication claims | OPEN |
| X144-002 | independent geodesy/reproducibility owner | Revalidate scale bounds with an independent engine and, for strong environment identity, publish a digest-pinned OCI/Nix/Guix closure. | Repository cannot independently certify its own engine or external image. | independent ground-metric certification and publication | OPEN |
| X144-003 | radar-science owner | Supply preregistered independent single/multi-radar real-case cohorts and target provenance. | Current evidence is synthetic/bounded offline. | generalized skill and publication | OPEN |

## Acceptance summary

- [x] Every pasted review statement is represented exactly once and mapped to a semantic fingerprint.
- [x] This cycle starts on a fresh branch exactly at current `origin/main`; PR #143 head is reachable through its merge commit.
- [x] Prior interval, area, affine, audit-generation and direct-selector guards are identified for preservation.
- [x] Every repository-actionable P1/P2 is fixed or evidence-disproved.
- [x] Every fix has a targeted adversarial regression.
- [x] Affected generations, digests, checked-in evidence and documentation are synchronized.
- [x] Focused, adjacent and broad CPU suites pass with exact counts.
- [x] Five consecutive adversarial review passes find no open repository item.
- [ ] PR head equals the pushed commit and exact-head required CI is terminal green.
- [x] External polygon, independent geodesy/environment and real-radar actions remain visible.
- [x] Merge remains HOLD unless explicitly authorized.

## Current decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Internal offline exploratory research | GO WITH EXPLICIT ASSUMPTIONS | Current metric evidence and interval policy are disclosed; external cohort/polygon claims remain excluded. |
| Confirmatory distance/PSR/FSS/verification/FSO | CODE-CLOSED / EXTERNAL HOLD | Set-valued consumers fail closed; X144-001/X144-002/X144-003 remain required for external claims. |
| Publication-suppressed shadow | HOLD | Same scientific semantics and external cohort remain unresolved. |
| Canary | NO-GO / OUT OF SCOPE | Operational deployment is unsupported. |
| State-advancing LIVE | NO-GO / OUT OF SCOPE | Operational deployment is unsupported. |
| External publication | HOLD | X144-001/X144-002/X144-003 and current P1 closure. |
| PR #144 merge | HOLD | Implementation, local verification, independent review and exact-head CI required. |

## Implementation generation ledger

| Family | Predecessor | Current |
|---|---|---|
| package | 0.108.0 | 0.109.0 |
| radar spatial grid | v5 | v6 |
| observation geometry / source registry | v6 | v7 |
| observation error plan | v12 | v13 |
| observation mask / error algorithm | v10 / v11 | v11 / v12 |
| verification bundle | v18 | v19 |
| variational FSO / FSOI | v24 / v20 | v25 / v21 |
| semantic generation / scoring case | v21 / v22 | v22 / v23 |
| semantic replay bundle/method | v23 | v24 |
| holdout plan | v33 | v34 |
| forecast/CLI/action artifact | v69 / v75 / v6 | v70 / v76 / v7 |
| metric report / generator | v2 / v3 | v3 / v4 |

Historical scientific and deployment generations are audit-readable only. No current
contract is operationally accepted, and no operational selector or activation is added.

## Local verification evidence

- `tests/test_run_artifact.py`: 48 passed, 28 subtests.
- Contract registry + nowcast + sensitivity: 255 passed, 268 subtests.
- Ledger + variational + CLI: 211 passed, 132 subtests.
- Remaining adjacent modules: 77 passed, 8 subtests.
- Promotion partitions: 226 passed, 22 subtests.
- Complete partitioned CPU inventory: **817 passed, 458 subtests**, zero failures.
- Additional post-audit regression: 6 passed for v66/v69 audit mapping,
  current lineage rejection and registry reachability.
- Required product-source `basedpyright`: 0 errors, 7487 baseline warnings.
- `compileall`: PASS.
- sdist/wheel build and isolated wheel import/CLI `--help` smoke: PASS for 0.109.0.
- Metric report source/full canonical check: PASS; report SHA-256
  `9658c906cf04f7aa2481c93f87408e803fefe37855b4670704f7ebbb7fb71bcf`.

## Five consecutive adversarial audits

1. **Set-direction audit — PASS.** Completeness uses `possibly_inside`, positive
   support uses `certainly_inside`, and non-monotone PSR/FSS reject nonempty annuli.
2. **Generation/audit-boundary audit — PASS after forward fix.** Found and fixed
   missing `v17-audit` validator coverage and `v19-audit` registry readability;
   durable v66/v69 fixtures now lock both mappings.
3. **Cold replay/tamper audit — PASS.** Replay v24 preserves raw projected source
   cubes and metric evidence, then recomputes source selection, masks, observation
   error and verification v19; no caller-supplied interval output is trusted.
4. **Numerical endpoint audit — PASS.** Invalid interval constructors, equality,
   `epsilon=0`, near-one finite epsilon, strict score overlap and all-invalid source
   behavior fail closed as specified.
5. **Documentation/scope audit — PASS after forward fix.** Removed stale “current
   v66 deployment” wording, marked the certificate state machine historical/audit-only,
   synchronized current generations and retained all external HOLD/NO-GO boundaries.

## Post-merge authority record (append-only)

This section records repository state after the historical pre-merge decisions
above. It does not retroactively rewrite their timing or scientific scope.

| Field | Final evidence |
|---|---|
| PR status | MERGED / PASS |
| PR head | `fb555b3fea945f4e90794d672ce7b69a702a5436` |
| Merge commit | `cc6a70361eb311729575cfe21d7ca1d88361abbd` |
| PR-head tree | `6ec30dd1744473e4b984d58caa987ce8f1a5f57c` |
| Merge tree | `6ec30dd1744473e4b984d58caa987ce8f1a5f57c` |
| Tree identity | SAME |
| Merged at | `2026-08-25T22:57:27Z` |
| PR-head CI | run `32872167216` — SUCCESS |
| Main push CI | run `32908578267` — SUCCESS |
| Required jobs | Python 3.10 CPU, Python 3.12 CPU, Wheel and CLI smoke — SUCCESS |
| Scientific external gates | HOLD — X144-001/X144-002/X144-003 remain open |
| Operational deployment | NO-GO / explicitly out of scope |
