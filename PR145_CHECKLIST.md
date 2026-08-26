# PR #145 detection-limit interval and directed-boundary closure checklist

## Authority snapshot

- Cycle authority key: `main@cc6a70361eb311729575cfe21d7ca1d88361abbd` + user-provided post-merge review of PR #144 + `2026-08-26 Asia/Tokyo`
- Predecessor checklist/PR/merge commit: `PR144_CHECKLIST.md` / PR #144 / `cc6a70361eb311729575cfe21d7ca1d88361abbd`
- Review source: user-provided additional review of current `main`
- Reviewer-stated base/head: `main@cc6a70361eb311729575cfe21d7ca1d88361abbd`
- Verified repository base: `origin/main@cc6a70361eb311729575cfe21d7ca1d88361abbd`, tree `6ec30dd1744473e4b984d58caa987ce8f1a5f57c`
- Verified predecessor head: PR #144 head `fb555b3fea945f4e90794d672ce7b69a702a5436`; reachable from `origin/main`
- Worktree: `/Users/yhlee/ADVAR`; branch `agent/pr145-detection-interval-directed-rounding`; pre-existing untracked user files inventoried and preserved
- CI snapshot: PR #144 required checks `SUCCESS`; main push run `32908578267` Python 3.10 CPU, Python 3.12 CPU/basedpyright, and Wheel/CLI smoke all `SUCCESS`
- Fresh branch start: exactly `origin/main@cc6a70361eb311729575cfe21d7ca1d88361abbd`

## Deduplication and non-regression ledger

| Finding ID | Semantic fingerprint (`boundary + invariant + acceptance intent`) | Prior finding/status | Current disposition | Existing guard to preserve |
|---|---|---|---|---|
| R145-001 | ground-range interval → detection-limit interval → report-kind certification must use category-directed bounds | R144-004 CLOSED for upper-bound echo/range semantics; new category-direction counterexample | refined repository P1 | strict source-score dominance, range-upper eligibility/error, report-kind threshold equality convention |
| R145-002 | floating-point interval/footprint → directed enclosure → certain/possible sets and maximum physical gates cannot be relaxed | R144-001/R144-006 CLOSED at nominal endpoints; new ULP-boundary counterexample | refined repository P1 | `certainly_inside / uncertain / possibly_inside`, exact threshold decisions, invalid interval constructors |
| R145-003 | geodetic execution roots → Linux dependency inspection → non-ELF inspector must not be passed to `ldd` | R144-008 CLOSED only on the producing macOS host; new Linux portability counterexample | refined repository P1 | inspector identity/hash, unresolved-library fail-close, loader/locale rejection |
| R145-004 | Python process runtime → native-extension enumeration → closure claim must describe all enumerated loaded extensions | R144-008 acknowledged host limit | refined repository P2 | explicit independent sealed-environment requirement |
| R145-005 | contract capability registry → runtime constants/lifecycle reachability → registry must be the current-generation authority | R144-007 partially closed with metadata/probe-name checks | refined repository P2 | readable generated README table, empty operational acceptance sets |
| R145-006 | merged PR authority record → checklist governance → GitHub status and scientific HOLD must be represented separately | PR144 checklist pre-merge record | new repository P2 | append-only prior finding decisions; external HOLD and operational NO-GO |
| X145-001 | governed study domain → polygon authority → publication claim | X144-001 OPEN | external carry-forward | bbox-only claim remains explicit |
| X145-002 | independent geodetic engine/database/environment → scale revalidation → certified ground metric | X144-002 OPEN | external carry-forward | repository evidence remains sampled internal evidence |
| X145-003 | preregistered independent real-radar cohort → generalized skill/publication | X144-003 OPEN | external carry-forward | synthetic/bounded offline evidence only |

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R145-001 | P1-SCIENTIFIC | Current verification collapses the distance-dependent detection limit to its upper bound for echo, clear, and censored categories, certifying ambiguous clear/censored cells. | projected range → ground-range interval → detection-limit/report-kind state → verification/FSO | REPRODUCED: at 100 km, current `T+` accepts `Z=0` as clear/censored while `T- < Z < T+`. | repository-actionable | Return lower/upper detection-limit tensors, apply upper to detected and lower to clear/censored, preserve an explicit uncertainty mask, and fail source selection/verification closed. | At projected 100 km and epsilon 0.006, a 0 dBZ observation between the two limits is uncertain for all three report kinds; values strictly outside the interval certify only the matching category. | CLOSED: lower/upper tensors, category-directed certification, explicit uncertainty and zero source score | PASS: category counterexample, source-row replay, v20 bundle/cold replay | PENDING exact-head CI |
| R145-002 | P1-NUMERICAL | Nearest-rounded interval endpoints, `+64 ULP` certain-footprint inclusion, and generic absolute tolerance on maximum speed can admit values outside physical limits. | scalar/tensor interval arithmetic → footprint partition and motion gates | REPRODUCED: `994+32 ulp` is current-certain with ground upper >1000 m; 10.0000005 m/s is accepted at a 10 m/s maximum. | repository-actionable | Add outward rounding, classify footprint from directed distance bounds, and remove generic tolerance from physical maximum gates. | `994 m + 32 ulp` is uncertain, not certain; 10.0000005 m/s is rejected at a 10 m/s maximum; float32 interval overlap remains unassigned. | CLOSED: scalar/tensor outward rounding and hard physical maximum gates | PASS: 32-ULP footprint, barely excessive speed, float32/float64 interval tests | PENDING exact-head CI |
| R145-003 | P1-PORTABILITY | Linux full evidence generation can execute `ldd /usr/bin/ldd` and fail because the inspector is a shell script. | evidence generator → dependency root enumeration → Linux closure | REPRODUCED structurally: `_execution_environment()` includes the inspector in roots and Linux closure unconditionally calls `ldd` for every root. | repository-actionable | Hash the inspector separately, inspect only ELF dynamic roots, bind the shebang interpreter separately, and add an Ubuntu self-test. | A script inspector is never passed to `ldd`; all ELF roots are inspected; unresolved libraries and loader overrides fail closed. | CLOSED: inspector identity separated; ELF/Mach-O roots and shebang chain explicit | PASS locally; required Ubuntu self-test added to Wheel/CLI job | PENDING exact-head CI |
| R145-004 | P2-PROVENANCE | The report names only `_sqlite3` and `_decimal`, not the complete enumerated native-extension set loaded by the generator. | Python execution process → native extension/runtime closure → report claim | REPRODUCED: the generator process currently loads additional `_hashlib`, `_json`, `_posixsubprocess`, `math`, `select`, and other native modules. | repository-actionable | Enumerate loaded extension modules deterministically and downgrade closure wording to the exact measured scope unless sealed image identity exists. | Synthetic/real loaded extension sets are sorted, unique, hashed, closure-inspected when applicable, and represented in the report digest. | CLOSED: loaded file-backed extensions are sorted, hashed and closure-bound; claim is scoped to enumerated roots | PASS: generator source binding, environment self-test and report v4 validation | PENDING exact-head CI |
| R145-005 | P2-GOVERNANCE | Runtime contract constants duplicate registry strings and lifecycle probes prove only AST presence, not executable family reachability. | registry → production constants → construct/round-trip/scientific/operational lifecycle | REPRODUCED: FSO/FSOI/verification constants are hand-written outside the registry and lifecycle checks only inspect named AST methods. | repository-actionable | Add typed `current_contract()` lookup, derive current production constants from it, and execute explicit family lifecycle cases rather than checking test-name strings alone. | Current verification/FSO/FSOI/replay constants equal registry lookups; lifecycle cases actually run; empty operational sets reject action. | CLOSED: runtime current constants derive from registry; file-based executable lifecycle probes | PASS under `python -I -m pytest`; deployment acceptance remains empty | PENDING exact-head CI |
| R145-006 | P2-GOVERNANCE | `PR144_CHECKLIST.md` still records pre-merge PR/CI state after the exact tree merged and main push CI passed. | repository audit record → GitHub authority state | REPRODUCED against GitHub: PR #144 is merged at `cc6a703…`; PR-head and main-push checks are all successful. | repository-actionable | Add an append-only post-merge authority section without rewriting prior scientific dispositions. | Checklist records head, merge commit, identical tree, PR-head/main-push CI success, external HOLD, and operational NO-GO separately. | CLOSED: append-only PR #144 authority record added | PASS: head/merge/tree/PR-head CI/main-push CI and scientific HOLD are separate | PENDING exact-head CI |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S145-001 | Distance/speed interval equations and maximum-bound direction are conceptually correct. | PR #144 scalar interval API and tests | Directed rounding may widen but must not reverse the equations or endpoint convention. |
| S145-002 | Set-valued footprint consumers are correctly separated. | completeness uses possible; causal support uses certain; PSR/FSS reject annulus | New numerical fix must preserve consumer-specific policies. |
| S145-003 | Source selection uses strict lower-over-competing-upper dominance. | source-selection v3 and current verification tests | Detection classification uncertainty must not restore nominal argmax. |
| S145-004 | Current operational selection and deployment lineage are explicitly unsupported. | empty operational sets and dedicated error | No deployment activation or positive operational path may be added. |
| S145-005 | Geodetic evidence binds locale, loader environment, tools, database and enumerated host libraries. | metric report v4/generator v5 | Portability fix retains unresolved-library and loader-override fail-close. |

## External actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X145-001 | scientific governance owner | Supply an approved study polygon, source authority, boundary convention and version. | Repository has bbox evidence only. | polygon-based confirmatory and publication claims | OPEN |
| X145-002 | independent geodesy/reproducibility owner | Revalidate scale bounds with an independent engine and publish a sealed execution-environment identity. | Repository cannot independently certify its own engine/environment. | independent ground-metric certification and publication | OPEN |
| X145-003 | radar-science owner | Supply preregistered independent single/multi-radar real-case cohorts and target provenance. | Current evidence is synthetic/bounded offline. | generalized skill and publication | OPEN |

## Acceptance summary

- [x] Every pasted review statement is represented exactly once and fingerprinted.
- [x] The post-merge cycle starts on a fresh branch exactly at `origin/main`; PR #144 head is reachable through its merge commit.
- [x] Existing interval, footprint, source dominance, audit-generation and deployment NO-GO guards are identified for preservation.
- [x] Every claim is independently reproduced or evidence-disproved on the current tree.
- [x] Every repository-actionable P1/P2 is fixed or evidence-disproved.
- [x] Every fix has a targeted adversarial regression.
- [x] Affected contract generations, digests, evidence and documentation are synchronized.
- [x] Focused, adjacent and broad CPU suites pass: `819 passed`, `463 subtests passed`; basedpyright `0 errors`; wheel/CLI smoke PASS.
- [x] Five consecutive adversarial review passes find no open repository item: science, generation/replay, numerical boundaries, Linux provenance, and governance/operational boundary.
- [ ] PR head equals the pushed commit and exact-head required CI is terminal green.
- [x] External polygon, independent geodesy/environment and real-radar actions remain visible.
- [x] Merge remains HOLD unless explicitly authorized.

## Consecutive adversarial review record

| Pass | Attack surface | Result |
|---:|---|---|
| 1 | Detection-limit monotonicity, category-directed inequalities, uncertain-source eligibility and selected-source gather | PASS — independent 100 km counterexample reproduces `T- < 0 < T+`; all three report kinds remain uncertain and receive source score zero. |
| 2 | Current/legacy generation graph, algorithm digests, required tensor roles, cold replay and verification→FSO→FSOI→scoring mapping | PASS — v4/v6/v7/v14/v20/v26/v22/v25 graph and three new source tensor roles are exact; round-trip tests pass. |
| 3 | Scalar/tensor outward rounding, ULP footprint, float32/float64 bounds and hard maximum-speed gate | PASS — directed enclosures hold; 32-ULP boundary is uncertain; barely excessive speed is rejected. |
| 4 | Linux/macOS dependency inspector separation, shebang chain, native extensions and loader override | PASS — clean self-test succeeds; injected `LD_LIBRARY_PATH` fails closed; inspector is not a dynamic closure root. |
| 5 | Empty operational acceptance, current deployment-lineage rejection, runtime registry authority, README narrative and staging scope | PASS after fixing stale README generation prose; user untracked files remain excluded. |

## Current decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Internal offline exploratory research | GO WITH EXPLICIT ASSUMPTIONS | Repository P1/P2 findings are code-closed; external study-domain/geodesy/cohort assumptions remain explicit. |
| Confirmatory verification/FSO/FSOI | CODE-CLOSED / EXTERNAL HOLD | R145-001/R145-002 are closed; X145-001/X145-002/X145-003 remain required. |
| Publication-suppressed shadow | HOLD | External polygon, independent geodesy and cohort evidence remain unresolved. |
| Canary | NO-GO / OUT OF SCOPE | Operational deployment is unsupported. |
| State-advancing LIVE | NO-GO / OUT OF SCOPE | Operational deployment is unsupported. |
| External publication | HOLD | X145-001/X145-002/X145-003 and repository P1 closure required. |
| PR #145 merge | HOLD | Implementation, local verification, independent review and exact-head CI required. |
