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

## PR #145 implementation-head authority

- PR: [#145](https://github.com/gonos2k/AD4DVAR-radar/pull/145)
- Implementation head: `e244d85060e0941d7d82fc3029cf0cd0be85e0a3`
- Exact-head CI run: `32977924481` — `SUCCESS`
- Required jobs: Python 3.10 CPU `SUCCESS`; Python 3.12 CPU/basedpyright `SUCCESS`; Wheel/CLI smoke plus Linux geodetic environment self-test `SUCCESS`
- GitHub state at verification: `OPEN`; mergeability `MERGEABLE`
- Merge authority: `HOLD`; exact-head CI success does not authorize merge or change external scientific gates

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
| R145-001 | P1-SCIENTIFIC | Current verification collapses the distance-dependent detection limit to its upper bound for echo, clear, and censored categories, certifying ambiguous clear/censored cells. | projected range → ground-range interval → detection-limit/report-kind state → verification/FSO | REPRODUCED: at 100 km, current `T+` accepts `Z=0` as clear/censored while `T- < Z < T+`. | repository-actionable | Return lower/upper detection-limit tensors, apply upper to detected and lower to clear/censored, preserve an explicit uncertainty mask, and fail source selection/verification closed. | At projected 100 km and epsilon 0.006, a 0 dBZ observation between the two limits is uncertain for all three report kinds; values strictly outside the interval certify only the matching category. | CLOSED: lower/upper tensors, category-directed certification, explicit uncertainty and zero source score | PASS: category counterexample, source-row replay, v20 bundle/cold replay | PASS: run `32977924481` |
| R145-002 | P1-NUMERICAL | Nearest-rounded interval endpoints, `+64 ULP` certain-footprint inclusion, and generic absolute tolerance on maximum speed can admit values outside physical limits. | scalar/tensor interval arithmetic → footprint partition and motion gates | REPRODUCED: `994+32 ulp` is current-certain with ground upper >1000 m; 10.0000005 m/s is accepted at a 10 m/s maximum. | repository-actionable | Add outward rounding, classify footprint from directed distance bounds, and remove generic tolerance from physical maximum gates. | `994 m + 32 ulp` is uncertain, not certain; 10.0000005 m/s is rejected at a 10 m/s maximum; float32 interval overlap remains unassigned. | CLOSED: scalar/tensor outward rounding and hard physical maximum gates | PASS: 32-ULP footprint, barely excessive speed, float32/float64 interval tests | PASS: run `32977924481` |
| R145-003 | P1-PORTABILITY | Linux full evidence generation can execute `ldd /usr/bin/ldd` and fail because the inspector is a shell script. | evidence generator → dependency root enumeration → Linux closure | REPRODUCED structurally: `_execution_environment()` includes the inspector in roots and Linux closure unconditionally calls `ldd` for every root. | repository-actionable | Hash the inspector separately, inspect only ELF dynamic roots, bind the shebang interpreter separately, and add an Ubuntu self-test. | A script inspector is never passed to `ldd`; all ELF roots are inspected; unresolved libraries and loader overrides fail closed. | CLOSED: inspector identity separated; ELF/Mach-O roots and shebang chain explicit | PASS locally; required Ubuntu self-test added to Wheel/CLI job | PASS: run `32977924481` |
| R145-004 | P2-PROVENANCE | The report names only `_sqlite3` and `_decimal`, not the complete enumerated native-extension set loaded by the generator. | Python execution process → native extension/runtime closure → report claim | REPRODUCED: the generator process currently loads additional `_hashlib`, `_json`, `_posixsubprocess`, `math`, `select`, and other native modules. | repository-actionable | Enumerate loaded extension modules deterministically and downgrade closure wording to the exact measured scope unless sealed image identity exists. | Synthetic/real loaded extension sets are sorted, unique, hashed, closure-inspected when applicable, and represented in the report digest. | CLOSED: loaded file-backed extensions are sorted, hashed and closure-bound; claim is scoped to enumerated roots | PASS: generator source binding, environment self-test and report v4 validation | PASS: run `32977924481` |
| R145-005 | P2-GOVERNANCE | Runtime contract constants duplicate registry strings and lifecycle probes prove only AST presence, not executable family reachability. | registry → production constants → construct/round-trip/scientific/operational lifecycle | REPRODUCED: FSO/FSOI/verification constants are hand-written outside the registry and lifecycle checks only inspect named AST methods. | repository-actionable | Add typed `current_contract()` lookup, derive current production constants from it, and execute explicit family lifecycle cases rather than checking test-name strings alone. | Current verification/FSO/FSOI/replay constants equal registry lookups; lifecycle cases actually run; empty operational sets reject action. | CLOSED: runtime current constants derive from registry; file-based executable lifecycle probes | PASS under `python -I -m pytest`; deployment acceptance remains empty | PASS: run `32977924481` |
| R145-006 | P2-GOVERNANCE | `PR144_CHECKLIST.md` still records pre-merge PR/CI state after the exact tree merged and main push CI passed. | repository audit record → GitHub authority state | REPRODUCED against GitHub: PR #144 is merged at `cc6a703…`; PR-head and main-push checks are all successful. | repository-actionable | Add an append-only post-merge authority section without rewriting prior scientific dispositions. | Checklist records head, merge commit, identical tree, PR-head/main-push CI success, external HOLD, and operational NO-GO separately. | CLOSED: append-only PR #144 authority record added | PASS: head/merge/tree/PR-head CI/main-push CI and scientific HOLD are separate | PASS: run `32977924481` |

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
- [x] Implementation head `e244d85060e0941d7d82fc3029cf0cd0be85e0a3` equals the pushed commit and required CI run `32977924481` is terminal green.
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
| PR #145 merge | HOLD | Implementation, local verification, five adversarial reviews and exact-head CI are complete; explicit merge authorization remains required. |

## Post-CI independent adversarial review — reopened closure cycle

The implementation-head review above remains an immutable record of what was
known at `e244d850...`.  A subsequent independent review of the exact PR tree
identified six additional repository-actionable numerical and replay-boundary
findings.  They reopen the PR closure cycle without changing the external
science or deployment gates.

### Follow-up deduplication and non-regression ledger

| Finding ID | Semantic fingerprint (`boundary + invariant + acceptance intent`) | Prior finding/status | Current disposition | Existing guard to preserve |
|---|---|---|---|---|
| R145-007 | durable replay manifest generation → historical tensor-role/file-set identity → v19–v24 remain byte-auditable after current roles grow | R145-005 closed only for current registry reachability; new historical-generation counterexample | new repository P1 | current v25 requires all three detection-interval tensors; legacy semantic action remains prohibited |
| R145-008 | source range/elevation → detection-limit arithmetic → lower/upper tensors enclose the exact category threshold | R145-001 closed at nominal examples; compositional-rounding counterexample | refined repository P1 | detected uses upper; clear/censored use lower; ambiguous cells receive zero source score |
| R145-009 | source evidence → score interval → strict dominance cannot be created by intermediate float32 rounding | R145-002/S145-003 closed only at endpoint rounding; compositional-score counterexample | refined repository P1 | unique strict lower-over-all-upper winner; overlap remains unassigned |
| R145-010 | zero projected distance/speed → outward interval → exact origin remains in a zero-radius footprint | R145-002 closed at positive radii; new zero-boundary counterexample | refined repository P1 | positive-distance outward enclosure and footprint partition |
| R145-011 | physical radius helper → projected boundary → helper and interval decisions agree at exact endpoints | R145-002 closed at sampled ULP case; new helper contradiction | refined repository P1 | PSR/FSS uncertain-annulus fail-close |
| R145-012 | scalar area/spatial-age arithmetic → directed physical bound → threshold decisions never use a narrowed interval | R145-002 closed for distance/speed; new adjacent directed-rounding gaps | refined repository P2 | existing area fail-close and spatial-age mask semantics |

### Follow-up adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R145-007 | P1-HIGH | Legacy v19–v22 manifests use the live current tensor-role set, and v23/v24 lack typed durable-load generations and correct file-set routing. | historical manifest → decoder → directory members/shards → audit load | REPRODUCED: an origin/main-format role set is rejected by every v19–v22 constructor; v24 has no exact decoder/file-set branch. | repository-actionable | Freeze the historical v19–v24 role set, add v23/v24 audit manifest types, and route their sharded raw+verification directories explicitly. | A PR #144-format v24 bundle directory loads as audit-only; v19–v22 accept only their frozen role set; current-only roles remain forbidden. | CLOSED: frozen v19 role set, typed v23/v24 audit manifests, exact generation/file routing | PASS: six-generation role matrix, current-role rejection, real sharded v24 directory cold-load, executable lifecycle probe | PENDING new exact-head CI |
| R145-008 | P1-SCIENTIFIC | One final `nextafter` does not enclose intermediate rounding in the distance-dependent detection-limit polynomial. | projected range interval → range/elevation terms → category certification | REPRODUCED by the supplied float32 coefficients: the next float above the stored upper can still be below the exact-real threshold. | repository-actionable | Compute the monotone polynomial with compositional directed bounds in canonical float64 and outward-cast to the source dtype. | Exact rational/binary-float oracle is enclosed for the supplied counterexample in float32 and representative float64 cases. | CLOSED: canonical float64 compositional range-square/coefficient/elevation/sum bounds with outward dtype cast | PASS: exact `Fraction` oracle and category/source replay regression | PENDING new exact-head CI |
| R145-009 | P1-SCIENTIFIC | Final endpoint widening of a float32 score cannot prevent a false strict-dominance winner created by rounded intermediate terms. | source evidence → score interval → strict-dominance selection | REPRODUCED by the supplied A/B pair: code selects A although exact score intervals overlap. | repository-actionable | Compute score bounds compositionally in canonical float64, widen every primitive (including temporal quality), and certify only non-overlapping intervals. | Supplied A/B pair remains unassigned in float32; a clearly separated pair selects deterministically in float32/float64. | CLOSED: canonical float64 score interval, directed primitive bounds and strict interval dominance | PASS: supplied float32 overlap remains unassigned; separated source behavior preserved | PENDING new exact-head CI |
| R145-010 | P1-FUNCTIONAL | Unconditional upward rounding maps exact zero distance to `5e-324`, excluding the origin from a zero-radius footprint. | zero distance/speed → interval → footprint consumers | REPRODUCED: `projected_ground_distance_interval(0, 0.006)` returns `[0, 5e-324]`. | repository-actionable | Preserve exact `[0,0]` for zero projected distance/speed while retaining outward rounding for positive values. | Public zero interval and current-grid zero-radius footprint include exactly the origin. | CLOSED: distance and speed preserve exact zero intervals | PASS: zero public intervals and zero-radius footprint contain only the origin | PENDING new exact-head CI |
| R145-011 | P1-NUMERICAL | Projected-radius helper endpoints use nearest rounding and contradict the authoritative interval decision at 994/1006 m. | metric evidence helper → PSR/search radius → physical threshold | REPRODUCED: helper `inside=994` has ground upper greater than 1000; helper `outside=1006` has ground lower below 1000. | repository-actionable | Round the certain-inside projected radius inward and the certainly-exceeds radius outward. | Helper outputs classify consistently through `projected_ground_distance_interval()` at both boundaries. | CLOSED: certain radius rounds inward and exceed radius rounds outward | PASS: both helper endpoints agree with authoritative interval decisions | PENDING new exact-head CI |
| R145-012 | P2-NUMERICAL | Area interval and spatial-age lower-bound arithmetic still use nearest-rounded intermediate denominators/products. | projected area/grid spacing → physical interval/age gate | REPRODUCED analytically for the supplied area and 1000 m spacing examples. | repository-actionable | Apply directed denominator, division, multiplication and final cast rules to area and spatial-age calculations. | Exact rational oracle lies inside area bounds; spatial maximum age is never above the exact lower-bound value. | CLOSED: directed area denominator/division and scalar spatial-age lower arithmetic | PASS: exact rational area enclosure and conservative spatial-age oracle | PENDING new exact-head CI |

### Follow-up acceptance summary

- [x] Every new review statement is represented once with a stable fingerprint.
- [x] R145-007/R145-010/R145-011 are directly reproduced on PR head `7630ff935188a7336a2e45e1675d80a3247a3b47`; R145-008/R145-009/R145-012 are reproduced by the supplied exact counterexamples and current arithmetic trace.
- [x] Every repository-actionable follow-up P1/P2 is fixed.
- [x] Every follow-up fix has an adversarial regression.
- [x] Focused and adjacent suites pass; a stable broad run reached `438 passed`, `247 subtests passed`, failure-free before manual interruption at 35 minutes. Full Python 3.10/3.12 CPU completion is delegated to exact-head CI.
- [x] basedpyright reports `0 errors`; geodetic source binding, compileall, diff check, sdist and wheel build pass.
- [ ] The pushed PR head and exact-head CI are terminal green.
- [x] External X145-001/X145-002/X145-003 and operational NO-GO remain unchanged.
- [x] Merge remains HOLD.
