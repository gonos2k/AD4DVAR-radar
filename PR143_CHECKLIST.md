# PR #143 metric-distance uncertainty closure checklist

## Authority snapshot

- Cycle authority key: `ADVAR-R143-cce64f8-5f0c5f2-2026-08-25`
- Predecessor checklist/PR/merge commit: `PR142_CHECKLIST.md` / PR #142 / `cce64f8b902755e533d79d96b8fc1903ceb2463b`
- Review source: user-provided post-merge review of PR #142
- Review timestamp/time zone: 2026-08-25 / Asia-Tokyo
- Reviewer-stated base/head: `main@cce64f8`, PR #142 head `c2e02aa`
- Verified repository base: `origin/main@cce64f8b902755e533d79d96b8fc1903ceb2463b`
- Verified predecessor PR/head/tree: PR #142 `MERGED`; head `c2e02aa0fb2bf496fd98364e6a3a1f91f3229fba`; merge/head tree `5f0c5f2e2af00cf7f855bff561c2391bfc992b00`; head is reachable from `origin/main`
- Worktree state: fresh branch `agent/pr143-metric-distance-boundaries`; user-owned untracked `.DS_Store`, `.omx/`, `graphify-out/`, `uv.lock`, and local HWPX documents preserved and excluded
- CI snapshot: PR #142 exact-head run `32715472963`; Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI smoke all `SUCCESS`
- Fresh branch start: `origin/main@cce64f8b902755e533d79d96b8fc1903ceb2463b`
- Project boundary: reproducible offline scientific validation; operational deployment is explicitly unsupported and out of scope

## Deduplication and non-regression ledger

| Finding ID | Semantic fingerprint (`boundary + invariant + acceptance intent`) | Prior finding/status | Current disposition | Existing guard to preserve |
|---|---|---|---|---|
| R143-001 | metric-domain linear scale evidence → projected distance/speed/radius → discrete scientific eligibility; uncertainty-crossing thresholds must fail closed | refinement of R142-001/R142-003 sampled scale and area interval closure | new repository-actionable P1 | area maximum/minimum interval rules, affine/shear footprints, and nominal exploratory calculations |
| R143-002 | current scientific evidence → operational selector reachability; unsupported behavior must be explicit rather than encoded as an impossible version predicate | refinement of R142-025 | new repository-actionable P1 | current v32 never authorizes deployment; v31 remains audit-readable only |
| R143-003 | geodetic generator bytes → dynamic runtime closure/locale → reproducible report | refinement of R142-008/R142-017/R142-022 | new repository-actionable P2 | exact generator source, PROJ/projinfo/proj.db hashes, poisoned PROJ data rejection, and byte-identical report |
| R143-004 | machine-readable generation capability registry → issuers/decoders/docs/tests; current/audit/unsupported sets must be consistent and reachable | refinement of R142-004/R142-015/R142-018/R142-024/R142-025 | new repository-actionable P2 | exact-type current validators and all historical audit decoders |
| X143-001 | governed study domain → polygon authority → publication claim | X142-001 OPEN | external carry-forward | repository continues to claim bbox membership only |
| X143-002 | independent geodetic engine/database → scale-error revalidation → ground-distance certification | X142-002 OPEN | external carry-forward | repository PROJ report remains sampled internal evidence only |
| X143-003 | independent real-radar cohort → single/multi-radar skill generalization → publication | X142-003 OPEN | external carry-forward | synthetic/offline results are not promoted to external skill claims |

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R143-001 | P1-SCIENTIFIC | The registered `0.006` linear scale budget is report-only for distance, speed, and radius decisions; nominal projected comparisons can change discrete masks and eligibility at uncertainty-crossing boundaries. | metric-domain evidence → distance/speed interval → footprint, PSR, motion and posterior gates | REPRODUCED: evidence policy is `registered-linear-scale-budget-report-only-v1`; radius and speed consumers compare projected norms directly. | repository-actionable | Add one typed projected-ground distance/speed interval API and conservative maximum/minimum relations; bind the fail-close policy to the current scientific contract and apply it to all physical radius/speed consumers. | At 1 km and epsilon 0.006: `993.999` is certainly within, `1000` uncertain, `1006.001` certainly exceeds; footprint, sidelobe mask, causal/amplitude support, state speed, pair disagreement, saturation and posterior paths use the same policy. | CLOSED | PASS | PENDING |
| R143-002 | P1-ARCHITECTURE | `NeuralPriorPromotionEvidence` constructs only v32 while the operational selector requires exact current type plus v31, so its success/fallback code is unreachable and six positive tests are skipped. | scientific promotion → operational selection API | REPRODUCED: the constructible and accepted contract sets are disjoint. | repository-actionable | Replace the impossible predicate with a dedicated unsupported exception and a public/current fail-closed entry point; remove unreachable selector implementation from current code, retain v31 audit-only decoding, and replace skips with explicit unsupported-boundary coverage. | Current v32 always raises `OperationalDeploymentUnsupportedError`; no current positive deployment test is skipped; v31 audit artifact cannot authorize; no public API suggests operational activation. | CLOSED | PASS | PENDING |
| R143-003 | P2-PROVENANCE | PROJ binaries/database and generator source are fixed, but loaded dynamic libraries, loader overrides, locale, platform/ABI/libc and SQLite runtime are not recorded or constrained. | geodetic toolchain/runtime → factor report bytes | REPRODUCED: generator clears only PROJ data overrides and records executable/database identities, not the dynamic execution closure or locale/platform identities. | repository-actionable within host-portable limits; independent engine validation remains X143-002 | Force canonical locale/timezone, reject loader overrides, record platform/architecture/libc/SQLite and actual executable dynamic-library closures with hashes; make source-only validation bind the new fields. | Locale/loader override injection fails; regenerated report records exact closure and remains byte-identical on the sealed host; runtime rejects field/hash tampering. | CLOSED | PASS | PENDING |
| R143-004 | P2-GOVERNANCE | Contract generations and capabilities are manually duplicated across issuers, decoders, tests, README and checklists, allowing drift and unreachable accepted combinations. | contract registry → construction/decoding/scientific/operational capability | REPRODUCED by distributed literals and the v32/v31 impossible operational predicate. | repository-actionable | Add a small authoritative registry for affected scientific/promotion contracts and a validator that checks issuable, audit-readable, scientific-eligible and operationally-accepted sets plus reachability; use it in code/tests/docs audit without introducing a general framework. | Current entries match issuers/decoders; every accepted combination is constructible; operational accepted set is explicitly empty; predecessor audit generations remain readable; README generation audit consumes the registry. | CLOSED | PASS | PENDING |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S143-001 | PR #142 binds exact EPSG:5179/PROJ/projinfo/proj.db/generator identities to a deterministic sampled report. | metric-domain evidence v1 and required source-only CI | New runtime closure must extend rather than weaken existing hashes and poisoned-PROJ-data checks. |
| S143-002 | Area thresholds use conservative projected-to-ground intervals and fail closed when the interval crosses a minimum or maximum. | `projected_area_interval_km2()` and centralized threshold validators | Distance work must preserve the exact area mathematics and strict boundary behavior. |
| S143-003 | Current grid/run/replay artifacts forward-bind metric evidence while historical generations remain audit-only. | grid v5, forecast v69, replay v23 and predecessor audit types | Any generation bump must keep predecessor byte decoding and reject predecessor semantic use. |
| S143-004 | PR #142 exact tree passed Python 3.10/3.12 CPU and Wheel/CLI required checks. | run `32715472963` | New cycle begins as a forward scientific hardening, not a rollback. |

## External actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X143-001 | scientific governance owner | Supply the approved study polygon, authority/source digest, boundary convention and versioned legal/scientific scope. | Current repository intentionally validates only sampled bbox coverage. | polygon-based confirmatory and publication claims | OPEN |
| X143-002 | independent geodesy reviewer | Recompute scale bounds with an independent engine/database and sign or publish the reproducible report. | Current evidence declares independent revalidation required. | certified ground-distance/area and external publication claims | OPEN |
| X143-003 | radar-science owner | Supply preregistered independent single/multi-radar real-case cohorts and target provenance. | Current repository evidence is synthetic/bounded offline. | generalized skill and external publication claims | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every finding is fingerprinted and mapped against PR #142 findings and guards.
- [x] This post-merge cycle starts on a fresh branch exactly at current `origin/main`.
- [x] External polygon, independent-geodesy and real-radar items remain visible and are not reimplemented.
- [x] Every repository-actionable P1/P2 is fixed or evidence-disproved.
- [x] Every fix has a targeted regression test.
- [x] Affected contract generations and committed evidence are synchronized.
- [x] Adjacent and broad CPU suites pass locally; exact Linux 3.10/3.12 remains the PR CI gate.
- [ ] PR head equals the reported pushed commit and required CI is terminal green.
- [x] Research, deployment and publication decisions remain separate.
- [x] Merge remains HOLD unless explicitly authorized.

## Current decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Registered-bbox offline exploratory research | GO WITH EXPLICIT ASSUMPTION | Sampled internal geodetic evidence and existing area fail-close guards. |
| Nominal projected-distance exploratory diagnostics | GO WITH INTERVAL DISCLOSURE | Evidence v2 preserves the nominal value and the registered ground interval. |
| Distance/speed-threshold confirmatory result | GO WITH INTERNAL SAMPLED EVIDENCE | Current consumers fail closed at interval-crossing boundaries; independent certification remains X143-002 HOLD. |
| Study-polygon claim | HOLD | X143-001. |
| Independent ground-distance certification | HOLD | X143-002. |
| Multi-radar generalized skill claim | HOLD | X143-003. |
| Operational deployment | NO-GO / EXPLICITLY UNSUPPORTED | Scientific v32 is not an operational authorization contract. |
| External publication | HOLD | X143-001/X143-002/X143-003 and current P1 closure. |
| PR #143 merge | HOLD | Implementation, independent review and exact-head CI required. |

## Local verification evidence

| Check | Result |
|---|---|
| Promotion regression before unreachable-code deletion | `226 tests`, `3142.096s`, `OK`, no skip summary |
| Post-deletion operational boundary | 4 targeted promotion tests `OK`; registry tests 3 `OK` |
| Nowcast suite | 162 tests `OK`, plus new current-boundary regression `OK` |
| Variational suite | 141 tests `OK`, plus new causal/amplitude boundary regression `OK` |
| Sensitivity suite | 85 tests `OK`, plus new soft-FSS boundary regression `OK` |
| Current geodetic report | full sealed-host `--check` reproduced SHA-256 `5ace1126287511bc6742d989b283b8b3194767d307294e11483d0747ac5f6e60` |
| Generator source binding | `--check-source-only` PASS |
| Loader override attack | nonempty `LD_LIBRARY_PATH` rejected before report generation |
| PROJ closure audit | every file-backed closure hash, closure digest and execution-environment digest independently recomputed |
| Product type check | official `check_basedpyright.py`: 0 errors |
| Package build | sdist and wheel `0.108.0` built successfully |
| Wheel-origin CLI smoke | wheel installed to an isolated target; `nowcast-npz-v75`, `forecast-run-v69`, 18×16×16 output and validity contract verified |

## Five-pass adversarial review record

1. **Scientific callsites:** pair dilation, PSR sidelobe mask, causal support, amplitude tolerance, soft FSS, state speed, pair disagreement, motion saturation and posterior dynamics all resolve through the current interval policy. Direct current-grid boundary tests prevent nominal fallback.
2. **Generation/history:** evidence v2 contains the distance policy, scale budget and execution closure; current grid digests forward-bind its digest; v1 is raw-byte audit-only and cannot become current scientific evidence.
3. **Operational reachability:** the public selector always raises `OperationalDeploymentUnsupportedError`; unreachable selector/infer implementations and six hidden legacy-positive scenarios were removed; the operational accepted registry set is explicitly empty.
4. **PROJ runtime closure:** canonical locale/timezone/network settings, loader override rejection, platform/ABI/libc/SQLite identities, ordered dynamic closure and content hashes were independently checked. Independent-engine certification remains X143-002.
5. **Artifact/governance:** registry-rendered README table, prose generation, package version, report/generator hashes, user-owned untracked files and diff whitespace were checked. Exact Linux PR CI remains pending and merge remains HOLD.

The callsite sweep is limited to current grid-bound nowcast/FSO scientific consumers addressed by R143-001. Historical physical-event catalog artifacts retain their separately declared `spatial_reference_digest`; they are not upgraded into independent ground-distance evidence by this PR and cannot release X143-001/X143-002/X143-003 publication holds.
