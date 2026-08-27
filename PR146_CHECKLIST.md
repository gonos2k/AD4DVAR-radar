# PR #146 directed affine and predecessor-audit closure checklist

## Authority snapshot

- Cycle authority key: `main@0b1ad558dbcf9b7a7f88b1d6eb0516127f387f9d` + user-provided post-merge review of PR #145 + `2026-08-27 Asia/Tokyo`
- Predecessor checklist/PR/merge commit: `PR145_CHECKLIST.md` / PR #145 / `0b1ad558dbcf9b7a7f88b1d6eb0516127f387f9d`
- Review source: user-provided additional review of current `main`
- Review timestamp/time zone: `2026-08-27 Asia/Tokyo`
- Reviewer-stated base/head: `main@0b1ad558dbcf9b7a7f88b1d6eb0516127f387f9d`; PR #145 head `d71986a4796c7a8167292790541e57652b3d02a2`
- Verified repository base: `origin/main@0b1ad558dbcf9b7a7f88b1d6eb0516127f387f9d`
- Verified predecessor PR/head/merge: PR #145 `MERGED`; head `d71986a4796c7a8167292790541e57652b3d02a2`; merge `0b1ad558dbcf9b7a7f88b1d6eb0516127f387f9d`; predecessor head is reachable from `origin/main`
- Worktree state: `/Users/yhlee/ADVAR`; branch `agent/pr146-directed-affine-audit-closure`; tracked tree clean at cycle start; pre-existing user untracked files inventoried and preserved
- CI snapshot: exact-head run `33040384491` Python 3.10 CPU, Python 3.12 CPU/type check, Wheel/CLI and Linux geodetic self-test all `SUCCESS`
- Fresh branch start (`origin/main` SHA): `0b1ad558dbcf9b7a7f88b1d6eb0516127f387f9d`

## Deduplication and non-regression ledger

| Finding ID | Semantic fingerprint (`boundary + invariant + acceptance intent`) | Prior finding/status | Current disposition | Existing guard to preserve |
|---|---|---|---|---|
| R146-001 | affine matrix/state offset → norm/determinant primitive interval → hard speed/radius/area gates cannot false-accept | R145-002/R145-012 closed only after a rounded projected scalar existed; new upstream primitive counterexamples | refined repository P1 | scalar directed ground intervals, hard maximum gates, zero preservation, set-valued footprint consumers |
| R146-002 | authoritative float64 observation geometry → evidence dtype → geometry/ranking cannot depend on reflectivity precision | R145-008/R145-009 closed for compositional arithmetic after geometry evidence; new upstream dtype counterexample | refined repository P1 | FP32/FP64 reflectivity support, category-directed threshold bounds, strict source-score dominance |
| R146-003 | frozen forecast-run v68 + nonempty deployment lineage v19 → legacy decode → v19-audit preservation and current-action rejection | R145 operational boundary closed only for current runs; immediate-predecessor audit mapping omitted | new repository P1 | current deployment lineage remains unsupported; historical digests remain immutable |
| R146-004 | frozen holdout-plan v33/v34 bytes → typed audit decoder → canonical digest preservation without current scientific action | predecessor generations were declared audit-readable; both issued v33 and v34 omitted | refined repository P1 | current v35 issuance/action remains generation-exact |
| R146-005 | source score upper bounds → strict competitor maximum → identical result in O(S·T·H·W) | R145-009 scientific result closed; performance structure unchanged | new repository P2 | exact strict dominance and tie/no-winner semantics |
| R146-006 | audit-readable generation registry → cold fixtures → every declared generation decodes, preserves digest, rejects forbidden action/tamper | R145-005 improved executable current lifecycle only | refined repository P2 | current lifecycle probes remain executable; audit-only generations cannot re-enter current action |
| R146-007 | package/checklist/top-level audit API → release authority → one coherent version and public/internal boundary | PR145 checklist remained pre-merge in follow-up section; exports are partial | new repository P2 | append-only historical record, external HOLD and operational NO-GO |
| X146-001 | approved study polygon + independent geodesy/environment + preregistered real-radar cohort → confirmatory/publication authority | X145-001/X145-002/X145-003 OPEN | external carry-forward | bounded bbox/offline assumption remains explicit |
| X146-002 | representative-tilt horizontal geometry → beam-path error envelope or interval model → long-range confirmatory clear/censor claim | explicit current model scope; not a hidden implementation defect | external/scientific scope | `geometry_model` and `radar_altitude_role=provenance_only` remain explicit |

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R146-001 | P1-NUMERICAL/SCIENTIFIC | Affine norm and determinant are nearest-rounded before interval enclosure, and FP32 state can downcast the float64 affine, allowing ULP-boundary false acceptance in speed, footprint and area gates. | authoritative affine/state → displacement/norm/determinant → ground interval → hard physical decision | **REPRODUCED then CLOSED**: float64 multiply/add/square/sqrt and determinant intervals now enclose every primitive; current motion, footprint, spacing and cell-count area gates consume those bounds; maximum index casts are inward. | repository-actionable | Add narrow float64 compositional interval helpers for affine displacement, integer offset norm and determinant/area; cast maximum search limits inward. | Reviewer motion, footprint and area exact-binary counterexamples fail closed; ordinary affine results remain stable. | **CLOSED** | **PASS** — `test_affine_hard_gates_enclose_each_authoritative_float_operation`; full nowcast suite | PENDING exact-head CI |
| R146-002 | P1-SCIENTIFIC | Observation range/elevation are downcast to reflectivity dtype before interval/ranking arithmetic, so FP32 dBZ can weaken authoritative float64 geometry. | grid/source registry → geometry evidence → detection/source eligibility and ranking | **REPRODUCED then CLOSED**: current evidence preserves projected/ground range lower+upper, elevation and score bounds as FP64 while selected reflectivity retains its source dtype. | repository-actionable | Keep range/elevation/score bounds in float64 independent of dBZ tensors; preserve range intervals in evidence and bump every changed semantic generation. | FP32 reflectivity retains authoritative 100000.003 m range and 20.0000005° elevation decisions; selected dBZ remains FP32. | **CLOSED** — current chain bumped through plan v15, verification v21, FSO v27, FSOI v23, replay v26, holdout v36 and run v71 | **PASS** — `test_fp32_reflectivity_does_not_downcast_observation_geometry`; full sensitivity suite | PENDING exact-head CI |
| R146-003 | P1-AUDIT | Legacy `forecast-run-v68` with historically valid nonempty lineage v19 is not mapped to `v19-audit`. | serialized forecast v68 → legacy loader → deployment lineage audit/current boundary | **REPRODUCED then CLOSED**: v68/v69/v70 now map raw v19 lineage to v19-audit while preserving canonical decision digests and forbidding current action. | repository-actionable | Map v68 and v69 lineage v19 to v19-audit and add frozen nonempty v68 cold fixture/tamper test. | Frozen v68 loads with all decision digests preserved as audit-only; tamper/current action is rejected. | **CLOSED** | **PASS** — expanded nonempty legacy deployment fixture in `test_v52_deployment_geometry_loads_as_audit_only`; full run-artifact suite | PENDING exact-head CI |
| R146-004 | P1-AUDIT | Issued holdout plans v33 and v34 have no typed audit wrapper or decoder branch. | canonical v33/v34 JSON/digest → holdout decoder → audit preservation/current action | **REPRODUCED then CLOSED**: v33/v34 plus immediate predecessor v35 now have raw canonical audit wrappers and exact decoder branches; current issuance is v36 only. | repository-actionable | Add minimal raw-payload/digest v33 and v34 audit types and exact decoder branches without current semantic construction. | Frozen v33/v34 canonical payloads decode, reproduce digest/nested fields and reject tamper/current scoring or promotion. | **CLOSED** | **PASS** — `test_holdout_plans_v33_through_v35_load_as_cold_audit_fixtures`; full ledger suite | PENDING exact-head CI |
| R146-005 | P2-PERFORMANCE | Strict source competitor calculation repeatedly concatenates all other sources, producing O(S²THW) traffic. | score upper tensor → competing upper → strict source dominance | **REPRODUCED then CLOSED**: top-2 competitor selection is O(S·T·H·W), including single-source and maximum-tie semantics. | repository-actionable | Replace quadratic loop with exact top-2 competitor calculation. | Random/tied/single-source top-2 output is byte-equal to a quadratic oracle; benchmark/complexity guard covers larger S. | **CLOSED** | **PASS** — `test_top2_source_dominance_matches_quadratic_oracle` | PENDING exact-head CI |
| R146-006 | P2-GOVERNANCE | Executable lifecycle probes cover one named current test per family but do not enumerate every audit-readable generation. | contract registry → historical fixture matrix → decode/digest/action/tamper lifecycle | **REPRODUCED then CLOSED for the registered scientific graph**: immutable per-generation audit probes are mandatory and forecast-run/holdout families are now registered; meta-tests execute every declared current and audit probe without skip. | repository-actionable | Add an explicit per-contract audit-generation probe registry and a common executable completeness test for relevant artifact families. | Every registered audit-readable contract has one executable probe covering canonical decode, digest, type, action policy and tamper rejection. | **CLOSED** | **PASS** — contract registry 8 tests / 12 subtests, including executable audit completeness | PENDING exact-head CI |
| R146-007 | P2-RELEASE/API | README still says v0.109 while package is 0.110.0, PR145 checklist lacks final merge authority, and top-level audit exports expose an inconsistent subset. | release metadata/audit record/public API → user authority | **REPRODUCED then CLOSED**: package is 0.111.0, README title is version-neutral and registry-generated table is current; PR145 has append-only merge authority; supported recent audit wrappers are consistently exported and tested. | repository-actionable | Remove the README title version, append PR145 post-merge authority, and consistently export all supported public audit wrappers. | Version check, post-merge record and import-surface test/documentation are synchronized. | **CLOSED** | **PASS** — README registry test, public audit export test, sdist/wheel build | PENDING exact-head CI |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S146-001 | Detection-limit category semantics now use upper for detected, lower for clear/censored, plus an uncertainty mask. | PR #145 current verification path and exact-oracle tests | Geometry precision fixes must not collapse the interval or remove explicit uncertainty. |
| S146-002 | Detection and source-score arithmetic is compositional in canonical float64 with outward casts. | PR #145 algorithm digests and source selection implementation | New affine primitive bounds feed this arithmetic rather than duplicate or weaken it. |
| S146-003 | Semantic replay v23/v24 now has typed audit decoders and exact sharded routing. | ledger audit classes and durable v24 cold-load regression | Immediate-predecessor audit fixes must preserve these generations. |
| S146-004 | Runtime current generations derive from an executable contract registry lifecycle. | `_contract_registry.py` and lifecycle meta-test | Extend historical coverage without returning to AST-only probes. |
| S146-005 | Current geometry model explicitly limits itself to projected-horizontal representative tilt with altitude provenance only. | registry/geometry contracts | Do not silently claim beam-path closure while external evidence remains absent. |

## External actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X146-001 | scientific governance, independent geodesy and radar-science owners | Supply approved polygon, independently sealed geodetic validation and preregistered single/multi-radar holdout cohorts. | Repository evidence is bounded bbox/internal geodesy/synthetic or limited offline evidence. | confirmatory publication, generalized multi-radar skill | OPEN |
| X146-002 | radar-science owner | Either preregister a beam-path interval geometry or validate representative-tilt error envelopes by radar/range/sweep/regime. | Current model intentionally omits beam height, slant range, refractivity and vertical blockage. | long-range clear/censor confirmatory interpretation | OPEN |

## Acceptance summary

- [x] Every pasted review statement is represented exactly once and fingerprinted.
- [x] This post-merge cycle starts on a fresh branch exactly at `origin/main`; PR #145 head is reachable through its merge commit.
- [x] Prior semantic replay and current lifecycle closures are preserved as strengths rather than duplicated findings.
- [x] Every current-tree claim is independently reproduced, disproved or classified.
- [x] Every repository-actionable P1/P2 is fixed or evidence-disproved.
- [x] Every fix has a targeted adversarial regression.
- [x] Affected contract generations, digests, fixtures, release metadata and public API are synchronized.
- [x] Full CPU inventory, static type, package build and local geodetic self-test pass; installed-wheel/CLI smoke awaits exact-head CI.
- [ ] PR head equals the reported pushed commit and required exact-head CI is terminal green.
- [x] External polygon/geodesy/cohort and beam-geometry actions remain visible.
- [x] Merge remains HOLD unless explicitly authorized.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Internal offline exploratory research | GO WITH EXPLICIT ASSUMPTIONS | R146 repository findings are code-closed locally; external geometry/geodesy assumptions remain. |
| Confirmatory verification/FSO/FSOI | HOLD | Repository hard-boundary fixes are locally green, but X146-001/X146-002 remain external gates and exact-head CI is pending. |
| Publication-suppressed shadow | HOLD | External scientific evidence and exact-head CI remain open. |
| Canary | NO-GO / OUT OF SCOPE | Operational capability remains intentionally empty. |
| State-advancing LIVE | NO-GO / OUT OF SCOPE | Operational capability remains intentionally empty. |
| External publication | HOLD | External polygon/geodesy/beam/cohort evidence remains open. |
| PR #146 merge | HOLD | PR creation, pushed-head identity and required exact-head CI are pending; merge requires explicit authorization. |

## Local implementation evidence

- `python -m compileall -q src tests`: **PASS**
- `check_basedpyright.py`: **PASS**, 0 errors
- `tests/test_nowcast.py tests/test_sensitivity.py`: **254 passed, 275 subtests passed**
- `tests/test_ledger.py tests/test_run_artifact.py tests/test_cli.py`: **119 passed, 92 subtests passed**
- `tests/test_contract_registry.py`: **8 passed, 12 subtests passed**
- `tests/test_promotion.py` (terminal chunks 1–40, 41–80, 81–120, 121–160, 161–200, 201–227): **227 passed, 26 subtests passed**
- Remaining acceptance/calibration/deployment/ensemble/matrix/metric/PCG/runtime/variational modules: **219 passed, 89 subtests passed**
- Full collected CPU inventory, exact module-sum: **827 passed, 494 subtests passed**
- Focused promotion regressions: **PASS**, including full semantic replay and holdout common-domain lifecycle
- sdist/wheel build for package `0.111.0`: **PASS**
- geodetic source binding and execution-environment self-test with loader override removed: **PASS**
- `git diff --check`: **PASS**
