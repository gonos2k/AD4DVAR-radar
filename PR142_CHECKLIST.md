# PR #142 geodetic metric-evidence closure checklist

## Authority snapshot

- Cycle authority key: `ADVAR-R142-e4e5dc7-b4a834b-2026-08-24`
- Predecessor checklist/PR/merge commit: `PR141_CHECKLIST.md` / PR #141 / `e4e5dc7c3f8c9a709f62cb2a3025d3849c4dc5b3`
- Review source: user-provided additional review of merged PR #141
- Review timestamp/time zone: 2026-08-24 / Asia-Tokyo
- Reviewer-stated base/head: `main@e4e5dc7`, PR #141 head `8a9185d65ba461ec77f8765ef4d9747e65e8a06b`
- Verified repository base: `origin/main@e4e5dc7c3f8c9a709f62cb2a3025d3849c4dc5b3`
- Verified PR/head/tree: PR #141 `MERGED`; reviewed head is reachable from `origin/main`; base tree `b4a834bf03040e071aa53d315f3dffd385dcc7bc`
- Worktree state: fresh branch `agent/pr142-metric-domain-evidence`; user-owned untracked `.omx/` preserved
- CI snapshot: exact-head run `32639810887`; Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI smoke all `SUCCESS`
- Fresh branch start: `origin/main@e4e5dc7c3f8c9a709f62cb2a3025d3849c4dc5b3`
- Project boundary: reproducible offline scientific validation; operational deployment is out of scope

## Deduplication and non-regression ledger

| Finding ID | Semantic fingerprint (`boundary + invariant + acceptance intent`) | Prior finding/status | Current disposition | Existing guard to preserve |
|---|---|---|---|---|
| R142-001 | metric-domain assumption → geodetic evidence identity → scale-budget claim; evidence must be reproducible and content-addressed | X141-002 / external evidence OPEN | refined repository evidence contract plus external independent validation | EPSG:5179 current-only, registered bbox inclusion, exact metric-domain digest |
| R142-002 | domain label → actual membership geometry; bbox must not claim polygon semantics | X141-001 / external legal study polygon OPEN | refined repository naming and optional polygon boundary | all grid cell centers and radar sites remain fail-closed outside the registered domain |
| R142-003 | linear projection error → area thresholds; near-boundary area decisions must not silently claim confirmatory certainty | new | new repository scientific guard | shear-aware FSS and physical cell-area calculations remain intact |
| R142-004 | scientific generation ledger → historical current transition; released versus reserved generations must be explicit | PR141 governance record / incomplete | new documentation correction | current FSOI remains exactly mapped to current FSO and verification generation |
| R142-005 | merged PR authority → post-merge evidence; checklist must match GitHub state | repeated governance pattern, prior R141-004 | new post-merge record | scientific HOLD gates remain separate from PR merge status |
| R142-006 | geodetic evidence → current grid/run/replay; scientific decisions must forward-bind the exact evidence digest | adversarial review of initial PR #142 implementation | new repository provenance correction | historical grid/evidence digests remain audit-loadable and unchanged |
| R142-007 | area-error interval → minimum evidence thresholds; lower-bound gates must fail closed near the threshold | refinement of R142-003 | new repository scientific correction | maximum-area interval policy and shear-aware area calculations remain intact |
| R142-008 | deterministic geodetic generator → one PROJ/EPSG authority; binary/database mismatch and poisoned environment must fail closed | adversarial review of initial PR #142 implementation | new repository provenance hardening | committed report remains byte-reproducible and content-addressed |
| R142-009 | shipped report bytes → typed evidence; sample digests and extrema must be recomputed rather than trusted | adversarial review of initial PR #142 implementation | new repository integrity hardening | report SHA and registered error budget remain exact |
| R142-010 | historical bbox payload → current API; canonical JSON must have an explicit lossless decoder | adversarial review of initial PR #142 implementation | new repository contract correction | historical `allowed_domain_polygon_digest` key and metric-domain digest remain stable |

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R142-001 | P1-SCIENTIFIC | The `0.006` linear scale-error budget is a fixed assertion; no sampled factors, engine/EPSG version, or report digest is preserved. | metric-domain contract → ground-distance interpretation → scientific claim | REPRODUCED: v1 validates the constant and bbox only; no factor report or geodetic-engine/database identity existed. | repository-actionable plus external independent-validation carry-forward | Add a typed, content-addressed geodetic evidence artifact and deterministic report; bind its digest to the current metric domain. | Evidence records engine/database identity, exact sample lattice, meridional/parallel scale digests, observed maximum, and report digest; tampering or budget exceedance fails. | ✅ evidence v1 + deterministic 17×17 PROJ report | ✅ targeted + regeneration PASS | ☐ PR/CI |
| R142-002 | P1-PROVENANCE | `allowed_domain_polygon_digest` is an axis-aligned bbox digest and membership is not point-in-polygon. | domain source/identity → grid/radar membership | REPRODUCED: the preimage and validator contain four bbox bounds and range comparisons only. | repository-actionable naming closure; governed study polygon remains X142-001 | Rename the current API/evidence to bbox semantics, preserve the historical serialized key for digest stability, and do not claim polygon authority. | Current API/evidence expose `allowed_domain_bbox_digest`; the historical payload remains byte-stable; polygon-based claims remain fail-closed as external HOLD. | ✅ exact bbox API/evidence; no invented polygon | ✅ targeted provenance PASS | ☐ PR/CI |
| R142-003 | P2-SCIENTIFIC | The linear scale-error budget is not propagated to area-sensitive thresholds or near-boundary decisions. | metric-domain evidence → cell area → removed/perturbed area guardrails | REPRODUCED: physical-area caps compared projected area directly with no uncertainty interval. | repository-actionable | Store the area-scale budget and require interval-stable area threshold decisions for current confirmatory learning guardrails. | Clearly below/above thresholds retain results; projected area whose uncertainty interval crosses the threshold fails closed; digest binds the margin policy. | ✅ area budget + centralized interval policy | ✅ pass/uncertain/exceeds + adjacent suites PASS | ☐ PR/CI |
| R142-004 | P2-GOVERNANCE | PR #141 records FSOI `v18 -> v19`, but the actual released current transition was `v17 -> v19`; v18 was reserved/unreleased. | generation ledger → durable audit interpretation | REPRODUCED from PR #140 source history and PR #141 checklist. | repository-actionable | Correct the transition and explicitly mark v18 reserved/unreleased with digest-version rationale. | Checklist matches code and predecessor history without implying a missing released artifact. | ✅ v17→v19; v18 reserved/unreleased | ✅ documentation review PASS | ☐ PR/CI |
| R142-005 | P2-GOVERNANCE | PR #141 checklist still records pre-merge HOLD/pending state after merge and successful exact-head CI. | post-merge scientific audit record | REPRODUCED from committed checklist and GitHub authority metadata. | repository-actionable | Record exact head, merge commit/time, run, job conclusions, and merged status while retaining scientific HOLDs. | Documentation matches GitHub metadata and does not claim confirmatory/publication/deployment GO. | ✅ post-merge authority synchronized | ✅ documentation review PASS | ☐ PR/CI |
| R142-006 | P1-HIGH | The new evidence reverse-binds the domain, but current grid/run/durable intervention artifacts do not preserve the evidence digest; durable replay falls back to a scalar projected-area comparison. | metric-domain evidence → grid/run/action replay → scientific decision | REPRODUCED: current grid payload had only `metric_domain_digest`, and durable replay called the scalar `cell_area_m2` branch. | repository-actionable | Forward-bind the current evidence digest in newly issued current grids/runs and durable action manifests; replay the same interval policy while preserving historical audit semantics. | Current artifacts reject a missing/wrong evidence digest; legacy artifacts remain audit-loadable; durable replay and in-memory safety use the same interval policy. | ✅ current grid/run + forecast v69 + durable action v6 | ✅ missing/wrong binding and cold replay tamper PASS | ☐ PR/CI |
| R142-007 | P1-SCIENTIFIC | Minimum-area evidence gates still compare projected area directly, so an uncertainty-crossing lower threshold can pass. | projected area → minimum scientific support → growth evidence | REPRODUCED: `minimum_growth_overlap_area_km2` used direct `>=` while maximum caps used the interval policy. | repository-actionable for current offline research; external publication gates remain HOLD | Add the dual minimum-area interval rule and apply it to current growth evidence. | Lower ground-area bound must meet the minimum; an interval crossing the minimum fails closed; clearly sufficient/insufficient cases are stable. | ✅ dual minimum-area policy + growth integration | ✅ insufficient/uncertain/passes and 4 km² integration PASS | ☐ PR/CI |
| R142-008 | P2-PROVENANCE | The generator can combine a manual projection definition with an unrelated `projinfo` database and does not clear `PROJ_DATA`. | source CRS authority → factor samples → report | REPRODUCED by static generator inspection. | repository-actionable | Derive the projection definition from the same `projinfo` database, require colocated/version-consistent tools, and clear both PROJ data environment variables. | Mismatched tool/database identity fails; deterministic regeneration remains byte-identical. | ✅ one colocated/version-matched PROJ toolchain | ✅ poisoned `PROJ_LIB`/`PROJ_DATA` regeneration byte-identical | ☐ PR/CI |
| R142-009 | P2-INTEGRITY | Runtime trusts report-provided sample digests and maxima after only checking the outer file SHA. | shipped report bytes → typed evidence semantics | REPRODUCED: per-sample scale values were not independently reduced at load time. | repository-actionable | Recompute lattice ordering, scale digests, and maxima from the report and compare them to all typed evidence fields. | Mutating one sample with a self-consistent outer report object is rejected; committed report passes. | ✅ runtime lattice/reduction replay | ✅ one-sample adversarial mutation rejected | ☐ PR/CI |
| R142-010 | P2-CONTRACT | `RadarMetricDomainContract(**contract.payload)` fails because the stable payload uses the historical polygon-named key while the current constructor uses bbox naming. | canonical payload → typed contract reconstruction | REPRODUCED with the current object. | repository-actionable | Add an explicit canonical-payload decoder that maps the historical key without changing the payload or digest. | `from_payload(contract.payload)` round-trips exactly; unknown/ambiguous keys fail. | ✅ explicit `from_payload()` decoder | ✅ exact round-trip and ambiguous-key rejection PASS | ☐ PR/CI |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S142-001 | PR #141 correctly binds current grid cell centers and radar sites to one metric-domain digest. | metric-domain v1, grid v5, registry v6 | Domain evidence work must preserve exact coordinate inclusion and reject origin `(0,0)`. |
| S142-002 | Scientific affine spacing and determinant reconstruction fail closed. | 1 m–100 km limits and representability tests | Underflow/overflow/zero-area adversarial tests remain green. |
| S142-003 | Shear-aware FSS remains supported while metre-based learning tiles fail closed on shear. | current sensitivity tests | New area margins must not disable valid projected-distance FSS. |
| S142-004 | PR #141 merged without a new P0 or rollback condition. | exact-head required CI success | This cycle is a forward scientific evidence improvement, not a rollback. |

## External scientific actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X142-001 | Scientific investigators | Approve and publish the legally/scientifically usable study polygon and source authority. | Signed or otherwise governed canonical polygon artifact and cohort protocol. | polygon-based confirmatory claim / external publication | OPEN |
| X142-002 | Independent geodetic reviewer | Recompute the committed EPSG:5179 scale report with an independent engine/database and accept the distance/area error budgets. | Independent report digest and comparison against repository evidence. | geodetically validated ground-distance claim / external publication | OPEN |
| X142-003 | Scientific investigators | Supply independent single-/multi-radar real-case cohorts. | Preregistered native observations and independent evaluation report. | multi-radar confirmatory claim / external publication | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every finding is fingerprinted and mapped against prior checklists, tests, contracts, and Git history.
- [x] This post-merge cycle starts on a fresh branch exactly at current `origin/main`.
- [x] Repeated external findings are refined and carried forward rather than hidden.
- [x] No accepted invariant or regression guard was removed or weakened.
- [x] Every repository-actionable P1/P2 is fixed or evidence-disproved.
- [x] Every fix has a targeted regression test or deterministic document check.
- [x] Adjacent scientific CPU suites pass after adversarial-review corrections (`503 passed`).
- [x] Existing serialized scientific digest preimages remain stable; current scientific artifacts also forward-bind the evidence digest.
- [x] PR #141 post-merge evidence is synchronized.
- [ ] PR #142 head equals the reported pushed commit.
- [ ] Exact-head CI results are classified.
- [x] Research, confirmatory claims, publication, and deployment remain separate decisions.
- [x] External scientific evidence items remain visible with owner and required action.
- [x] Merge remains HOLD unless explicitly authorized.

## Current decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Registered-bbox offline research | GO WITH EXPLICIT ASSUMPTION | PR #141 coordinate and affine guards remain valid. |
| Repository geodetic evidence claim | GO / SAMPLED EVIDENCE | R142-001 is closed with a deterministic PROJ 9.7.1/EPSG v12.029 report; independent revalidation remains X142-002. |
| Study-polygon research claim | HOLD | Repository bbox semantics are corrected; X142-001 must supply the governed polygon. |
| Area-sensitive confirmatory metric | FAIL-CLOSED / EXTERNAL HOLD | R142-003/007 reject uncertainty-crossing maximum caps and minimum growth evidence; independent scale evidence remains required for an external claim. |
| Single-/multi-radar exploratory research | CONDITIONAL GO | Must report the registered bbox and unverified geodetic budget. |
| Multi-radar confirmatory claim | HOLD | Repository closure and X142 external evidence are required. |
| External publication | HOLD | Independent geodetic validation, study polygon, and cohort evidence are required. |
| Operational deployment | NO-GO / OUT OF SCOPE | This project is for scientific validation, not deployment authorization. |
| PR #142 merge | HOLD | Independent review and exact-head required CI are required. |

## Implemented evidence boundary

```text
Package                              0.106.0 -> 0.107.0
RadarMetricDomainContract            v1 digest preserved exactly
RadarMetricDomainEvidence            new v1
Geodetic report                      new v1
Registered maximum linear error      0.006
Registered maximum area error        0.012036
Area threshold policy                interval-stable / fail-closed
Metric evidence binding              grid/run current + durable action v6
Forecast run artifact                v68 audit-only -> v69 current
CLI output contract                  nowcast-npz-v74 -> v75
Verification/FSO/replay generations  unchanged
Deployment generations               unchanged / out of scope
```

The evidence points to the historical/current metric-domain digest rather than
changing that digest. Historical grid payloads remain audit-loadable, while
new current grid/run/action artifacts additionally preserve the exact evidence
digest. The report SHA remains bound through shipped source and package data.

## Local validation evidence

```text
Targeted and adjacent scientific CPU suites:
  tests.test_nowcast
  tests.test_sensitivity
  tests.test_variational
  tests.test_ledger
  tests.test_nowcast                                      160 PASS
  tests.test_sensitivity + tests.test_variational
    + tests.test_ledger                                  272 PASS
  test_run_artifact discovery                             49 PASS
  tests.test_cli                                          21 PASS
  current full-product semantic replay promotion           1 PASS
  total                                                  503 PASS

Geodetic report regeneration:
  PROJ 9.7.1 / EPSG v12.029 / 17 x 17 lattice
  report SHA-256:
    2f87382f3af7c190ec344e9eb4f4efbad688b1795f6e7081017e6c780f0d4e7f
  EPSG PROJJSON digest:
    2b7176e8ed8279b569e1be3fa843225e85f6880c52354ded49c9bef14d81d667
  byte-identical regeneration with poisoned PROJ_LIB/PROJ_DATA PASS

Historical metric-domain digest:
  origin/main and working tree:
    c3fb6d3ce964863757dac7f1a58c1195723349c3a655f18276a78b71a883865c
  exact equality PASS

Scientific package evidence:
  isolated wheel build PASS
  advar/data/epsg5179_metric_domain_evidence_v1.json present in wheel

Static/syntax evidence:
  python -m py_compile PASS
  git diff --check PASS
  basedpyright unavailable in the local environment

Broad all-tests discovery:
  intentionally stopped after 37 minutes in unrelated long promotion tests;
  no failure had occurred. This follows the project boundary that prioritizes
  scientific validation and avoids spending the cycle on deployment checks.
```
