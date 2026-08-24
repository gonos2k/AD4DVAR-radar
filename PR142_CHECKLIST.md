# PR #142 geodetic metric-evidence closure checklist

## Authority snapshot

- Cycle authority key: `ADVAR-R142-e4e5dc7-b4a834b-2026-08-24`
- Predecessor checklist/PR/merge commit: `PR141_CHECKLIST.md` / PR #141 / `e4e5dc7c3f8c9a709f62cb2a3025d3849c4dc5b3`
- Review source: user-provided additional review of merged PR #141
- Review timestamp/time zone: 2026-08-24 / Asia-Tokyo
- Reviewer-stated base/head: `main@e4e5dc7`, PR #141 head `8a9185d65ba461ec77f8765ef4d9747e65e8a06b`
- Verified repository base: `origin/main@e4e5dc7c3f8c9a709f62cb2a3025d3849c4dc5b3`
- Verified PR/head/tree: PR #141 `MERGED`; reviewed head is reachable from `origin/main`; base tree `b4a834bf03040e071aa53d315f3dffd385dcc7bc`
- Worktree state: fresh branch `agent/pr142-metric-domain-evidence`; all user-owned untracked artifacts (including `.omx/`, `.DS_Store`, `graphify-out/`, and local documents) preserved and excluded from the PR
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
| R142-011 | sampled geographic lattice → validated projected coverage → grid/radar membership; a broad historical bbox must not stand in for sampled evidence coverage | adversarial review of the first hardened implementation | new current-evidence eligibility correction | the historical bbox contract and digest remain unchanged |
| R142-012 | legacy nested observation bytes → historical replay semantics; current evidence validation must not retroactively strengthen old observation constructors | adversarial review of generation propagation | new legacy/current validator separation | v17 nested observation arithmetic and canonical identities remain historical |
| R142-013 | observation algorithm identity → implementation semantics; unchanged v10/v11 algorithms must retain exact golden digests | adversarial review of generation propagation | restored original algorithm preimages and added golden tests | source-composed observation derivation rules are unchanged |
| R142-014 | preregistered plan source → scoring start → replay → completion; format-valid digests must not substitute for the active scientific implementation | adversarial review of scientific replay | new active-boundary closure | cold audit decoding remains independent of the installed package |
| R142-015 | legacy outer generations and durable JSON schemas → audit loader; new optional fields must not appear even as `null` in historical bytes | adversarial review of durable compatibility | exact-type audit decoders and exact legacy member sets | v5 intervention and v22/v32/v14/v31 scientific artifacts remain audit-only |
| R142-016 | scientific generation edits → unrelated ledger transition logic; mechanical branch edits must not alter the operational history state machine | final adversarial diff review | accidental fall-through removed before commit | deployment remains fail-closed and out of project scope |
| R142-017 | geodetic report values → generator implementation; report provenance must identify the exact script and canonical output contract | final adversarial provenance review | generator source and output semantics added to evidence | independent reproduction remains an external HOLD |
| R142-018 | grid v5 contract string → scientific capability; evidence-absent audit payload and evidence-bound current payload must not be conflated | final adversarial contract review | outer current validators remain the capability boundary | historical v5 bytes remain constructible for audit |
| R142-019 | durable manifest checksum → semantic member set; checksummed but unreferenced JSON keys must fail | final adversarial durable-replay review | exact v5/v6 manifest schemas added | v3/v4 historical decoding remains unchanged |
| R142-020 | generic numerical tolerance → physical area interval; a dimensionless/numeric tolerance must not relax a km² scientific minimum | final adversarial scientific review | tolerance removed from the area interval gate | support-count tolerance remains unchanged |
| R142-021 | predecessor decoder → real directory/ledger load; outer audit types require more than constructor-only tests | final adversarial test review | v22 directory cold-load and v31 ledger-row fixtures added | predecessor semantic execution remains prohibited |
| R142-022 | generator source digest → committed report → required CI; a recorded source hash must be checked against current script bytes | fourth consecutive adversarial review | platform-independent source-only CI check added | exact PROJ regeneration remains a separately recorded local/external action |
| R142-023 | validation partition counts → reported total; every evidence subtotal must reconcile exactly | fifth consecutive adversarial review | stale `434` subtotal corrected to the executed `435` | no scientific or runtime behavior changed |

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
| R142-008 | P2-PROVENANCE | The generator can combine a manual projection definition with an unrelated `projinfo` database and does not clear `PROJ_DATA`. | source CRS authority → factor samples → report | REPRODUCED by static generator inspection. | repository-actionable | Derive the projection definition from the selected `projinfo` database, require one colocated and version-consistent toolchain, record all exact binary/database hashes, and clear both PROJ data environment variables. | Path/database or version mismatch fails; exact identities are recorded; deterministic regeneration remains byte-identical. | ✅ colocated, exact-identity toolchain | ✅ poisoned `PROJ_LIB`/`PROJ_DATA` regeneration byte-identical | ☐ PR/CI |
| R142-009 | P2-INTEGRITY | Runtime trusts report-provided sample digests and maxima after only checking the outer file SHA. | shipped report bytes → typed evidence semantics | REPRODUCED: per-sample scale values were not independently reduced at load time. | repository-actionable | Recompute lattice ordering, scale digests, and maxima from the report and compare them to all typed evidence fields. | Mutating one sample with a self-consistent outer report object is rejected; committed report passes. | ✅ runtime lattice/reduction replay | ✅ one-sample adversarial mutation rejected | ☐ PR/CI |
| R142-010 | P2-CONTRACT | `RadarMetricDomainContract(**contract.payload)` fails because the stable payload uses the historical polygon-named key while the current constructor uses bbox naming. | canonical payload → typed contract reconstruction | REPRODUCED with the current object. | repository-actionable | Add an explicit canonical-payload decoder that maps the historical key without changing the payload or digest. | `from_payload(contract.payload)` round-trips exactly; unknown/ambiguous keys fail. | ✅ explicit `from_payload()` decoder | ✅ exact round-trip and ambiguous-key rejection PASS | ☐ PR/CI |
| R142-011 | P1-SCIENTIFIC | The report sampled an inscribed geographic lattice, but current membership initially reused the much broader historical bbox and could claim evidence outside sampled coverage. | factor report → projected coverage → current grid/radar eligibility | REPRODUCED with a broad-bbox point outside the report's projected coverage. | repository-actionable | Derive and content-address the conservative projected coverage from the sampled source boundary and validate every current grid and radar site against it. | A point inside the historical bbox but outside sampled coverage is rejected; current fixtures remain valid. | ✅ sampled coverage digest and bounds in evidence v1 | ✅ broad-bbox adversarial point rejected | ☐ PR/CI |
| R142-012 | P1-PROVENANCE | Applying current evidence checks inside a shared legacy geometry helper would make historical v17 nested artifacts depend on new repository resources. | v17 nested observation replay → current v18 eligibility | REPRODUCED during adversarial generation review. | repository-actionable | Keep historical range/elevation arithmetic unchanged and call a separate current-evidence validator only from new issue/v18 paths. | Legacy geometry without evidence still performs historical arithmetic but cannot enter current v18 validation. | ✅ legacy helper/current validator split | ✅ legacy arithmetic and current rejection PASS | ☐ PR/CI |
| R142-013 | P1-CONTRACT | Adding metric-evidence fields to unchanged observation algorithm payloads silently changed v10/v11 digests despite no algorithmic change. | algorithm preregistration → canonical implementation identity | REPRODUCED by digest comparison against `origin/main`. | repository-actionable | Restore exact historical preimages and use outer scientific generation bumps for the new eligibility boundary. | Mask v10 and error v11 retain exact golden SHA-256; current verification is v18 and v17 is audit-only. | ✅ exact preimages restored; outer generations advanced | ✅ golden digest and mapping tests PASS | ☐ PR/CI |
| R142-014 | P1-SCIENTIFIC | A holdout plan could store arbitrary format-valid scoring source/metric-engine digests, weakening the claimed preregistration edge. | plan registration → scoring start → replay → completion | REPRODUCED in constructor-only plan tests. | repository-actionable | Permit arbitrary digests only for typed cold audit objects; require exact active source and metric-engine identity at every live ledger boundary. | Mismatch is rejected at plan append, scoring start, replay append, and semantic completion; historical payload still decodes. | ✅ four active-boundary checks | ✅ mismatched source/engine adversarial tests PASS | ☐ PR/CI |
| R142-015 | P1-AUDIT | New fields and subclasses could make v5 durable manifests or v22/v32/v14/v31 scientific artifacts look current or change their exact JSON member sets. | durable bytes → typed audit generation → current scientific use | REPRODUCED for a v5 manifest containing new keys and for subclass-sensitive current checks. | repository-actionable | Omit new keys entirely from v5, reject their presence on load, use exact generation types, and add typed audit decoders for every predecessor. | Real predecessor-shaped fixtures audit-load; attempts to use them for current replay/scoring fail closed. | ✅ v5 exact schema + four audit types | ✅ key-presence and audit-boundary tests PASS | ☐ PR/CI |
| R142-016 | P1-INTEGRITY | A mechanical generation edit briefly inserted a promotion-contract branch into an unrelated operational-history decoder. | ledger history transition → state reconstruction | CAUGHT before commit by adversarial diff review; no intended behavior relied on it. | repository-actionable | Restore unconditional rejection and keep the operational selector unchanged. | Existing history tests pass; current scientific v32 evidence remains rejected by deployment. | ✅ stray branch removed; explicit out-of-scope rejection | ✅ targeted ledger/promotion regression PASS | ☐ PR/CI |
| R142-017 | P2-PROVENANCE | The report identified PROJ binaries and `proj.db`, but not the repository generator source that connected those tools to the committed reductions. | generator implementation → canonical report bytes → typed evidence | REPRODUCED: long-term replay depended implicitly on the Git commit. | repository-actionable | Record the exact generator source SHA-256 and canonical output contract in the report and typed evidence. | Generator mutation changes the report/evidence identity; `--check` binds current source to committed bytes. | ✅ generator contract/source/output binding | ✅ generator tamper fields rejected; `--check` PASS | ☐ PR/CI |
| R142-018 | P2-CONTRACT | `radar-spatial-grid-identity-v5` can represent both historical evidence-absent and current evidence-bound payloads. | grid contract string → current scientific capability | CONFIRMED; all active v18/run69/replay23 boundaries nevertheless reject absence. | documentation and regression closure; a v6 grid cascade is unnecessary for this outer-generation change | State explicitly that the outer validator, not the v5 string alone, grants current capability. | Evidence-absent v5 can audit-load but fails `validate_current_metric_domain_evidence()` and v18 issuance. | ✅ capability boundary documented | ✅ missing evidence/current issuance rejection PASS | ☐ PR/CI |
| R142-019 | P2-INTEGRITY | A durable v5/v6 manifest could contain an unreferenced extra JSON key if its checksum was recomputed. | manifest bytes → logical replay preimage | REPRODUCED: loader validated required values but not exact key equality. | repository-actionable | Define exact v5 and v6 field sets and reject additions/removals before replay. | Extra v6 key and v6-only keys under v5 both fail before tensor replay. | ✅ exact current durable schemas | ✅ extra-key and cross-generation key tests PASS | ☐ PR/CI |
| R142-020 | P2-SCIENTIFIC | The general `1e-6` tolerance was added to area km² before the minimum interval gate, permitting up to roughly 1 m² shortfall. | projected support area → conservative ground lower bound → minimum evidence | REPRODUCED mathematically and in the call path. | repository-actionable | Remove the generic tolerance from physical-area interval decisions. | A minimum only `5e-7 km²` above the lower bound fails; clearly sufficient area passes. | ✅ strict lower-bound comparison | ✅ near-boundary regression PASS | ☐ PR/CI |
| R142-021 | P2-TEST | Initial predecessor tests decoded v22/v32/v14 constructors only and had no v31 ledger row or v22 directory cold-load. | predecessor durable bytes → audit loader → current rejection | REPRODUCED as a test-coverage gap. | repository-actionable | Add predecessor-shaped sharded directory and indexed promotion fixtures. | v22 directory byte-load succeeds with all semantic flags false and rejects typed cases; v31 ledger row returns exact audit type. | ✅ directory + ledger fixtures | ✅ audit load/current-use rejection PASS | ☐ PR/CI |
| R142-022 | P2-PROVENANCE | The report records the generator source SHA, but required CI did not compare it with the current script; source/report drift could pass ordinary package tests. | generator bytes → committed report provenance → required CI | REPRODUCED: only local full-PROJ `--check` exercised the binding. | repository-actionable | Add a PROJ-independent source-only check and run it in the required Wheel/CLI job. | Current script/report pass; one-byte script mutation fails without invoking PROJ. | ✅ `--check-source-only` + required CI step | ✅ current/tampered source regression PASS | ☐ PR/CI |
| R142-023 | P3-GOVERNANCE | The final validation total was updated to 511 while its nowcast-family subtotal still read 434, producing an internally inconsistent evidence table. | executed test partitions → checklist evidence total | REPRODUCED by arithmetic during the fifth review. | repository-actionable | Correct the stale subtotal and recheck all partitions. | `435 + 6 + 70 = 511`; `366 + 2 + 50 = 418`. | ✅ evidence table reconciled | ✅ executed pytest summaries rechecked | ☐ PR/CI |

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
- [x] Adjacent scientific CPU suites pass after adversarial-review corrections (`511 passed`, `418 subtests passed`).
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
Grid v5 capability boundary          outer current validator, not string alone
Forecast run artifact                v68 audit-only -> v69 current
CLI output contract                  nowcast-npz-v74 -> v75
Verification bundle                 v17 audit-only -> v18 current
Variational FSO / FSOI              v23/v19 -> v24/v20 current
Semantic replay                     v22 audit-only -> v23 current
Holdout plan                        v32 audit-only -> v33 current
Holdout scoring artifact            v14 audit-only -> v15 current
Scientific promotion evidence       v31 audit-only -> v32 current
Deployment generations               unchanged / out of scope
```

The evidence points to the historical/current metric-domain digest rather than
changing that digest. Historical grid payloads remain audit-loadable, while
new current grid/run/action artifacts additionally preserve the exact evidence
digest. The report SHA remains bound through shipped source and package data.

## Local validation evidence

```text
Targeted and adjacent scientific CPU suites:
  nowcast + sensitivity + variational + ledger           435 PASS
    subtests                                              366 PASS
  selected scientific promotion/replay boundaries          6 PASS
    subtests                                                2 PASS
  run artifact + CLI                                      70 PASS
    subtests                                               50 PASS
  total                                                  511 PASS
    total subtests                                        418 PASS

Geodetic report regeneration:
  PROJ 9.7.1 / EPSG v12.029 / 17 x 17 lattice
  report SHA-256:
    6eeb22c0665566b69ce6590b7176607492a473e63e03f743178283aca771a098
  typed evidence digest:
    de36641c6f97178ebfd1a1fda88a33f2b8c333ce46acbadef2a965e4d3e219ad
  generator source SHA-256:
    bdfe670be17f4fa85047f0c0614e56756d95ec2bc30c597c4b13303d646db508
  validated projected coverage digest:
    993053c5ac611186134ea14f07af6202f5ba254c21aa0a1d0c44c8a9d9fa5849
  validated projected coverage (metre):
    easting 592664..1576674; northing 976711..2251910
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
  basedpyright --level error on changed source: 0 errors

Broad all-tests discovery:
  intentionally stopped after 37 minutes in unrelated long promotion tests;
  no failure had occurred. This follows the project boundary that prioritizes
  scientific validation and avoids spending the cycle on deployment checks.
```
