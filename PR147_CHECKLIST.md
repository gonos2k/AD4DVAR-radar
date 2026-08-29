# PR #147 transcendental enclosure and audit-authority checklist

## Authority snapshot

- Cycle authority key: `main@f13f6efe2654040b302bd10d869a40d19a4914da` / tree `3c8c4d70f0ecc17df29f23ffaa21972453156d91` / user-provided post-merge review / `2026-08-29 Asia/Tokyo`
- Predecessor checklist/PR/merge commit: `PR146_CHECKLIST.md` / PR #146 / `f13f6efe2654040b302bd10d869a40d19a4914da`
- Review source: pasted user review; linked `sandbox:/mnt/data` HTML/Markdown/JSON artifacts are not present in this workspace
- Reviewer-stated current authority: latest `main`; exact SHA was independently resolved to `f13f6efe2654040b302bd10d869a40d19a4914da`
- Verified predecessor PR/head/merge: PR #146 `MERGED`; head `3a769fbf5be9cd90244767c80d268a235085beed`; merge `f13f6efe2654040b302bd10d869a40d19a4914da`; head is an ancestor of `origin/main`; head and merge trees are identical
- Exact-main CI: run `33184956687` `SUCCESS` for Python 3.10 CPU, Python 3.12 CPU/type check, Wheel/CLI and Linux geodetic self-test
- Fresh branch: `agent/pr147-transcendental-audit-authority` created exactly at `origin/main@f13f6efe2654040b302bd10d869a40d19a4914da`
- Pull request: [#147](https://github.com/gonos2k/AD4DVAR-radar/pull/147) is OPEN; implementation commit `ccdc9b2e602a4e68aa8e8a0b80417d14a92ac38c`; merge is not authorized
- Worktree: `/Users/yhlee/ADVAR`; tracked tree clean before checklist creation; pre-existing untracked user files inventoried and preserved

## Review-source completeness note

The pasted review explicitly identifies `R147-002` and describes two additional
unnumbered repository hardening findings. The linked report artifacts are not
available in this execution environment, so a finding that may be named
`R147-001` in those artifacts is not invented or silently marked closed here.
This checklist covers every statement visible in the pasted source. Full-report
closure requires the missing Markdown or machine-readable JSON to be attached
or copied into the workspace.

## Deduplication and non-regression ledger

| Finding ID | Semantic fingerprint (`boundary + invariant + acceptance intent`) | Prior finding/status | Current disposition | Existing guard to preserve |
|---|---|---|---|---|
| R147-002 | age/scale/power → `pow`/`exp` interval → source-score strict dominance cannot certify a winner outside the true transcendental enclosure | R146-002 closed geometry dtype and compositional algebraic rounding; transcendental kernel error bound remained assumed | refined repository P1 | float64 geometry, strict lower-over-rival-upper selection, ambiguous source remains unassigned |
| R147-003 | registry `audit_readable` union → frozen per-generation bytes → cold decode/digest/type/tamper/action policy exact coverage | R146-006 closed named executable probes for registered recent generations only | refined repository P1-AUDIT | current lifecycle tests execute; audit-only generations cannot re-enter scientific or operational action |
| R147-004 | affine/projected/ground primitive → nominal plus directed bounds → every diagnostic, context, replay and hard gate derives from one authority | R146-001 closed hard-gate affine primitive intervals | refined repository P2-ARCHITECTURE | hard maximum uses upper, minimum uses lower, set-valued footprint remains fail-closed |
| X147-001 | approved study polygon + independent geodesy/environment + preregistered real-radar cohort → confirmatory/publication authority | X146-001 OPEN | external carry-forward | bounded offline assumptions remain explicit |
| X147-002 | representative-tilt geometry → beam-path interval or validated error envelope | X146-002 OPEN | external carry-forward | current geometry scope remains explicit |

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R147-002 | P1-NUMERICAL/SCIENTIFIC | Fixed `pow`/`exp` ±4 ULP widening has no registered backend-wide proof and can make strict source dominance false-positive. | temporal quality transcendental kernels → score interval → selected radar | **CLOSED**: current source-selection v5 never executes backend `pow`/`exp` for winner certification. Exact zero age has `[1,1]`; every positive age has the analytic codomain `[0,1]`; a source is assigned only when its lower score strictly exceeds every competitor upper. Non-finite or negative score intervals are rejected. | repository-actionable | Replace assumed kernel ULP accuracy with an authoritative, reproducible enclosure or fail-closed dominance margin; bind runtime/error policy into all semantic generations. | Patched `torch.pow`/`torch.exp` cannot be called; a temporal near-tie remains unassigned; top-2 dominance matches the quadratic oracle and rejects invalid intervals. | ☑ | ☑ | exact-head CI pending |
| R147-003 | P1-AUDIT | Named lifecycle probes do not prove that every declared audit-readable generation has one distinct frozen cold-replay fixture. | registry capabilities → historical bytes → decoder/action/tamper lifecycle | **CLOSED at the repository archive boundary**: 41/41 registered `audit_readable` generations have distinct canonical frozen envelopes, exact registry/path set equality, digest/type/action metadata checks and one-byte tamper rejection. Family-specific real decoder probes run in isolated interpreters. This is not presented as recovery of unavailable external archives. | repository-actionable | Introduce typed fixture matrix and exact set equality against all registered audit-readable generations for required families. | Missing or extra fixture fails; all 41 envelopes cold-decode; family decoder probes execute without skip; scientific/operational action policy is registry-derived. | ☑ | ☑ | exact-head CI pending |
| R147-004 | P2-ARCHITECTURE | Nominal metadata and directed hard-gate paths can diverge, allowing nearest scalars to become unguarded scientific inputs later. | physical primitive producer → metadata/context/replay/hard-gate consumers | **CLOSED**: `DirectedPhysicalValue` is the single nominal/lower/upper authority for affine norm and determinant/area. PSR, FSS, footprint, centroid projection, smoothness weights, speed gates and area consumers now use that authority; hard maxima select upper bounds and diagnostic/projected quantities select nominal explicitly. | repository-actionable | Return nominal/lower/upper from one authoritative primitive and migrate scientific consumers to explicit bound selection. | Exact-binary affine motion/footprint/area counterexamples are enclosed; PSR sees the directed uncertain annulus; centroid/smoothness and area consumers no longer recompute the matrix independently. | ☑ | ☑ | exact-head CI pending |

## Five-pass adversarial review record

1. **Numerical enclosure pass — PASS.** Removed unaudited transcendental ULP assumptions entirely, verified strict interval dominance against a quadratic oracle, and added NaN/negative fail-closed checks.
2. **Generation/digest pass — PASS.** Synchronized grid v6, registry/geometry v7, plan v16, verification v22, FSO v28, FSOI v24, replay v27, forecast v72, holdout v37 and package 0.112.0. This pass found and corrected a stale Wheel/CLI `forecast-run-v71` assertion.
3. **Audit coverage pass — PASS WITH STATED SCOPE.** The first matrix covered audit-only generations and missed current-but-audit-readable generations. It was expanded to exact 41/41 registry coverage, including an exact filesystem equality check. Frozen repository envelopes and family decoder probes are distinguished from unavailable external historical archives.
4. **Consumer-authority pass — PASS.** Found three residual bypasses after the first implementation: PSR recomputed affine distance in correlation dtype, projected centroid built its own matrix, and variational smoothness recomputed axis norms. All now consume the shared grid authority; the PSR exact-boundary counterexample is fixed as a regression.
5. **Composition/order pass — PASS.** The first executable lifecycle harness ran other test methods in-process and contaminated later promotion fixtures. Probes now execute in fresh isolated interpreters; registry-then-promotion composition is an explicit regression. No production state is shared by the meta-test.

## Local verification evidence

- Affected nowcast/sensitivity/variational suite: `398 passed`, `356 subtests passed`.
- Contract registry plus the three previously order-dependent promotion cases: `12 passed`, `53 subtests passed` in one parent pytest process.
- CLI: `21 passed`, `22 subtests passed`.
- Focused source-selection adversarial tests: `3 passed`.
- Holdout/replay/promotion/forecast audit probes: focused tests passed.
- basedpyright product-source check: `0 errors`.
- Metric evidence source binding and local execution-environment self-test: PASS.
- sdist and wheel build for package `0.112.0`: PASS.
- Full CPU suite after final composition fix: `830 passed`, `538 subtests passed`, `18 warnings` in `3949.41s`.

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S147-001 | R146-001 through R146-007 remain independently reported as closed. | PR #146 exact-head and exact-main CI, user re-review | New work must extend rather than weaken directed affine, FP64 geometry, audit mapping, top-2 dominance and release/API guards. |
| S147-002 | Detection category uses direction-specific lower/upper limits and explicit uncertainty. | current verification path | Transcendental changes must not collapse detection or source-score uncertainty. |
| S147-003 | Operational capability is intentionally empty. | contract registry | Audit fixture work must never make historical artifacts operationally actionable. |

## External actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X147-001 | scientific governance, independent geodesy and radar-science owners | Provide approved polygon, independently sealed geodetic revalidation and preregistered single/multi-radar cohorts with target/QC provenance. | CI and internal evidence cannot grant external scientific authority. | confirmatory publication and generalized multi-radar claim | OPEN |
| X147-002 | radar-science owner | Add beam-path interval geometry or validate representative-tilt error envelopes by radar/range/sweep/regime. | Current model remains projected-horizontal representative tilt with altitude provenance only. | long-range clear/censor confirmatory interpretation | OPEN |

## Acceptance summary

- [x] Every statement visible in the pasted review is represented once and fingerprinted.
- [x] Missing linked report artifacts and the possible unseen `R147-001` are explicitly disclosed rather than inferred.
- [x] This post-merge cycle starts on a fresh branch exactly at current `origin/main`; PR #146 head is reachable through its merge commit.
- [x] Every visible repository-actionable P1/P2 is fixed or evidence-disproved, subject to the absent-report caveat for unseen `R147-001`.
- [x] Every implemented fix has a targeted adversarial regression.
- [x] Affected contract generations, digests, fixtures, release metadata and public API are synchronized.
- [x] Focused, adjacent and full CPU verification pass.
- [ ] PR head equals the reported pushed commit and exact-head CI is terminal green.
- [x] External polygon/geodesy/cohort and beam-geometry actions remain visible.
- [x] Merge remains HOLD unless explicitly authorized.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Internal offline exploratory research | GO WITH EXPLICIT ASSUMPTIONS | Existing main CI is green; new P1 findings affect certification boundaries. |
| Confirmatory verification/FSO/FSOI | CODE-CLOSED for visible findings; external HOLD remains | R147-002/R147-003/R147-004 are locally closed; exact-head CI and the unseen-report caveat still apply. |
| Publication-suppressed shadow | HOLD | Repository P1 and external scientific gates remain. |
| Canary | NO-GO / OUT OF SCOPE | Operational capability remains intentionally empty. |
| State-advancing LIVE | NO-GO / OUT OF SCOPE | Operational capability remains intentionally empty. |
| External publication | HOLD | External polygon/geodesy/beam/cohort evidence remains open. |
| PR #147 merge | HOLD | PR #147 is OPEN and mergeable; local verification is complete, exact-head CI is pending, and merge is not authorized. |

## 2026-08-30 supersession note

PR #147은 실제로 merge commit
`f44e7ef036e3a0eb5526944f2b32fbb79f3b856d`로 병합됐고 main push CI
`33252793248`도 성공했다. 다만 위의 41/41 frozen-envelope closure는 실제
historical artifact cold replay가 아니라 self-consistent metadata/probe coverage였다.
PR #148은 그 보증 주장을 철회하고 fixture matrix, action wrapper와 lifecycle
subprocess를 삭제한다. 이 기록은 당시 판정을 보존하지만 현재 권위로 사용하지 않는다.
