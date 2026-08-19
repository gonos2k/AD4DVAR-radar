# AD4DVAR-radar scientific validation closure checklist

## Authority snapshot

- Review source: user-supplied additional review for `main@fb6e593`, v0.93.0
- Review timestamp/time zone: 2026-08-19 / Asia-Tokyo
- Verified base: `origin/main@fb6e5936549c894fc2abae562bb9345a61331f28`
- Verified preceding PR: PR #130, head `3d01e70db65cd42e91c810cb403745a45d0c8f04`, merge `fb6e5936549c894fc2abae562bb9345a61331f28`
- Verified PR #130 CI: Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI smoke all successful in run `32211799387`
- Worktree state before edits: tracked files clean; user-owned `.omx/` preserved and excluded
- Product intent: scientific validation and offline research, not state-advancing deployment
- Deployment decision: candidate-smoke may remain as a non-authoritative engineering check; shadow, canary, and LIVE are NO-GO and are not a release objective
- Local scientific contract version: package `0.94.0`; deployment/run generations intentionally unchanged
- Broad adjacent evidence: acceptance+sensitivity+promotion+ledger `350 passed`, `139 subtests passed`, `18 warnings` in `2045.96s`
- Final targeted evidence: `13 passed`, `18 warnings` in `5.25s`
- Static evidence: basedpyright `0 errors`; `git diff --check` clean
- Package evidence: isolated-output sdist and wheel build succeeded as `advar_radar_nowcast-0.94.0`
- Final local diff audit: intended scientific files only; deployment builder/workflow unchanged; `.omx/` excluded
- Prepared PR narrative: `PR131_PR_BODY.md` leads with the scientific-validation pivot, explicit deployment non-goals, allowed claims, and external evidence holds

## Scientific validation findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| SCI-001 | P1-HIGH | Real-case acceptance checks generic JSON identity rather than typed scientific semantics and cross-stage lineage. | native radar → analysis → forecast → verification → target → scoring | REPRODUCED, FAIL-CLOSED LOCALLY | repository-actionable | Downgrade current harness to a content-addressed artifact index; add typed validators only with real-case artifacts. | Generic JSON cannot set semantic readiness. | ☑ fail-close | ☑ targeted | ☐ |
| SCI-002 | P1-HIGH | Independent physical-event count depends on caller-provided labels. | acceptance case → sample-size preflight | REPRODUCED, FAIL-CLOSED LOCALLY | repository-actionable | Do not calculate independent sample size from caller labels; derive it only after typed event identity exists. | Relabelled duplicate closure never satisfies sample size. | ☑ fail-close | ☑ targeted | ☐ |
| SCI-003 | P1-HIGH | Observation-error generation policy is not preregistered before forecast scoring. | holdout plan → verification weights | REPRODUCED, FIXED LOCALLY | repository-actionable | Add a deterministic observation-error plan and byte-identical derivation receipt. | Post-forecast registry/algorithm/parameter changes fail. | ☑ | ☑ targeted | ☐ |
| SCI-004 | P1 | Inverse-variance weighting is not a probabilistic observation-error convolution or censored likelihood. | probabilistic verification scores | REPRODUCED, FIXED LOCALLY AS REPORT-ONLY | repository-actionable after statistical contract is explicit | Add diagnostic proper scores using forecast variance plus observation variance and censored targets. | Synthetic calibrated forecasts reward the correct predictive distribution. | ☑ diagnostic | ☑ targeted | ☐ |
| SCI-005 | P1 | Missing-state taxonomy is declarative rather than a per-cell scientific state. | verification observation → metric eligibility | REPRODUCED, FIXED LOCALLY | repository-actionable | Add typed per-cell observation state and enforce value/mask/weight/source invariants. | Clear, echo, missing, QC-invalid, blocked, censored, and unassigned states remain distinguishable. | ☑ | ☑ targeted | ☐ |
| SCI-006 | P1 | Spatial-correlation digest is not consumed by uncertainty inference. | correlation blocks → confidence bounds | REPRODUCED, FIXED LOCALLY AS DIAGNOSTIC-ONLY | repository-actionable | Add event/block membership to cluster-aware inference or mark it diagnostic-only. | Pixel block metadata cannot claim inferential use; current bounds remain physical-event clustered. | ☑ diagnostic | ☑ targeted | ☐ |
| SCI-007 | P1 | No repository-owned mock-free real radar corpus is available for the mandatory scenarios. | scientific claims → external evidence | EXTERNAL_BLOCKED | external-action | Obtain legally usable fixed native cases and independent reference observations. | Clean offline replay with immutable checksums and no direct digest injection. | N/A | N/A | N/A |

## Resolution sequence

| Order | Gate | Resolution | Status | Next evidence |
|---:|---|---|---|---|
| 1 | Preregister the estimand | Observation-error algorithms, registries, parameters, censoring, and source assignment are fixed in `VerificationObservationErrorPlan-v1` before scoring. | DONE | External protocol owner reviews the scientific assumptions. |
| 2 | Preserve observation semantics | Seven typed cell states and exact mosaic source-map consistency distinguish echo, clear, censoring, missingness, QC rejection, blockage, and source non-assignment. | DONE | Real radar cases exercise every state. |
| 3 | Add proper probabilistic diagnostics | Quantized Gaussian likelihood uses forecast plus observation variance; below-detection observations use a left-censored likelihood. The artifact is report-only. | DONE, DIAGNOSTIC | Compare against weighted legacy metrics under a preregistered protocol. |
| 4 | Bound correlation claims | Spatial-block metadata is explicitly diagnostic-only; confirmatory inference remains clustered by physical event. | DONE, DIAGNOSTIC | Approve a block model before any inferential promotion. |
| 5 | Remove false semantic readiness | Generic stage JSON is accepted only as a content-addressed index and can never set semantic E2E readiness. | DONE, FAIL-CLOSED | Implement typed stage replay only when real artifacts exist. |
| 6 | Remove label-based sample-size inflation | Caller event labels are reported but never counted as independent physical events. | DONE, FAIL-CLOSED | Derive event identity from typed native/target/time/domain evidence. |
| 7 | Run external scientific validation | Legally usable radar corpus, independent observations, and a preregistered analysis protocol are required. | EXTERNAL HOLD | X-001, X-002, and X-003. |

The sequence is intentionally one-way: steps 1–6 prevent synthetic or
caller-declared metadata from being promoted into a scientific claim, while
step 7 supplies the real observations needed to evaluate the method. No
deployment capability is a prerequisite or acceptance criterion for this
sequence.

## Deployment findings retained as non-goal boundaries

| ID | Priority if LIVE were enabled | Finding | Current-tree result | Disposition |
|---|---|---|---|---|
| DEP-001 | P0 | Release approval does not bind the exact runtime/interpreter/install closure selected by runtime activation. | REPRODUCED | Do not enable state-advancing deployment; no implementation in the scientific-validation PR. |
| DEP-002 | P0 | Builder and product use different sequence-1 activation genesis digests. | REPRODUCED | Candidate-smoke remains non-authoritative; product deployment is NO-GO. |
| DEP-003 | P1 | Activation row and mutable head can diverge under direct SQL. | UNVERIFIED | Out of current scientific scope; deployment remains NO-GO. |
| DEP-004 | P1 | Activation chronology and rollback reason contracts need additional hardening. | UNVERIFIED | Out of current scientific scope; deployment remains NO-GO. |

## Strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S-001 | CPU numerical and variational core has no newly reviewed P0. | PR #130 final-head CPU suites: 777 tests, 410 subtests. | Keep full FP32/FP64 CPU suite required. |
| S-002 | Native/raw/analysis and training/target provenance are typed and durable. | Current v68/v74 artifact chain. | Preserve current-contract and audit-only separation. |
| S-003 | Observation quality and standard deviation already affect metric weights. | Verification observation-error plan v1, realized contract v3, and per-cell observation state. | Compare existing weighted metrics with new proper-score diagnostics. |
| S-004 | Acceptance remains report-only and never authorizes deployment. | `authorizes_deployment=false`. | Preserve unconditional non-authorization. |

## External scientific actions

| ID | Owner | Required action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X-001 | Radar-data owner | Supply legally usable native radar cases covering required outage, handoff, censoring, and storm regimes. | Immutable source/checksum manifest. | real-case scientific claims | OPEN |
| X-002 | Independent verification owner | Supply observations not derived from the candidate training/verification source closure. | Source authority and observation-error metadata. | independent skill claims | OPEN |
| X-003 | Scientific protocol owner | Approve observation likelihood, censoring, event grouping, and cluster inference before inspecting candidate results. | Preregistered signed protocol. | confirmatory promotion/publication | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable scientific P1 is fixed, explicitly diagnostic-only, or fail-closed.
- [x] Every fix has a targeted regression test.
- [x] Adjacent and broad CPU suites pass.
- [x] Current scientific artifact producers and consumers are synchronized.
- [x] External real-case evidence is clearly separated from synthetic regression evidence.
- [x] Candidate-smoke, scientific report, and deployment authority remain separate decisions.
- [x] State-advancing deployment remains NO-GO.
- [x] Merge remains HOLD unless explicitly authorized.

## Current decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Development and offline research | GO | Existing numerical/provenance evidence. |
| Synthetic scientific regression | GO | Must remain labeled synthetic. |
| Real-case descriptive report | HOLD | Current harness is artifact-index-only; X-001 remains open. |
| Confirmatory promotion claim | HOLD | X-002 and X-003 remain open; SCI-004/SCI-006 remain diagnostic-only pending protocol approval. |
| Candidate-smoke engineering check | GO | Non-authoritative only. |
| Publication-suppressed shadow | NO-GO | Deployment is not a project objective. |
| Canary / state-advancing LIVE | NO-GO | DEP-001 and DEP-002 remain open by scope decision. |
| MPS automatic deployment | NO-GO | CPU scientific validation only. |
| External scientific publication | HOLD | Independent real-case evidence absent. |
| PR merge | HOLD | PR creation and final-head CI evidence are pending. |
