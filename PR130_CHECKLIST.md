# AD4DVAR-radar review closure checklist — PR #130

## Authority snapshot

- Review source: user-pasted additional review of merged PR #129 / v0.92.0.
- Review timestamp/time zone: 2026-08-19 JST.
- Reviewer-stated base: `main@94c39db`, package `0.92.0`.
- Verified repository base:
  `origin/main@94c39db1106b7d59e7c10eb662803ada6975ef3f`.
- Verified PR #129:
  head `0b53e2ab5ca2c39f5cb9a70dc6af9a1bede00d40`, merge commit
  `94c39db1106b7d59e7c10eb662803ada6975ef3f`, and head reachable from
  `origin/main`.
- Verified PR #129 CI: Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI
  smoke all succeeded in run `32144868593`.
- Working branch: `agent/pr130-release-activation-chain`, created exactly
  from verified `origin/main`.
- Worktree before editing: tracked files clean; user-owned `.omx/` remains
  untracked and out of scope.

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R130-001 | P1-HIGH | The durable product activation receipt preserves only host-activation authority; release approval is not cryptographically required by the ledger or operational decision. | release approval → host activation → operational decision | REPRODUCED | repository-actionable | Add a typed release-approval artifact, bind its digest and authority to activation v3, and require current release plus activation trust with distinct keys throughout issuance/restart. | Host-only forged activation, mismatched/expired/revoked release approval, same release/activation key, mutated approval, and candidate-smoke approval all fail. | ☑ | ☑ targeted + full | ☐ final-head CI |
| R130-002 | P1-HIGH | Activation sequence is only positive/unique and is not a monotonic current-head chain; a still-valid older receipt may be reused after a newer activation. | activation ledger → automatic-use authorization | REPRODUCED | repository-actionable | Add predecessor/head state, exact `head+1` append, latest-only automatic use, and new-sequence signed rollback transitions. | Sequence gaps, stale predecessors, old-receipt reuse, duplicate sequence with another receipt, and unsigned rollback all fail. | ☑ | ☑ targeted + full | ☐ final-head CI |
| R130-003 | P1-HIGH | Real-case acceptance validates generic JSON identity rather than 14 typed current contracts and cross-stage semantic edges. | evidence files → live-review eligibility | REPRODUCED | repository-actionable | Introduce current typed stage validators, add release and runtime activation stages, and verify the complete edge graph while remaining REPORT_ONLY. | Minimal fake JSON, legacy/audit contracts, invalid signatures, broken edges, and missing runtime activation fail. | ☐ | ☐ | ☐ |
| R130-004 | P1-HIGH | Independent physical-event counts trust caller labels and do not reconstruct exact preflight cohorts/cells from artifact closure. | case manifest → sample-size eligibility | REPRODUCED | repository-actionable | Derive event identity from track/native/target/time/domain/catalog evidence and reproduce every preflight cohort count. | Relabelled duplicate closures, reused native/target evidence, split tracks, out-of-cohort cases, and empty cell counts do not satisfy preflight. | ☐ | ☐ | ☐ |
| R130-005 | P1 | Observation-error bytes affect scoring but their generator, parameters, registries, censoring, source assignment, and correlation policy are not preregistered before scoring. | holdout plan → verification estimand | REPRODUCED | repository-actionable | Add `VerificationObservationErrorPlan-v1` and require deterministic stored derivation equality before evaluation/promotion. | Post-forecast quality/std/reference/calibration/censor/source-policy changes fail; deterministic replay passes byte-identically. | ☐ | ☐ | ☐ |
| R130-006 | P1 | Inverse-variance metric weighting is not a probabilistic observation likelihood and may distort NLL/CRPS/PIT when forecast spread differs. | observation error → probabilistic score | REPRODUCED | repository-actionable, diagnostic-first | Keep deterministic weighted metrics; add observation-variance convolution and censored-threshold proper-score diagnostics before any promotion use. | Gaussian variance addition and censored likelihood match analytic cases; legacy weighted diagnostics remain byte-stable. | ☐ | ☐ | ☐ |
| R130-007 | P1/P2 | Missing-state taxonomy and spatial-correlation digest are declarative; no per-cell typed state or cluster-level consumption exists. | verification tensor → physical/statistical meaning | REPRODUCED | repository-actionable | Add typed cell-state tensor and exact state/mask/weight/source invariants; consume block membership or mark it diagnostic-only. | Every state enforces its physical rule; censored cells use registered logic; unused block evidence cannot claim inferential effect. | ☐ | ☐ | ☐ |
| R130-008 | P1 | Full runtime rehash is required on operational validation but has no latency budget or kernel-backed immutable measurement session. | current-runtime proof → decision deadline | REPRODUCED | repository-actionable plus platform evidence | Add a startup full measurement/session contract, immutable-mount measurement binding, expiry/mutation fail-close, and latency benchmark; do not add an unbacked TTL cache. | Runtime/mount/native-library mutation and expiry fail; p99 is measured against the decision guard budget. | ☐ | ☐ | ☐ |
| R130-009 | P2 | Historical authenticity and current automatic usability are conflated, so expired/revoked runtime evidence can make historical artifacts unreadable. | durable audit → current deployment usability | REPRODUCED | repository-actionable | Split historical verification from current automatic-use validation and return an explicitly non-usable audit type. | Expired historical evidence remains authenticity-readable but cannot be selected or deployed. | ☐ | ☐ | ☐ |
| R130-010 | P2 | `PR129_CHECKLIST.md` still reports the pre-merge head, pending CI, old local count, and open PR state; its title is repository-inaccurate. | checked-in governance evidence → audit | REPRODUCED | repository-actionable | Separate pre-merge and post-merge evidence and add deterministic GitHub-derived final evidence without rewriting historical facts. | Final evidence names merge/head/run/counts exactly and a consistency validator rejects stale state. | ☐ | ☐ | ☐ |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S130-001 | Runtime closure v2 performs exhaustive active-root, interpreter, stdlib, native-library, ownership, and safe-open validation. | `src/advar/runtime_closure.py`; PR #129 Wheel/CLI second-environment reproduction. | Preserve import-hook/bytecode/shadow/extra-distribution/symlink/mutation tests. |
| S130-002 | Runtime activation identity reaches operational decisions and `forecast-run-v67`, with current-runtime replay. | `DeploymentRuntimeActivationReceipt-v2`, ledger schema 41, run loader. | Preserve exact runtime, host, instance, expiry, and restart equality tests. |
| S130-003 | Hash-locked CPU wheelhouse and offline second-environment installation are reproducible. | PR #129 run `32144868593`. | Preserve `--no-index`, hash equality, missing/extra wheel, and CPU-only checks. |
| S130-004 | Observation quality and inverse-variance reliability weight affect actual deterministic scoring. | Verification bundle and promotion evaluation tests. | Preserve zero invalid weight and deterministic weighted score tests. |
| S130-005 | Acceptance remains non-authorizing even when its report-only matrix is complete. | `authorizes_deployment: false`. | Preserve unconditional non-authorization until a separate protected authority is implemented. |
| S130-006 | No new numerical-core P0 was reproduced. | PR #129 CPU suites and current static review. | Preserve PCG, Hessian action, positivity, transport budget, and derivative-session suites. |

## External actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X130-001 | Radar-data and verification owner | Supply legally usable fixed native radar cases, independent observations, physical tracks, calibration/error registries, and exact sample-size cohorts. | No qualifying real-case corpus is committed. | suppressed shadow, canary, LIVE, publication | OPEN |
| X130-002 | Protected release/deployment owner | Provision distinct current release, host-activation, operational-decision, and acceptance authorities plus immutable deployment staging. | Repository CI owns only ephemeral candidate-smoke keys. | deployable bundle, canary, LIVE, signed external certification | OPEN |
| X130-003 | Deployment platform owner | Provide and attest an immutable runtime measurement mechanism such as read-only image/mount plus fs-verity/IMA-equivalent measurement and latency SLO. | Repository code cannot establish host kernel policy. | runtime hot path, canary, LIVE | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [ ] Every repository-actionable P0/P1/P2 is fixed or evidence-disproved.
- [x] Every fix implemented in PR #130A has a targeted regression test.
- [x] PR #130A adjacent and broad suites pass locally.
- [x] PR #130A schema, digest preimage, producer, consumer, CLI, and workflow
  generations are synchronized locally.
- [ ] PR head equals the reported pushed commit.
- [ ] CI failures are classified as code or external infrastructure/policy.
- [x] Research, suppressed shadow, canary, LIVE, and external publication have separate decisions.
- [x] External actions remain visible with owner and required action.
- [x] Merge remains HOLD unless explicitly authorized.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Development and offline research | GO | Numerical/provenance/runtime integrity paths remain usable; new deployment authorization work is fail-closed. |
| Candidate-smoke bundle/runtime CI | GO | Hash-locked offline reproduction remains a non-deployable test authority. |
| Publication-suppressed shadow | HOLD | Requires R130-001 through R130-010 plus X130-001 evidence. |
| Canary | HOLD | Requires repository closure plus X130-001 through X130-003. |
| State-advancing LIVE | HOLD | Release 2-of-2, latest-head rollback prevention, semantic evidence, statistics, and external authorities remain incomplete. |
| External publication | HOLD | Requires signed semantic real-case evidence and independent operational certification. |
| MPS automatic scoring/deployment | NO-GO | Current operational contract remains CPU-only. |
| PR merge | HOLD | No PR #130 exists yet; merge requires explicit authorization after final-head CI. |

## PR #130A local evidence

- `DeploymentBundleReleaseApproval-v1` is independently signed by a root-
  approved release authority and is embedded by digest and payload through
  runtime activation v3, operational certificate v8, deployment decision v19,
  lineage v19, and `forecast-run-v68`.
- Schema 42 stores immutable release approvals, append-only activation receipts,
  and a per-deployment current head. Inserts require exact `head + 1` and exact
  predecessor; automatic use requires the selected receipt to equal the head.
  Returning to older runtime bytes requires a new signed sequence with an
  explicit rollback-reason digest.
- Product, CLI, and candidate-smoke flows use distinct release and runtime
  authorities. Same-key role reuse, release mutation, expired/revoked approval,
  sequence gaps, head rewind/delete, stale receipt reuse, and unsigned rollback
  are covered by adversarial regressions.
- Local full CPU suite: 777 passed, 410 subtests passed, 18 existing TorchScript
  deprecation warnings in 2,103.17 seconds. The eight failures in the first
  diagnostic run were two stale schema expectations and one shared selector
  fixture omission; the corrected eight-node rerun passed before the clean full
  rerun.
- Static evidence: basedpyright error-level 0 errors/0 warnings, compileall
  passed, CI YAML parsed, and `git diff --check` passed. Final-head GitHub CI is
  not yet available.
