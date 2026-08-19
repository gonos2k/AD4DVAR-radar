# AD4DVAR-radar review closure checklist — PR #129

## Authority snapshot

- Review source: user-pasted additional review of merged PR #128 / v0.91.0.
- Review timestamp/time zone: 2026-08-18 JST.
- Reviewer-stated base: `main@fc071bf2c9c607707c87a2543a0957db36520416`.
- Verified repository base: `origin/main@fc071bf2c9c607707c87a2543a0957db36520416`.
- Verified base tree: `c9c5adee6ecbb7447a76223e352d97dd2d078db2`.
- Verified PR #128 merge: merge commit `fc071bf2c9c607707c87a2543a0957db36520416`
  has PR head `fc69650290eae3dbb10e0d07b13f6cb45db752e3` as its second parent;
  the reviewed tree is reachable from `origin/main`.
- Working branch: `agent/pr129-runtime-ledger-semantic-acceptance`.
- Pre-delivery snapshot: no PR #129 or candidate commit existed when this
  checklist was opened.
- Worktree before editing: tracked files clean; user-owned `.omx/` remains
  untracked and out of scope.
- Initial CI snapshot: PR #128 required run
  [32109448015](https://github.com/gonos2k/AD4DVAR-radar/actions/runs/32109448015)
  succeeded; merge-push run
  [32116507805](https://github.com/gonos2k/AD4DVAR-radar/actions/runs/32116507805)
  was in progress and was infrastructure evidence, not a substitute for PR #129 tests.
- Post-merge evidence (GitHub API revalidated 2026-08-19 JST): PR
  [#129](https://github.com/gonos2k/AD4DVAR-radar/pull/129) final head
  `0b53e2ab5ca2c39f5cb9a70dc6af9a1bede00d40` was merged as
  `94c39db1106b7d59e7c10eb662803ada6975ef3f`.
- Final-head required run
  [32144868593](https://github.com/gonos2k/AD4DVAR-radar/actions/runs/32144868593)
  completed successfully: Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI
  smoke all reported `SUCCESS`.

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R129-001 | P1-HIGH | Runtime-tree v1 hashes only expected distribution-owned files and excludes executable `.pyc`; unowned import hooks, shadow modules, unexpected distributions, interpreter/stdlib identity, and permissions are outside the closure. | installed bundle → executing process | REPRODUCED | repository-actionable | Introduce runtime-tree v2 with exhaustive active-root census, forbidden import-hook/bytecode policy, exact distribution inventory, interpreter/stdlib/extension identity, and deployable permission checks. | Unowned `sitecustomize.py`, executable `.pth`, `.pyc`, extra distribution, shadow package, writable runtime, and changed interpreter/stdlib identity fail. | ☑ | ☑ | ☑ final-head CI |
| R129-002 | P1-HIGH | Runtime activation receipt is an external script artifact and is not a mandatory typed ledger relation for operational decisions or restart loads. | verified host runtime → ledger decision authority | REPRODUCED | repository-actionable | Add typed runtime activation/trust contracts, ledger activation table, decision/certificate/run binding, expiry/revocation, and launch-time tree revalidation. | A decision without a current exact runtime activation, with an expired/revoked receipt, or after runtime mutation fails issuance and restart. | ☑ | ☑ | ☑ final-head CI |
| R129-003 | P1-HIGH | Real-case harness accepts 13 generic self-declared JSON artifacts rather than current typed product artifacts and cross-stage semantics. | evidence files → acceptance eligibility | REPRODUCED | repository-actionable | Replace generic digest inspection with a 14-stage current-contract validator registry, including runtime activation, signature/trust checks, and exact cross-stage edge graph. | Minimal fake JSON, legacy/audit contracts, broken edge digests, invalid signatures, and omitted runtime activation fail before eligibility. | ☐ | ☐ | ☐ |
| R129-004 | P1-HIGH | Independent physical-event count uses caller-supplied labels and is not derived from the native/target/track closure or exact preflight cohorts. | acceptance manifest → sample-size eligibility | REPRODUCED | repository-actionable | Derive typed event identity from physical track, native acquisition closure, target source/time, interval, and spatial domain; verify global closure uniqueness and exact preflight cohort/cell membership. | Relabeled identical closures, reused native/target sources, split tracks, and out-of-cohort cases do not increase the independent count. | ☐ | ☐ | ☐ |
| R129-005 | P1 | Observation-error bytes are lineage-bound but the generator, parameters, registries, censoring, source assignment, and correlation policy are not preregistered before scoring. | holdout plan → verification weights | REPRODUCED | repository-actionable | Add a typed preregistered observation-error plan and require deterministic derivation equality in target/scoring/promotion paths. | Post-forecast quality/reference-std/registry/censor/source-policy changes fail; deterministic replay is byte-identical. | ☐ | ☐ | ☐ |
| R129-006 | P1/P2 | Missing-state taxonomy and spatial-correlation digest are declarative metadata without a per-cell state tensor or cluster-level statistical use. | verification observation → metric/statistical inference | REPRODUCED | repository-actionable | Add typed per-cell observation states with exact validity/weight/source invariants and either consume typed block membership in clustered inference or mark it diagnostic-only. | Each state enforces its physical mask/weight rule; censored cells use the registered policy; declared statistical block evidence must be consumed. | ☐ | ☐ | ☐ |
| R129-007 | P2 | Candidate CI uses the same Ed25519 key for bundle and runtime activation roles; deployable role/key separation is not enforced by the receipt contract. | release approval → host activation | REPRODUCED | repository-actionable | Bind signer roles and require distinct release, host-activation, and operational-decision authorities for deployable mode. | Same public key or role reused across deployable stages fails. | ☑ | ☑ | ☑ final-head CI |
| R129-008 | P2 | Acceptance evidence may remain owner-writable, so its guarantee is point-in-time hash equality rather than immutable certification evidence. | report-only artifact store → external certification | REPRODUCED | repository-actionable with external deployment policy | Preserve owner-writable REPORT_ONLY inputs, but require immutable/root-owned staging and signed acceptance authority for certification mode. | REPORT_ONLY accepts controlled owner-write; certification rejects writable file/ancestry and mutation after snapshot. | ☐ | ☐ | ☐ |
| R129-009 | P2 | Report `verified_at` copies submitter `created_at`; validator observation and independent review authorization chronology are absent. | submitter evidence → review chronology | REPRODUCED | repository-actionable | Separate `evidence_created_at`, trusted `validator_observed_at`, and signed `review_authorized_at`; sign the report with an acceptance authority. | Backdated submitter time cannot become validator time; signature and chronology mutations fail. | ☐ | ☐ | ☐ |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S129-001 | Dependency artifact closure is exact-hash and network-disabled. | Bundle v3 wheelhouse validator and second-environment install. | Preserve missing/extra/version/hash and `--no-index` regressions. |
| S129-002 | Observation quality and inverse-variance scale affect actual scoring weights. | Verification observation-error contract and weighted target/evaluation path. | Preserve zero-weight invalid cells and deterministic weighted score tests. |
| S129-003 | Acceptance evidence cannot directly authorize deployment. | Reports always set `authorizes_deployment: false`. | Preserve a hard REPORT_ONLY mode independent of completeness. |
| S129-004 | CPU numerical core has no newly reproduced P0. | PR #128 full CPU suite and current static review. | Preserve PCG, positivity, transport-budget, and derivative-session suites. |

## External actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X129-001 | Radar-data and verification owner | Supply legally usable fixed native radar cases, independent observations, physical tracks, calibration registries, and required sample-size cohorts. | No real-case corpus exists in the repository. | suppressed shadow evidence, canary, LIVE, external publication | OPEN |
| X129-002 | Protected release/deployment owner | Provision distinct protected release, host-activation, operational-decision, and acceptance-review authorities plus immutable deployment staging. | Repository CI owns only ephemeral candidate-smoke keys. | deployable bundle, canary, LIVE, signed external certification | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [ ] Every repository-actionable P0/P1/P2 is fixed or evidence-disproved.
- [x] Every fix implemented in PR #129A has a targeted regression test.
- [x] Adjacent and broad suites pass.
- [x] PR #129A schema, receipt, certificate, decision, lineage, run, and output generations are synchronized locally.
- [x] PR #129A bundle/runtime manifests and distribution documents are synchronized locally.
- [x] PR #129 base/head refs were verified at delivery.
- [x] Final-head CI completed without an unclassified failure.
- [x] Research, suppressed shadow, canary, LIVE, and external publication have separate decisions.
- [x] External actions remain visible with owner and required action.
- [x] The pre-merge HOLD was preserved until explicit merge authorization;
  final merge evidence is recorded separately above.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Development and offline research | GO | Current CPU numerical and provenance paths; new acceptance/runtime work remains report-only. |
| Publication-suppressed shadow | HOLD | Requires R129-001 through R129-009 and X129-001 evidence. |
| Canary | HOLD | Requires repository closure plus X129-001 and X129-002. |
| State-advancing LIVE | HOLD | Runtime activation is ledger-enforced in PR #129A; R129-003 through R129-006, R129-008, R129-009, X129-001, and X129-002 remain open. |
| External publication | HOLD | Requires signed semantic acceptance evidence and protected authorities. |
| MPS automatic scoring/deployment | NO-GO | Current operational contract remains CPU-only. |
| PR merge | MERGED | Final head `0b53e2ab5ca2c39f5cb9a70dc6af9a1bede00d40`; required run `32144868593` succeeded; merge commit `94c39db1106b7d59e7c10eb662803ada6975ef3f`. |

## PR #129A local evidence

- Runtime closure v2 performs exhaustive import-root census, active `sys.path`
  allowlisting, bytecode/import-hook rejection, exact distribution inventory,
  interpreter/stdlib/native-library hashing, and deployable ownership checks.
- `DeploymentRuntimeActivationReceipt-v2` is typed, role-bound, expiry/current-
  revocation checked, and linked through schema 41, operational certificate v7,
  deployment decision v18, lineage v18, and `forecast-run-v67`.
- Operational issuance, committed reads, selection, and v67 restart compare the
  current process closure with the signed deployable receipt; candidate-smoke
  receipts are rejected.
- Separate bundle and runtime-activation keys are used in CI, and duplicate
  public-key aliases or multi-role deployment authorities are rejected.
- Local evidence: runtime/bundle 8 passed + 4 subtests; ledger 43 passed + 27
  subtests; run-artifact/CLI/ledger adjacent group reached 98 passed before the
  corrected schema assertion, followed by a clean ledger rerun; promotion was
  completed in ordered segments with every collected node passing; basedpyright
  0 errors; YAML/actionlint, lock synchronization, diff check, and secret-pattern
  scan passed.
- The earlier local full-suite snapshot was 772 passed, 410 subtests, and 18
  warnings in 2,110.13 seconds. The final PR report additionally recorded 776
  local passes; final-head GitHub Python 3.10/3.12 jobs each recorded 774
  passed, 2 skipped, and 410 subtests. LIVE, canary, suppressed shadow, and
  external publication remain HOLD.
