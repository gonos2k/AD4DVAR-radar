# PR #128 dROADi review closure checklist

## Authority snapshot

- Review source: user-pasted additional review, latest `main`, v0.90.0.
- Review timestamp/time zone: 2026-08-18 JST.
- Reviewer-stated base/head: latest `main`; no commit SHA stated.
- Verified repository base: `origin/main@90e0520bca51923ae4fdf4bd0468928e018aab72`.
- Verified base tree: `49ab7c0fa21a3248fb2c97fdaef2d6afde033ef1`.
- Working branch: `agent/pr128-hermetic-deployment-real-case-evidence`.
- Candidate package version: `0.91.0`.
- Verified PR: [#128](https://github.com/gonos2k/AD4DVAR-radar/pull/128),
  open and non-draft.
- Verified candidate head: `3c9fb9c61bef7567f993997d2daf690356d9f7b3`.
- Verified candidate tree: `4fbd6e64d5b9a170260e2e25c09935803035d02d`.
- Verified GitHub base: `main@90e0520bca51923ae4fdf4bd0468928e018aab72`.
- Candidate CI run:
  [32102928474](https://github.com/gonos2k/AD4DVAR-radar/actions/runs/32102928474),
  three required checks in progress at checklist synchronization time.
- Worktree state before editing: tracked files clean; user-owned `.omx/` remains
  untracked and out of scope.
- CI snapshot: main run
  [32033357234](https://github.com/gonos2k/AD4DVAR-radar/actions/runs/32033357234)
  succeeded at `90e0520bca51923ae4fdf4bd0468928e018aab72`.

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R128-001 | P1 | The signed deployment bundle retains hashes but not the selected dependency wheel bytes. | release bundle → offline installer | REPRODUCED | repository-actionable | Add a manifest-bound Linux CPU wheelhouse closure selected by the exact lock. | Missing, extra, wrong-version, and one-byte-mutated wheels fail; network-disabled install succeeds. | ☑ | ☑ unit + wheel build; offline CI pending | ☐ |
| R128-002 | P1 | Installation attestation does not prove the exact importable runtime file tree or reproduce it across clean installs. | offline install → activation identity | REPRODUCED | repository-actionable | Add deterministic runtime-tree attestation and verification plus a signed host activation receipt, transitively bound by the signed bundle digest. | Two clean offline installs have the same runtime digest; ABI/platform, identity relabel, and installed-byte mutations fail. | ☑ | ☑ unit; two-venv CI pending | ☐ |
| R128-003 | P1 | No one-command, mock-free, publication-suppressed acceptance harness proves the native-input-to-activation chain on fixed cases. | operational evidence | REPRODUCED | repository-actionable | Add a report-only acceptance manifest/runner that consumes typed product stage artifacts and emits a content-addressed report without state-advancing authority. | Manifest/file mutation, direct digest injection, missing stage, reused event, publication-capable mode, and a count not reproduced from the typed preflight fail. | ☑ | ☑ 4 harness integrity tests | ☐ |
| R128-004 | P1 | Independent real-radar event evidence has not been produced on this clean local checkout. | scientific/operational evidence | EXTERNAL_BLOCKED | external-action | Provide licensed/redacted fixed native radar cases and run the report-only harness until the existing sample-size preflight is satisfied. | Native bytes through activation replay with no mocks, direct SQL, or private constructors. | n/a | ☐ | n/a |
| R128-005 | P1 | Verification target byte provenance does not itself encode calibration, observability, censoring, correlated-error, and mosaic-source error semantics. | verification target → promotion statistics | REPRODUCED | repository-actionable | Add a typed observation-error contract and bind it to the exact scoring target identity, evaluation, metric weights, and replay generation. | Low-quality/unobserved cells cannot contribute; error-scale sensitivity and source-specific mosaic rules are exact and replayed. | ☑ | ☑ | ☐ |
| R128-006 | P2 | Current/legacy trust domains are concentrated in large `promotion.py` and `ledger.py` modules. | maintainability/version cascade | REPRODUCED | repository-actionable, gated | Publish an enforced domain routing/refactor gate now; defer code movement until external acceptance evidence exists, as the review requires. | The boundary document forbids legacy production re-entry and requires byte-identical payload/digest/schema/replay fixtures for the later split. | ☑ | ☑ document audit | ☐ |
| R128-007 | P2 | Wheel metadata ranges alone do not recreate the controlled deployment closure. | generic `pip install` | ALREADY_CLOSED | already-closed within controlled deployment | Keep generic resolver installs explicitly outside deployment authority; make the new offline bundle command the only controlled install path. | README uses `--no-index` wheelhouse install and verifier rejects missing wheelhouse/runtime attestations for current bundles. | ☑ | ☑ | ☐ |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S128-001 | Current operational input provenance is mandatory and root-trust-bound. | `forecast-run-v66` plus operational provenance plan, raw resolution, analysis derivation, and activation receipt. | Existing durable-load and direct-SQL adversarial tests. |
| S128-002 | Native acquisition, missing observation, and grid-product identities are distinct. | Current native/raw-resolution types and mosaic/missing replay tests. | Preserve typed identity equality and missing-receipt tests. |
| S128-003 | Training feature, target, normalization, split, and trust lineage are byte-backed. | PR #126 contracts and 757-test evidence. | Preserve current generation and audit-only legacy loaders. |
| S128-004 | Provenance recovery and terminal publication fail closed. | Prepared/active/expired reconciliation and writer-lock authorization receipts. | Preserve crash-fault and receipt-chain tests. |
| S128-005 | Derivative preflight and VJP execute in the same sealed session. | Current `_derivative_session` tests. | Preserve `.data` mutation and completion revalidation regressions. |

## External evidence actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X128-001 | Radar-data/product owner | Supply legally usable fixed real-case native bytes, source metadata, and independent verification observations. | No real-case corpus is tracked in the repository. | state-advancing LIVE and unsupervised production promotion | OPEN |
| X128-002 | Protected release owner | Produce a deployable-mode bundle with the out-of-band release key in a root-owned staging area. | Repository CI can only create `candidate-smoke` bundles. | signed production release and external publication | OPEN |

## Acceptance matrix

- [ ] Single-site normal three-frame native input replays end to end.
- [ ] Late frame produces the exact missing/deadline/publication decision.
- [ ] Complete outage remains typed absence and cannot become clear sky.
- [ ] Mosaic handoff preserves time-resolved source and error contracts.
- [ ] Repacked duplicate acquisition retains one native identity.
- [ ] Delayed/corrupt target source fails before promotion.
- [ ] Ingestor, processor, and target-source revocation fail closed.
- [ ] Provenance rename/DB/activation crash points reconcile idempotently.
- [ ] Deployment activation exposes only old/old or new/new state.
- [ ] Clean network-disabled install uses only manifest-bound wheel bytes.
- [ ] Restart replay preserves forecast, selection, certificate, bundle, and runtime digests.

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P0/P1/P2 is fixed, safely gated, or evidence-disproved.
- [x] Every code fix has a targeted regression test.
- [x] Adjacent and broad suites pass (final full rerun: 763 passed + 406
  subtests).
- [x] Changed verification/replay schemas, digest preimages, producers, consumers, and audit-only predecessor loaders are updated.
- [x] Evidence, manifests, distribution documents, and workflow expectations are synchronized.
- [x] PR head equals the reported pushed commit.
- [ ] CI failures are classified as code or external infrastructure/policy.
- [x] Research, suppressed shadow, canary, LIVE, and external publication have separate decisions.
- [x] External evidence requirements remain visible with owner and blocked stage.
- [x] Merge remains HOLD unless explicitly authorized.

## Local verification evidence

- `basedpyright src/advar`: 0 errors (pre-existing strict warnings remain).
- Focused security/contract suite: 10 passed, 6 subtests passed.
- Acceptance harness after typed preflight binding: 4 passed.
- Initial full CPU suite: 760 passed, 406 subtests passed, with two test-only
  call sites missing the new mandatory observation-error weight.
- Corrected failure nodes: 2 passed.
- Final full CPU suite after all code changes: 763 passed, 406 subtests passed
  in 35 minutes 22 seconds; failure/error 0.
- Wheel build: `advar_radar_nowcast-0.91.0-py3-none-any.whl`; metadata name and
  version verified.
- Dependency-lock synchronization, YAML parse, actionlint built-in checks and
  `git diff --check`: passed.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Development and offline research | GO | Current CPU numerical and provenance contracts. |
| Publication-suppressed shadow | HOLD | Harness is implemented; X128-001 fixed real-case manifest/evidence remains absent. |
| Canary | HOLD | Requires hermetic install plus external real-case evidence. |
| State-advancing LIVE | HOLD | Requires X128-001 and a protected deployable bundle. |
| External publication | HOLD | Requires X128-001, X128-002, and sample-size preflight completion. |
| PR merge | HOLD | PR #128 is open; required CI and separate trust-root review are pending. |
