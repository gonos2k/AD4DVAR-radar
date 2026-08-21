# PR #134 follow-up / PR #135 confirmatory observation-semantics checklist

## Authority snapshot

- Review source: user-pasted additional review of merged PR #134.
- Review timestamp/time zone: 2026-08-21 / Asia-Tokyo.
- Reviewer-stated head/merge: `ca997070b606a5ec2088a0749db6289590b4259a` / `15a671c28230ee90655649f85a9c64cace8b2383`.
- Verified repository base: `origin/main@15a671c28230ee90655649f85a9c64cace8b2383`; reviewed head is reachable from `origin/main`.
- Verified PR/CI: PR #134 MERGED; exact-head run `32450937446` completed SUCCESS for Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI smoke.
- Follow-up branch: `agent/pr135-confirmatory-observation-semantics`, created exactly from `origin/main`.
- Initial worktree state: clean tracked tree; pre-existing untracked `.omx/` is user-owned and excluded.
- Scope boundary: reproducible offline scientific validation only; operational deployment remains NO-GO/out of scope.
- Merge authority: HOLD; no merge is authorized by this checklist.

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R134-012 | P1-SCIENTIFIC | Absolute acquisition timestamps and cell offset tensors are not linked, so temporal age can disagree with signed chronology. | Asynchronous mosaic observation time | CLOSED | repository-actionable | Bind per-source nominal acquisition times and define cell time/age algebra in the current signed source/evidence contract; recompute age product-side. | 600-second absolute age plus zero local offset yields age 600; zero absolute age plus -60-second local offset yields age 60; timestamp-only and offset-only relabeling fail. | ☑ | ☑ | pending |
| R134-013 | P1-SCIENTIFIC | Product-derived censoring makes `OBSERVED_CLEAR` unreachable in the current taxonomy. | Clear/no-echo/censored estimand | CLOSED | repository-actionable | Encode signed native report-kind evidence that distinguishes detected echo, confirmed clear, and below-detection censoring. | Confirmed-clear and censored inputs reach distinct states and impose their intended false-alarm/censored penalties; impossible combinations fail. | ☑ | ☑ | pending |
| R134-014 | P2-PROVENANCE | Cold replay may validate detection/censor policy only for selected source rows. | Canonical source-cube provenance | CLOSED | repository-actionable | Add one full-cube `validate_against_registry()` path and call it from issue, mask derivation, and cold reconstruction. | Re-signed mutation of a never-selected source limit/censor row fails in producer, derivation, and cold replay. | ☑ | ☑ | pending |
| R134-015 | P2-INTEGRITY | A current manifest may declare a valid but unreferenced tensor shard. | Durable replay exact byte closure | CLOSED | repository-actionable | Require exact equality between declared and referenced shard digest sets and hash every declared member before byte verification. | Unreferenced NPZ, arbitrary extra file, and manifest-only shard digest fail; shared content shard remains allowed. | ☑ | ☑ | pending |
| R134-016 | P2-SCIENTIFIC | Stale acquisition is collapsed into attenuation/QC invalidity. | Missingness and exclusion diagnosis | CLOSED | repository-actionable | Preserve temporal-valid and attenuation-QC masks separately and emit a distinct stale-acquisition state/reason. | Stale, attenuation-invalid, beam-blocked, and source-missing cells remain distinguishable after durable replay. | ☑ | ☑ | pending |
| R134-017 | P2-SCIENTIFIC | Detection limit is one scalar per radar and lacks range/elevation/time dependence. | Real-radar censor threshold model | CLOSED for preregistered baseline; empirical calibration remains OPEN as X135-002 | repository-actionable baseline model; empirical calibration remains external | Preregister a deterministic low-dimensional source-specific detection-limit function and derive the full field product-side without double-counting attenuation uncertainty. | Limit varies deterministically with registered range/elevation parameters; coefficient or evidence relabeling fails; scalar legacy inputs are audit-only. | ☑ | ☑ | pending |
| R134-018 | P2-GOVERNANCE | PR #134 body still says merge HOLD after the PR was merged. | Review authority record | CLOSED | repository-actionable external metadata | Synchronize the merged head, merge commit, exact-head run, and three successful jobs in PR #134 metadata and the follow-up checklist. | GitHub PR state/body and repository evidence report the same merged head, merge commit, and CI run. | ☑ | ☑ | external metadata complete |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S135-001 | Legacy v2/v3 observation derivation is audit-only. | Merged PR #134 code and direct regression. | Never permit legacy inputs to enter a current derivation factory. |
| S135-002 | v13/v14 replay directories audit-load by exact generation. | Merged PR #134 durable fixtures. | Preserve exact member sets and exact-type current checks. |
| S135-003 | Replay verification levels are distinct and source/runtime exact. | Current byte/reconstruction/semantic/full replay flags. | Keep source/runtime mismatch audit-loadable but semantically false. |
| S135-004 | Current tensor replay is content-addressed and lazy. | Replay v15 shard store and adversarial tests. | Preserve content deduplication while tightening exact shard closure. |
| S135-005 | Required CI is centered on scientific CPU/package evidence. | Merged PR #134 workflow; package smoke completes without deployment activation. | Do not reintroduce operational bundle/activation as required PR evidence. |

## Implementation evidence

- Package/scientific generation: `0.100.0`; observation-error plan v6; radar source/registry v3; signed source identity v2; mask evidence/artifact v4; derivation input v5; observation-error artifact v5; error contract v8; verification bundle v11; scoring replay bundle/method v16; semantic generation v14; scoring case v15.
- Chronology contract: per-source nominal acquisition timestamps are signed; cell age is recomputed as verification valid time minus nominal acquisition time minus the nonpositive cell-local offset. The realized absolute-age tensor is included in the error-contract and verification-bundle identities.
- Observation estimand: source-signed report kind distinguishes detected echo, confirmed clear, and below-detection censoring. Current derivation emits a separate `STALE_ACQUISITION` state and retains attenuation-QC validity independently.
- Detection threshold: current radar registry preregisters a deterministic base + range-quadratic + elevation-excess model. The product derives the complete source cube and excludes attenuation from the threshold function to avoid double penalization.
- Durable replay: current v16 preserves absolute age, report kind, temporal-valid, and confirmed-clear tensors; v15 is byte-audit only. Declared shard digests must exactly equal referenced shard digests while identical tensor content may share one shard.
- Governance: PR #134 body now records merged head `ca997070b606a5ec2088a0749db6289590b4259a`, merge commit `15a671c28230ee90655649f85a9c64cace8b2383`, exact-head run `32450937446`, and all three required jobs as SUCCESS.
- Local non-deployment scientific verification: **777 passed / 437 subtests** total: sensitivity **77 / 121**, promotion **231 / 22**, ledger **43 / 27**, numerical/runtime/artifact suites **400 / 245**, and acceptance/CLI **26 / 22**. CI-authoritative source basedpyright reports **0 errors**; changed-module bytecode compilation, `git diff --check`, and the `0.100.0` wheel/sdist build pass. Deployment-bundle and runtime-closure suites are intentionally excluded because operational deployment is out of scope.

## External scientific actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X135-001 | Scientific study owner | Supply legally usable independent per-radar observations and run the preregistered confirmatory protocol across the required independent physical-event cohort. | Repository contains no independent real-radar cohort satisfying the protocol. | confirmatory real-radar claim / external publication | OPEN |
| X135-002 | Radar science owner | Calibrate and justify the temporal-age and spatial detection-limit parameters against independent radar characteristics and regimes. | Repository can define and replay a baseline model but cannot infer empirical calibration authority from synthetic fixtures. | final real-radar parameter approval / external publication | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P0/P1/P2 is fixed or evidence-disproved.
- [x] Every fix has a targeted regression test.
- [x] Adjacent and broad non-deployment scientific suites pass.
- [x] Every changed schema, attested source, digest preimage, and required nested field has regenerated producer and consumer evidence.
- [x] Evidence/manifests/distribution documents are synchronized when affected.
- [ ] PR head equals the reported pushed commit.
- [ ] CI failures are classified as code or external infrastructure/policy.
- [x] Offline research, real-radar confirmatory claims, external publication, and operational deployment have separate decisions.
- [x] External scientific evidence remains visible with owner and required action.
- [x] Merge remains HOLD unless explicitly authorized.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Development and offline numerical/synthetic research | GO | PR #134 exact-head CI is green; no new numerical P0 was reported. |
| Exploratory real-radar analysis | CONDITIONAL GO | Must retain full source/time/state evidence and make no confirmatory skill claim. |
| Confirmatory real-radar scientific claim | HOLD | Repository contracts are closed; X135-001/002 independent cohort and empirical parameter evidence remain open. |
| External publication | HOLD | Requires repository closure plus independent cohort and parameter evidence. |
| Operational deployment | NO-GO / out of scope | Project is explicitly centered on scientific validation. |
| PR #135 merge | HOLD | No merge authorization; implementation is complete and exact-head CI is pending. |
