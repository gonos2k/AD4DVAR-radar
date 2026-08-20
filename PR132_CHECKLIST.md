# PR #132 review / PR #133 follow-up scientific observation-error checklist

## Authority snapshot

- Review source: user-supplied focused review of merged PR #131
- Review timestamp/time zone: 2026-08-20 Asia/Tokyo
- Reviewer-stated base/head: PR #131 latest changes
- Verified repository base for the second follow-up: `origin/main@35c9494be0d172aa5f4f4822186c1d918965954f`
- Verified PR #131/head/tree: merged PR #131; head `cae966315fe9f7351fb55b9bcb89e5e1a584f491`; merge `e696e4343a1a5a6f707a6de9bc3e75e13e20a8a8`; identical tree `14a5bd16d905dd3a8037e7de55473138ca6cd37e`
- Verified PR #132 merge: head `2a33d085ae802700db7116b840b822994c37331a`; merge `35c9494be0d172aa5f4f4822186c1d918965954f`; reviewed head is reachable from `origin/main` and both trees are `6ac1544b2c6e9050dcb67ebd818ec57cab6e4ce1`
- Worktree state: second-follow-up branch `agent/pr133-source-specific-mosaic-replay` starts exactly at the verified PR #132 merge; only the nine intentional follow-up files are modified; unrelated `.omx/` excluded and untouched
- CI snapshot: PR #132 follow-up head `2a33d085ae802700db7116b840b822994c37331a` passed Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI in run `32353005357`
- Scope boundary: reproducible offline scientific validation only; confirmatory real-case and publication claims remain HOLD; operational deployment remains out of scope
- Local contract version: package `0.97.0`; observation-error plan v3, source-specific signed mask evidence/source identity v1, mask derivation artifact v1, observation-error derivation artifact v2, realized contract v5, verification bundle v8, scoring replay bundle/method v13, semantic generation v11, scoring case v12, holdout scoring artifact v13; forecast/deployment/ledger generations unchanged
- Local targeted evidence: source-specific confirmatory replay 1 test with 18 adversarial subtests; semantic replay integration 4 tests with 2 subtests; failure 0
- Local adjacent evidence: sensitivity 76 passed with 113 subtests; run-artifact 48 passed with 28 subtests; acceptance 5 passed; failure 0
- Static/package evidence: authoritative basedpyright 0 errors; `git diff --check` clean; isolated 0.97.0 sdist/wheel build succeeded; public exports verified
- PR #132 first-head evidence: PR `#132`; head `6335e2a1f430a612430f13852b8fb1e2fe6cda7a`; tree `2c37a06aee08b1edea65d7afda7845bdc2ed15f8`; Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI required checks all succeeded
- Follow-up review source: user-supplied focused review of PR #132; 2026-08-20 Asia/Tokyo
- Second follow-up review source: user-supplied focused review of head `2a33d085ae802700db7116b840b822994c37331a`; 2026-08-20 Asia/Tokyo

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R132-001 | P1-HIGH | A pre-registered observation-error plan does not deterministically produce the realized quality, standard-deviation, cell-state, and source-map tensors. | plan → realized scientific tensor lineage | REPRODUCED, CLOSED ON FIRST PR HEAD: plan v2 pins the product algorithm; typed raw identity is derived from exact inputs; artifact replay regenerates every tensor; direct tensors remain exploratory-only | repository-actionable | Add a typed deterministic derivation artifact and replay the product-owned derivation byte-for-byte; mark direct tensor construction exploratory-only. | Quality/std/state/source mutation fails; identical inputs produce identical raw/artifact digests; algorithm substitution fails; v7 rejects exploratory contracts. | ☑ | ☑ targeted + adjacent | ☑ first-head CI |
| R132-002 | P1-HIGH | Mosaic source indices are not bounded by or ordered against a typed radar/calibration registry. | mosaic source map → radar identity/calibration parameters | REPRODUCED, CLOSED ON FIRST PR HEAD: ordered registry owns index→radar→calibration→quality/std and validates `-1 <= index < radar_count` | repository-actionable | Add a canonical ordered mosaic source registry and bind every source index to radar identity, calibration epoch, quality, and observation-error parameters. | `index == radar_count`, `999`, registry reorder, and cross-radar calibration substitution fail; exact registry replay passes. | ☑ | ☑ targeted + adjacent | ☑ first-head CI |
| R132-003 | P1-HIGH | Observation-state input masks are caller supplied and are not regenerated from the preregistered algorithms and raw verification evidence. | raw evidence + preregistered algorithms → range/beam/QC/censor/source masks | REPRODUCED, FIXED LOCALLY: plan v3 pins mask algorithms, thresholds, and verification-source authority; signed typed raw evidence deterministically generates all masks and source assignment | repository-actionable | Add typed mask-derivation artifacts whose canonical payloads are regenerated from typed raw evidence and whose exact masks are replayed before observation-error derivation. | Recomputed-digest mutations of QC, censoring, beam-blockage, source-present, and source-assignment evidence fail source attestation; all stored mask mutations fail replay. | ☑ | ☑ targeted + adjacent | ☐ follow-up PR/CI |
| R132-004 | P1-HIGH | Observation-error derivation is not directly bound to verification valid times, grid contract, radar product, typed source identity, and acquisition-time identity. | observation derivation → verification source time/grid/product identity | REPRODUCED, FIXED LOCALLY: signed typed source identity binds canonical valid/acquisition times, grid, product, native identity, and exact raw evidence; v8 requires bundle equality | repository-actionable | Introduce derivation-input v2 with canonical UTC valid times, grid/product digests, typed upstream source identity, and acquisition-time identity; require exact bundle equality. | Same tensors relabeled with different time, grid, product, native source, or acquisition identity fail at v8/source-signature validation. | ☑ | ☑ targeted + adjacent | ☐ follow-up PR/CI |
| R132-005 | P2 | One scalar quality/std pair per radar cannot represent range-, elevation-, blockage-, or attenuation-dependent observation error. | typed raw evidence → spatial observation-error field | REPRODUCED, FIXED LOCALLY: registry values remain baselines; digest-pinned spatial v2 algorithm derives per-cell quality/std from range, elevation, blockage, and attenuation evidence | repository-actionable | Preserve registry values as radar baselines and apply a product-owned, digest-pinned per-cell modulation from range, elevation, beam blockage, and attenuation-QC evidence. | Same-radar 10 km and 200 km valid cells receive different deterministic quality/std; exact replay passes. | ☑ | ☑ targeted + adjacent | ☐ follow-up PR/CI |
| R132-006 | P1-HIGH | Mosaic range, elevation, beam-blockage, and attenuation-QC evidence is shared across sources instead of being selected from the assigned radar. | selected mosaic source → source-specific spatial evidence | REPRODUCED, FIXED LOCALLY: all four fields are `[source,time,y,x]`, their source axis is signed with the ordered registry, and mask/error derivation gathers the assigned source | repository-actionable | Represent every spatial evidence field as `[source,time,y,x]`, bind the source dimension to the ordered registry, and gather only the assigned source before mask/error derivation. | Keeping the source map fixed while swapping radar A/B spatial evidence fails source attestation or changes the selected replay; registry reorder remains rejected. | ☑ | ☑ targeted | ☐ follow-up PR/CI |
| R132-007 | P1-MEDIUM | The scientific input semantics changed from caller-mask v7 to product-owned source/mask v8 while semantic scoring replay retained bundle/method v12, semantic generation v10, and case v11. | verification semantics → durable semantic scoring replay identity | REPRODUCED, FIXED LOCALLY: current replay is bundle/method v13, generation v11, case v12, scoring artifact v13; prior v12 objects decode only as typed audit artifacts | repository-actionable | Advance replay bundle/method to v13, semantic generation to v11, and scoring case to v12; retain the prior generation as explicit audit-only input. | Current replay emits only the new generation; prior v12/v10/v11 payload loads as audit-only and cannot enter current scoring/promotion. | ☑ | ☑ 3 targeted | ☐ follow-up PR/CI |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S132-001 | Generic artifact indexing cannot claim semantic E2E validation or independent sample size. | Acceptance report keeps semantic/sample-size/scientific-review/deployment booleans false and reports only declared labels. | Existing acceptance fail-closed tests remain required. |
| S132-002 | Seven observation states preserve missing, QC-invalid, blocked, censored, and unassigned semantics. | `VerificationCellState` and state/tensor invariants in `sensitivity.py`. | Preserve invalid weight/std zero rules and point-score exclusion for censored cells. |
| S132-003 | Gaussian observation-error diagnostics remain report-only. | Effective variance combines forecast and observation variance while `diagnostic_only=True`. | No promotion or deployment authority may consume this diagnostic. |
| S132-004 | The second review found no new P0 and confirmed mask replay, time/grid/product binding, and registry ordering. | Independent focused review of exact head `2a33d085…`. | Preserve the v8 confirmatory boundary while closing source-specific spatial evidence and replay generation. |

## External scientific evidence actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X132-001 | Independent scientific investigators | Supply legally usable real radar cases and independent verification observations; run the deterministic derivation/replay without mocks. | Repository cannot manufacture independent physical events. | confirmatory real-case claim / publication | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P1/P2 is fixed or evidence-disproved locally.
- [x] Every fix has a targeted regression test.
- [x] Adjacent and broad affected CPU suites pass locally on the second follow-up tree.
- [x] Public exports, contracts, README, and package generations are synchronized for the second follow-up contract.
- [ ] PR head equals the reported pushed commit.
- [ ] CI failures are classified as code or external infrastructure/policy.
- [x] Offline research, confirmatory real-case review, publication, and deployment have separate decisions.
- [x] External scientific evidence remains visible with owner and required action.
- [x] Merge remains HOLD unless explicitly authorized.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Offline numerical research | GO | CPU/synthetic regression and deterministic observation-error derivation pass locally. |
| Exploratory observation diagnostics | GO | Direct tensor construction remains explicitly exploratory-only. |
| Mosaic confirmatory evaluation | HOLD | Repository contracts are locally closed; X132-001 and exact new-head CI remain open. |
| Independent real-case skill claim | HOLD | X132-001. |
| External scientific publication | HOLD | X132-001 plus independent review. |
| Operational deployment | NO-GO / out of scope | Project is scientific validation tooling, not deployment software. |
| Follow-up PR merge | HOLD | Exact new-head CI and explicit merge authorization remain pending. |
