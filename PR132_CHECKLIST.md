# PR #132 scientific observation-error closure checklist

## Authority snapshot

- Review source: user-supplied focused review of merged PR #131
- Review timestamp/time zone: 2026-08-20 Asia/Tokyo
- Reviewer-stated base/head: PR #131 latest changes
- Verified repository base: `origin/main@e696e4343a1a5a6f707a6de9bc3e75e13e20a8a8`
- Verified PR #131/head/tree: merged PR #131; head `cae966315fe9f7351fb55b9bcb89e5e1a584f491`; merge `e696e4343a1a5a6f707a6de9bc3e75e13e20a8a8`; identical tree `14a5bd16d905dd3a8037e7de55473138ca6cd37e`
- Worktree state: clean tracked worktree on `agent/pr132-deterministic-observation-error`; unrelated `.omx/` excluded and untouched
- CI snapshot: PR #131 required Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI checks all succeeded; PR #132 not yet created
- Scope boundary: reproducible offline scientific validation only; confirmatory real-case and publication claims remain HOLD; operational deployment remains out of scope
- Local contract version: package `0.95.0`; observation-error plan v2, realized contract v4, derivation artifact v1, verification bundle v7; forecast/deployment/ledger generations unchanged
- Local targeted evidence: 7 tests passed in 3.027s
- Local adjacent evidence: acceptance+sensitivity+promotion combined serial run: 309 passed, 117 subtests passed, 18 warnings in 2089.49s; failure 0
- Static/package evidence: authoritative basedpyright 0 errors; `git diff --check` clean; isolated sdist/wheel build succeeded for 0.95.0

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R132-001 | P1-HIGH | A pre-registered observation-error plan does not deterministically produce the realized quality, standard-deviation, cell-state, and source-map tensors. | plan → realized scientific tensor lineage | REPRODUCED, FIXED LOCALLY: plan v2 pins the product algorithm; typed raw identity is derived from exact inputs; artifact replay regenerates every tensor; direct tensors remain exploratory-only | repository-actionable | Add a typed deterministic derivation artifact and replay the product-owned derivation byte-for-byte; mark direct tensor construction exploratory-only. | Quality/std/state/source mutation fails; identical inputs produce identical raw/artifact digests; algorithm substitution fails; v7 rejects exploratory contracts. | ☑ | ☑ targeted + adjacent | ☐ PR/CI |
| R132-002 | P1-HIGH | Mosaic source indices are not bounded by or ordered against a typed radar/calibration registry. | mosaic source map → radar identity/calibration parameters | REPRODUCED, FIXED LOCALLY: ordered registry owns index→radar→calibration→quality/std and validates `-1 <= index < radar_count` | repository-actionable | Add a canonical ordered mosaic source registry and bind every source index to radar identity, calibration epoch, quality, and observation-error parameters. | `index == radar_count`, `999`, registry reorder, and cross-radar calibration substitution fail; exact registry replay passes. | ☑ | ☑ targeted + adjacent | ☐ PR/CI |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S132-001 | Generic artifact indexing cannot claim semantic E2E validation or independent sample size. | Acceptance report keeps semantic/sample-size/scientific-review/deployment booleans false and reports only declared labels. | Existing acceptance fail-closed tests remain required. |
| S132-002 | Seven observation states preserve missing, QC-invalid, blocked, censored, and unassigned semantics. | `VerificationCellState` and state/tensor invariants in `sensitivity.py`. | Preserve invalid weight/std zero rules and point-score exclusion for censored cells. |
| S132-003 | Gaussian observation-error diagnostics remain report-only. | Effective variance combines forecast and observation variance while `diagnostic_only=True`. | No promotion or deployment authority may consume this diagnostic. |

## External scientific evidence actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X132-001 | Independent scientific investigators | Supply legally usable real radar cases and independent verification observations; run the deterministic derivation/replay without mocks. | Repository cannot manufacture independent physical events. | confirmatory real-case claim / publication | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P1 is fixed or evidence-disproved locally.
- [x] Every fix has a targeted regression test.
- [x] Adjacent and broad CPU suites pass.
- [x] Public exports, contracts, README, and package generations are synchronized.
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
| Mosaic confirmatory evaluation | HOLD | Repository contracts are closed locally; X132-001 independent real-case evidence remains external and CI is pending. |
| Independent real-case skill claim | HOLD | X132-001. |
| External scientific publication | HOLD | X132-001 plus independent review. |
| Operational deployment | NO-GO / out of scope | Project is scientific validation tooling, not deployment software. |
| PR merge | HOLD | Local and required CI evidence pending. |
