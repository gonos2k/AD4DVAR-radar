# PR #133 review / PR #134 scientific observation-provenance checklist

## Second-review authority snapshot

- Review source: user-pasted additional review of PR #134 final documentation head
- Review timestamp/time zone: 2026-08-21 / Asia-Tokyo
- Verified PR/base/head: PR #134 OPEN, base `main@a246966c68f2dcbaea89cc6905b9b1a58ef2e18c`, final reviewed head `2d64599ee7491b58ca36e9bcf799ea578d9d6866`
- Verified final-head CI: run `32408967808` completed SUCCESS for Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI smoke
- Merge authority: HOLD; no merge is authorized by this checklist

## Second-review findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R134-005 | P1-HIGH | The v2 observation derivation contract no longer requires or replays its source identity and mask derivation, weakening the same serialized generation. | Legacy scientific contract integrity | REPRODUCED | repository-actionable | Restore the exact PR #133 v2 replay/equality preimage and reject v2 from every current derivation factory. | Missing/unrelated/mutated v2 source or mask evidence is rejected; a serialized v2 object is audit-only and cannot enter current derivation. | ✅ | ✅ targeted + sensitivity | ⬜ |
| R134-006 | P1-HIGH | A real v13 durable replay directory with `raw_provenance.json` cannot be audit-loaded, and subclass-based `isinstance` checks request a nonexistent verification provenance member. | Durable legacy audit loading | REPRODUCED | repository-actionable | Decode the manifest before member-set validation and branch on exact generation/type. | A four-member PR #133 v13 fixture audit-loads with both semantic flags false; current semantic replay attempts are rejected. | ✅ | ✅ v13/v14 durable fixtures | ⬜ |
| R134-007 | P1-MEDIUM | The cold-start verification semantic flag does not prove equality with the active algorithm source or numerical runtime. | Scientific replay claim precision | REPRODUCED | repository-actionable | Require the current algorithm digest on append and split byte, reconstruction, and exact semantic flags on load. | Source/runtime mismatch remains audit-loadable but exact semantic verification is false; current source/runtime passes. | ✅ | ✅ source/runtime attacks | ⬜ |
| R134-008 | P1-SCIENTIFIC | Acquisition-time offset is sealed but does not affect eligibility, quality, or observation uncertainty. | Asynchronous mosaic representativeness | REPRODUCED | repository-actionable baseline model; real-data calibration remains external | Preregister maximum age, temporal quality decay, and temporal standard-error growth and apply them deterministically. | Stale cells fail the age gate; larger age monotonically lowers quality and raises standard error under the same source evidence. | ✅ | ✅ targeted + sensitivity | ⬜ |
| R134-009 | P1-SCIENTIFIC | Detection-limit and censor tensors are signed but their generation policy is not preregistered or product-recomputed. | Confirmatory censoring estimand | REPRODUCED | repository-actionable baseline policy; empirical threshold calibration remains external | Pin a product-owned registry-derived detection-limit/censor policy and reject evidence that disagrees with it. | Limit/censor relabeling with recomputed outer digests is rejected; deterministic replay is byte-identical. | ✅ | ✅ targeted + sensitivity | ⬜ |
| R134-010 | P2-ENGINEERING | A single replay NPZ duplicates source cubes and forces all tensors through one 8 GiB archive. | Large-cohort scientific replay scalability | REPRODUCED | repository-actionable | Store content-addressed replay shards and validate/load one shard at a time; deduplicate immutable tensor content. | Multi-shard replay passes with no monolithic archive; missing, extra, swapped, or mutated shards fail closed. | ✅ | ✅ shard attack matrix | ⬜ |
| R134-011 | P2-GOVERNANCE | The checklist records the implementation head/run but not the final documentation head/run. | Audit evidence accuracy | REPRODUCED | repository-actionable | Record implementation and final reviewed heads/runs separately and regenerate after the next push. | Local, remote, PR head, and exact-head CI evidence agree. | ✅ evidence model | ✅ diff/type checks | ⬜ external final-head CI |

## Second-review closure status

- Package/scientific-contract generation: `0.99.0`; observation-error plan v5; mask algorithm/artifact v3; observation-error algorithm/artifact v4; error contract v7; verification bundle v10; scoring replay bundle/method v15; semantic generation v13; scoring case v14.
- Legacy boundary: observation derivation inputs v2/v3 cannot be constructed or consumed by current derivation; v13 and v14 replay directories are byte-auditable only and never obtain current semantic flags.
- Replay claim split: bytes verified, verification provenance reconstructed, exact source/runtime semantic verification, and full typed-case scoring replay are separate booleans.
- Scientific baseline: acquisition age gates validity and deterministically decays quality/inflates standard error; detection limit and censor state are product-derived from the preregistered ordered radar registry.
- Storage: current replay tensors are content-addressed, deduplicated NPZ shards with per-shard and 64 GiB total expanded-byte budgets and lazy one-shard loading.
- Project boundary: development/offline numerical and synthetic research remain GO; exploratory real-radar work is conditional; confirmatory claims and external publication remain HOLD pending X134-001; operational deployment remains NO-GO/out of scope.
- Evidence rule: a committed checklist can record its parent implementation head and completed run. The final PR head and its check rollup are verified externally from GitHub because a commit cannot self-contain its own SHA or a future CI run ID.

## Second-review local evidence

- Promotion suite: `231 passed` in `2705.713s`.
- Sensitivity suite: `77 passed`; targeted current/legacy observation replay also passes after the final preimage-preservation edit.
- Ledger suite: `43 passed`.
- Current replay targeted test: shard creation, cold reconstruction, exact source/runtime gating, and missing/extra/swapped/bit-mutated shard attacks pass.
- Legacy durable replay targeted test: real v13 four-file and v14 five-file directories audit-load; semantic replay remains false and current typed-case replay is rejected.
- basedpyright (CI-authoritative `src/advar` scope): `0 errors`, `6994 warnings`, `0 notes`.
- Bytecode compilation and `git diff --check`: pass.
- Full CPU suite (Python 3.12): `789 passed`, `439 subtests passed`, `18` pre-existing TorchScript deprecation warnings, `2792.04s` (`0:46:32`).
- sdist/wheel build: `advar_radar_nowcast-0.99.0` artifacts built successfully with the CI-equivalent `python -I -m build --no-isolation` command.
- Built-wheel target import, both public CLI help paths, and a real three-frame `advar-nowcast` smoke pass; the output remains `nowcast-npz-v74` / `forecast-run-v68`.
- Exact hash-locked Linux CPU wheelhouse installation remains an exact-head CI requirement rather than a macOS local claim.

## Authority snapshot

- Review source: user-pasted additional review of merged PR #133
- Review timestamp/time zone: 2026-08-21 / Asia-Tokyo
- Reviewer-stated base/head: PR #133 head `ee243830...`, merge `a246966c...`
- Verified repository base: `origin/main@a246966c68f2dcbaea89cc6905b9b1a58ef2e18c`
- Verified PR/head/tree: PR #133 merged; head `ee243830174f3d00a7907f9e188c03d58f227b58`; merge `a246966c68f2dcbaea89cc6905b9b1a58ef2e18c`; head tree is reachable from `origin/main`
- Initial worktree state: clean tracked tree on `agent/pr134-observation-provenance-replay`, based exactly on `origin/main`
- CI snapshot: PR #133 exact-head run `32360783641` completed SUCCESS for Python 3.10 CPU, Python 3.12 CPU, and Wheel/CLI smoke

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R134-001 | P1-HIGH | Source-specific gather changed mask/error semantics without changing the preregistered algorithm identities or nested observation contract generations. | Preregistered scientific algorithm identity and current/audit generation boundary | REPRODUCED | repository-actionable | Introduce explicit source-specific mask/error algorithm generations, cascade current observation contracts, and decode prior generations as audit-only. | Old plan/evidence cannot enter current recomputation; current plan/evidence succeeds; changing current semantics without identity change fails a pinned-identity regression. | ✅ | ✅ targeted | ✅ exact-head CI |
| R134-002 | P1-MEDIUM | The scoring replay archive does not independently preserve the verification observation plan, registry, source identity, source-specific evidence, mask derivation, and error derivation needed for cold-start semantic recomputation. | Durable scientific replay bundle | REPRODUCED | repository-actionable | Add content-addressed verification provenance JSON/tensor members and reconstruct the current verification provenance chain from bundle bytes when caller cases are absent. | Delete original Python objects and reopen from bundle bytes only; require identical source/mask/error/v9 bundle digests with `verification_semantic_replay_verified=True`. Full forecast/model scoring remains separately gated by typed cases and `semantic_replay_verified`. | ✅ | ✅ targeted | ✅ exact-head CI |
| R134-003 | P1-SCIENTIFIC | Source-specific spatial QC is bound, but reflectivity, acquisition time, censor state, and detection limit remain mosaic-wide rather than selected-source derived. | Real-radar mosaic observation semantics | REPRODUCED | repository-actionable for the typed contract; real-data validation remains external | Introduce source-specific observation value/time/detection evidence or a signed mosaic-composition artifact and gather the selected source deterministically. | Swapping source-specific values/times/limits while holding the source map fixed is rejected; selected fields equal the selected source row. | ✅ | ✅ targeted | ✅ exact-head CI |
| R134-004 | P2 | Highest-scoring unavailable source does not fall back to the next available source, but the intended scientific policy needs an explicit regression and documentation. | Source assignment policy | REPRODUCED | repository-actionable design lock | Preserve the fail-closed upstream-assignment interpretation and pin it with a direct test and documentation. | Highest score unavailable while a lower score is available yields `SOURCE_MISSING` and never silently selects the lower-ranked source. | ✅ | ✅ targeted | ✅ exact-head CI |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S134-001 | Selected-source spatial range, elevation, blockage, and attenuation evidence is gathered and used for both masks and error tensors. | Merged PR #133 implementation and row-swap adversarial tests. | Preserve source-axis gather and exact ordered registry binding. |
| S134-002 | Scientific replay generation was raised and the preceding generation is audit-only. | Current replay bundle/method v13, semantic generation v11, case v12; v12 audit types. | Extend the same current-versus-audit separation to nested observation contracts. |
| S134-003 | PR #133 exact-head CI is fully green and no new numerical P0 was reported. | GitHub Actions run `32360783641`. | Keep CPU numerical/synthetic regression required. |

## External scientific actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X134-001 | Scientific study owner | Supply legally usable independent per-radar observations and run the preregistered mosaic confirmatory protocol across the required independent physical-event cohort. | Repository contains no independent real-radar confirmatory evidence satisfying this review. | confirmatory real-radar claim / external publication | OPEN |

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P0/P1/P2 is fixed or evidence-disproved in the working tree.
- [x] Every fix has a targeted regression test.
- [x] Adjacent and broad suites pass.
- [x] Every changed schema, attested source, digest preimage, and required nested field has regenerated producer and consumer evidence.
- [x] Evidence/manifests/distribution documents are synchronized when affected.
- [x] PR head equals the reported pushed commit.
- [x] CI failures are classified as code or external infrastructure/policy.
- [x] Offline research, real-radar confirmatory claims, external publication, and operational deployment have separate decisions.
- [x] External scientific evidence remains visible with owner and required action.
- [x] Merge remains HOLD unless explicitly authorized.

## Local verification evidence

- Full CPU suite (Python 3.12): `788 passed`, `439 subtests passed`, `18` pre-existing TorchScript deprecation warnings, `2748.85s`.
- Promotion suite: `230 passed`, `22 subtests passed`, `3120.46s`.
- Sensitivity suite: `76 passed`, `119 subtests passed`.
- Ledger suite: `43 passed`, `27 subtests passed`.
- basedpyright: `0 errors` (`6922` warnings, no new error gate failures).
- `git diff --check` and changed-module bytecode compilation: pass.
- sdist/wheel build: `advar_radar_nowcast-0.98.0` artifacts built successfully.
- Built-wheel target import and CLI help smoke: pass; exact hash-locked offline closure remains a required remote CI check.

## PR delivery evidence

- PR: `#134` — <https://github.com/gonos2k/AD4DVAR-radar/pull/134>
- Base: `main@a246966c68f2dcbaea89cc6905b9b1a58ef2e18c`
- Implementation head: `fd1e9b00193b6f5dacb5ccad5b0da1f0a43ff2eb`
- Exact-head CI run: `32402454698` — SUCCESS.
- Python 3.10 CPU: SUCCESS, `786 passed`, `2 skipped`, `439 subtests passed`, `1h8m28s`.
- Python 3.12 CPU: SUCCESS, `786 passed`, `2 skipped`, `439 subtests passed`, basedpyright `0 errors`, `1h8m4s`.
- Wheel and CLI smoke: SUCCESS, `4m20s`; hash-locked offline installation and bundle/runtime checks passed.
- Nonblocking infrastructure annotation: `actions/upload-artifact` Node.js 20 deprecation.
- PR remains OPEN and unmerged pending independent review and explicit merge authorization.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Development and offline numerical/synthetic research | GO | PR #133 exact-head CI is green; no new numerical P0 reported. |
| Exploratory real-radar mosaic analysis | CONDITIONAL GO | Source-composition contract and synthetic adversarial replay are implemented; descriptive use must retain source evidence and make no confirmatory skill claim. |
| Confirmatory real-radar scientific claim | HOLD | Requires repository closure plus X134-001 independent evidence. |
| External publication | HOLD | Requires confirmatory protocol and independent cohort evidence. |
| Operational deployment | NO-GO / out of scope | Project is explicitly centered on scientific validation, not operational deployment. |
| PR #134 merge | HOLD | PR is OPEN with green implementation-head CI; independent review and explicit merge authorization are still required. |
