# PR #133 review / PR #134 scientific observation-provenance checklist

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
