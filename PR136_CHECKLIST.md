# PR #135 follow-up / PR #136 scientific-semantics closure checklist

## Authority snapshot

- Review source: user-pasted additional review of merged PR #135.
- Review timestamp/time zone: 2026-08-22 / Asia-Tokyo.
- Reviewer-stated remote main: `71d222b66d4d3d689ac2581383e2bac29afe01ce`.
- Verified repository base: `origin/main@71d222b66d4d3d689ac2581383e2bac29afe01ce`.
- Verified prior PR: PR #135 MERGED; head `5b04f57977a600c1b43c442e264c1a459d98784e`; exact-head Python 3.10/3.12 CPU and Wheel/CLI checks SUCCESS.
- Follow-up branch: `agent/pr136-fso-scientific-semantics`, created exactly from `origin/main`.
- Initial worktree state: clean tracked tree; pre-existing untracked `.omx/` is user-owned and excluded.
- Scope boundary: reproducible offline scientific validation and FSO research only; operational deployment remains NO-GO/out of scope.
- Merge authority: HOLD; no merge is authorized by this checklist.

## Adversarial findings

| ID | Priority | Claim | Boundary | Current-tree result | Classification | Minimal action | Acceptance test | Implementation | Local tests | PR/CI |
|---|---|---|---|---|---|---|---|---|---|---|
| R135-006 | P1-FUNCTIONAL | Current verification bundle v11 is rejected by the FSO/FSOI validator, whose complete-verification allowlist stops at v4. | Current v12 → FSO → FSOI → learning | CLOSED | repository-actionable | Replace the loose allowlist with an exact FSO-generation-to-verification-generation mapping and keep legacy combinations audit-only. | v12/current FSO/FSOI/learning passes; legacy/current mismatches fail; legacy reconstruction remains audit-only. | ☑ | ☑ | PR pending |
| R135-007 | P1-PROVENANCE | Truth-radar selection depends on caller-supplied assignment-score tensors rather than a preregistered product-owned score function. | Multi-radar target selection | CLOSED for deterministic baseline; empirical calibration remains X136-002 | repository-actionable baseline | Derive scores product-side from signed physical evidence under a preregistered selection contract; freeze fallback and deterministic tie-break semantics. | Score-only mutation fails; stale-best/valid-second fallback and ordered-registry tie-break are deterministic. | ☑ | ☑ | PR pending |
| R135-001 | P1-SCIENTIFIC | `CONFIRMED_CLEAR` can be scored as an arbitrary point-valued floor, making the score depend on representation rather than the clear event. | Clear-sky estimand | CLOSED | repository-actionable | Treat confirmed clear categorically: exclude it from intensity point metrics and retain support/false-alarm scoring. | Clear retains categorical weight but has zero intensity/FSO point weight; strong forecast echo remains in false-support scoring. | ☑ | ☑ | PR pending |
| R135-002 | P1-SCIENTIFIC | At `Z == L`, detected and censored report kinds can both be accepted. | Detection-threshold partition | CLOSED | repository-actionable | Use one non-overlapping convention: echo `Z > L`, censored `Z <= L`, clear `Z < L`. | At `Z == L`, detected and clear fail while censored passes. | ☑ | ☑ | PR pending |
| R135-008 | P1-SCIENTIFIC | Scan age is represented only by scalar quality/std inflation, which cannot represent spatial phase displacement in FSS/object metrics. | Asynchronous spatial verification | CLOSED for preregistered conservative gate; empirical motion calibration remains X136-002 | repository-actionable baseline | Add a spatial-metric acquisition-age gate from grid spacing, reference speed, and maximum displacement; preserve non-spatial diagnostics separately. | Source age beyond the registered displacement support has zero FSO/spatial weight even when scalar observation weight remains positive. | ☑ | ☑ | PR pending |
| R135-009 | P2-MATH | Point Gaussian diagnostics use a global detection threshold while censored diagnostics use the local per-cell threshold. | Probabilistic observation support | CLOSED | repository-actionable | Pass sample-specific local detection limits through quantized-bin diagnostics and exclude clear point scoring. | Equal observations with different local limits use their own support thresholds. | ☑ | ☑ | PR pending |
| R135-003 | P2-PROVENANCE | Detection limit is product-derived from range/elevation tensors, but those geometry tensors are not independently derived from radar/grid geometry. | Radar geometry provenance | CLOSED for typed projected baseline; native beam/refractivity authority remains X136-003 | repository-actionable baseline | Add a typed geometry contract and product-side range/elevation derivation from registered radar/grid geometry. | Geometry-only mutation fails; v17 cold replay reconstructs the geometry and source fields exactly. | ☑ | ☑ | PR pending |
| R135-010 | P2-CONTRACT | The nominal acquisition timestamp reference is implicit while local offsets are constrained nonpositive. | Radar scan chronology | CLOSED | repository-actionable | Bind current scientific inputs to `volume_end` and validate `cell_time = nominal_time + offset`. | Timestamp reference is signed in plan/source identity; current nonpositive offsets are accepted only for `volume_end`. | ☑ | ☑ | PR pending |
| R135-005 | P2-GOVERNANCE | PR #135 checklist/README retain pre-merge or ambiguous current/legacy generation state. | Scientific audit record | CLOSED | repository-actionable | Record final PR #135 merge/CI evidence and separate current scientific generations from audit-only generations. | README and PR #135 checklist record head `5b04f579…`, merge `71d222b…`, run `32490139996`, and current v12/v17 generations. | ☑ | ☑ | PR pending |

## Friendly findings and strengths to preserve

| ID | Strength | Evidence | Regression guard |
|---|---|---|---|
| S136-001 | No new signature/digest/promotion bypass was identified. | Reviewer call-path analysis and PR #135 exact-head CI. | Preserve exact typed identity checks while simplifying scientific semantics. |
| S136-002 | `below_detection_reported` is derived compatibility state rather than an independent current input. | Current report-kind source evidence. | Do not reintroduce a second independent censoring authority. |
| S136-003 | Content-addressed verification replay and full-source-cube validation remain strong. | Current v17 replay, v16 audit boundary, and PR #135 adversarial regressions. | Extend replay fields without weakening exact shard or source/runtime checks. |
| S136-004 | Product scope is offline scientific validation, not deployment. | README and PR #135 review boundary. | Do not spend this PR on deployment bundle/runtime activation work. |

## External scientific actions

| ID | Owner | Required external action | Evidence | Blocks | Status |
|---|---|---|---|---|---|
| X136-001 | Scientific study owner | Supply an independent real-radar physical-event cohort and run the preregistered confirmatory protocol. | Repository contains synthetic/replay fixtures, not an independent confirmatory cohort. | confirmatory real-radar claim / publication | OPEN |
| X136-002 | Radar science owner | Empirically calibrate source-selection weights, age/displacement gates, detection-limit geometry, and clear/censor semantics against independent radar characteristics. | Repository can preregister deterministic baselines but cannot infer empirical validity from synthetic tests. | final parameter approval / publication | OPEN |
| X136-003 | Data-ingest owner | Provide authoritative native report-flag and radar geometry metadata needed for raw-to-contract replay on actual radar formats. | Generic repository fixtures do not establish each operational/native format's decoder semantics. | native-format confirmatory provenance | OPEN |

## Implementation evidence

- Package/scientific generation: `0.101.0`; holdout plan v27; observation-error plan v7; radar source/registry v4; signed source identity v3; mask evidence/artifact v5; derivation input v6; observation-error artifact v6; error contract v9; verification bundle v12; FSO v18; FSOI v14; scoring replay bundle/method v17; semantic generation v15; scoring case v16. Holdout plan v26 is audit-only.
- Source routing: product-owned eligibility-first score derived from registered range/elevation, absolute acquisition age, blockage, and attenuation QC; invalid best source falls back to the next valid source; ties use exact registry order.
- Observation estimand: confirmed clear is categorical for support/false-alarm scoring and is excluded from quantitative intensity/FSO point metrics; threshold equality belongs only to the censored state.
- Spatial chronology: current FSO/spatial support is gated by preregistered grid spacing, reference motion speed, and maximum cell-displacement fraction; scalar quality/std diagnostics remain separately reproducible.
- Durable replay: v17 retains spatial support and reconstructs geometry plus product-owned source fields from ledger bytes; v16 is audit-only.
- Governance: PR #135 post-merge evidence records final head `5b04f57977a600c1b43c442e264c1a459d98784e`, merge commit `71d222b66d4d3d689ac2581383e2bac29afe01ce`, exact-head run `32490139996`, and all three required jobs SUCCESS.
- Local non-deployment verification: sensitivity **77 passed**, ledger **43 passed**, and six focused promotion tests **6 passed** (current generation, local threshold, v17 cold replay, CPU semantic generation, v16 audit boundary, and unpatched full semantic promotion). Changed source modules report basedpyright **0 errors**; bytecode compilation, `git diff --check`, and the `0.101.0` wheel/sdist build pass.

## Acceptance summary

- [x] Every review statement is represented exactly once.
- [x] Every repository-actionable P0/P1/P2 is fixed or bounded by an explicit external scientific action.
- [x] Every repository fix has a targeted regression test.
- [x] Adjacent broad scientific suites and focused promotion semantic paths pass locally.
- [x] Every changed scientific schema, attested source, digest preimage, and required nested field has regenerated producer and consumer evidence.
- [x] Scientific evidence/manifests/distribution documents are synchronized when affected.
- [ ] PR head equals the reported pushed commit.
- [ ] CI failures are classified as code or external infrastructure/policy.
- [x] Offline research, FSO learning, real-radar confirmatory claims, external publication, and operational deployment have separate decisions.
- [x] External scientific actions remain visible with owner and required action.
- [x] Merge remains HOLD unless explicitly authorized.

## Final decision

| Scope | Decision | Evidence/condition |
|---|---|---|
| Development and bounded offline numerical/synthetic research | GO | PR #135 exact-head CI is green; current review reports no new numerical P0. |
| Current v12-based FSO/FSOI research path | GO | Exact generation mapping and focused v12 → FSO → FSOI → learning regression pass. |
| Single-radar exploratory analysis | CONDITIONAL GO | Preserve full source/time/state evidence and make no confirmatory claim. |
| Multi-radar exploratory analysis | GO for deterministic offline protocol | Product-owned selection and complete v17 replay are implemented; empirical validity remains X136-002. |
| Confirmatory clear-sky/spatial skill claim | HOLD on external evidence | Repository semantics are closed; X136-001/002/003 remain open. |
| External publication | HOLD | Requires repository closure and independent cohort/calibration evidence. |
| Operational deployment | NO-GO / out of scope | Project is explicitly centered on scientific validation. |
| PR #136 merge | HOLD | Implementation and scoped local verification are complete; PR and exact-head CI are pending. |
