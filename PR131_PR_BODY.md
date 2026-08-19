# Refocus ADVAR on scientific validation and fail-close unsupported claims

> **Project-direction change**
>
> ADVAR is developed for scientific validation and reproducible offline radar
> nowcast research. It is **not** a production deployment system. This PR does
> not enable shadow, canary, state-advancing LIVE, or automatic MPS deployment.

## Why this PR exists

Earlier work accumulated deployment-oriented contracts while the actual project
goal is to establish whether the numerical method and learned prior are
scientifically valid on independent radar cases. This PR makes that boundary
explicit in the product documentation and changes evidence handling so that
engineering artifacts cannot be mistaken for scientific proof.

The primary questions are now:

1. Are numerical results stable and reproducible on CPU?
2. Are observation, target, QC, censoring, and source semantics preserved?
3. Are evaluation policies preregistered before forecasts are inspected?
4. Are physical events and verification sources genuinely independent?
5. Do probabilistic diagnostics represent observation error and censoring?

## Scientific changes

- Preregister `VerificationObservationErrorPlan-v1` before scoring.
- Bind realized observation-error contract v3 and verification bundle v6 to the
  registered algorithms, registries, parameters, and exact tensors.
- Preserve seven per-cell observation states: clear, echo, source missing,
  QC-invalid, beam-blocked, below-detection censored, and mosaic-unassigned.
- Enforce exact mosaic source-map/state consistency.
- Add a report-only Gaussian diagnostic using
  `sqrt(forecast_variance + observation_variance)` and a left-censored
  likelihood below the detection limit.
- Mark spatial-correlation metadata as `diagnostic_only`; current confidence
  bounds remain clustered by independent physical event.
- Downgrade the generic real-case harness to a content-addressed artifact index.
  Generic JSON can no longer claim semantic E2E validity or scientific sample
  size.
- Stop counting caller-provided event labels as independent physical events.

## Explicit non-goals and safety boundary

The following deployment findings are intentionally **not** resolved here:

- builder/product activation-genesis interoperability;
- release-approved runtime-byte 2-of-2 enforcement;
- activation-head database hardening and production rollback governance.

Those paths remain unsupported and non-authoritative. Candidate-smoke checks
may continue only as reproducibility/supply-chain engineering checks. They do
not confer deployment authority.

| Capability or claim | Decision after this PR |
|---|---|
| CPU numerical development and offline research | GO |
| Synthetic scientific regression | GO, labeled synthetic |
| Generic artifact-index completeness | GO, report-only |
| Real-case descriptive scientific claim | HOLD pending legal radar corpus |
| Confirmatory skill or promotion claim | HOLD pending independent observations and preregistered protocol |
| External scientific publication | HOLD pending independent real-case evidence |
| Shadow, canary, or state-advancing LIVE | NO-GO / outside project scope |
| MPS automatic scoring or deployment | NO-GO |

## Evidence and validation

- Package: `0.94.0`.
- Forecast/deployment generations are intentionally unchanged.
- Adjacent scientific suite: 350 passed, 139 subtests passed.
- Final targeted scientific regressions: 13 passed.
- basedpyright: 0 errors.
- sdist and wheel build: `advar_radar_nowcast-0.94.0` successful.
- `git diff --check`: clean.

## Evidence still required outside the repository

- legally usable fixed native-radar cases covering outage, mosaic handoff,
  censoring, QC rejection, and multiple storm regimes;
- verification observations independent of the candidate training and target
  source closure;
- a preregistered scientific protocol for likelihood, censoring, event grouping,
  and cluster inference.

Until those inputs exist, the repository must not emit a positive semantic-E2E,
independent-sample-size, confirmatory-promotion, or publication-readiness claim.

## Reviewer focus

Please review this PR as a scientific-contract change, not as deployment
hardening. In particular, verify that:

- post-forecast observation-policy changes fail;
- contradictory cell state/source-map combinations fail;
- censored cells are not treated as point truth;
- proper-score diagnostics cannot authorize promotion;
- relabelled event digests cannot satisfy scientific sample size;
- generic stage JSON remains artifact-index-only and non-authoritative.

Detailed item-by-item evidence is in `PR131_CHECKLIST.md`.
