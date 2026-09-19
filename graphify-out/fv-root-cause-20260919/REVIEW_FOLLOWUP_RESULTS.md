# PR160 review follow-up

The merged baseline is 5d4666c (PR head 0eb2a9d); both have file tree
dc06c91484ae24fae659bb253cffed1dd2678c8f. Full CPU run 35424239984 was
cancelled at the user's request. UI, packaging and typechecking passed, but
there is no full-suite pass. Follow-up uses bounded local checks, not new CI.

## Fixed-support partial observations

Preparation now requires at least one accepted observation instead of requiring
every observation to be accepted. Existing complete initial active support and
known boundary requirements remain. No missing sample is relabeled clear sky.

Six small tests cover missing/QC/zero-quality samples through preparation,
analysis, refinement, forecast and exact-Hessian research response, masked-value
invariance, all-invalid rejection and unknown initial-support rejection.
Result: 6 passed in 47.52 s. The three end-to-end cases were then strengthened
to require exactly zero excluded-observation sensitivity with frozen background:
3 passed in 47.59 s. This does not cover arbitrary unknown initial support,
typed priors, grid/time contracts or legacy FSOI publication.

## Explicit background-mean derivative bridge

The research response can now take an explicit B(y, theta) builder and return
theta sensitivity using the same objective, forecast score and adjoint solve
as observation sensitivity. Mean changes are differentiable; support, precision,
verification weights and boundaries stay fixed. Typed priors remain rejected.
Invalid observations are canonicalized to the preparation min-dBZ placeholder
in both the baseline check and the derivative path; they do not become observed
clear sky or independent background information.

Four tests with nonzero theta and nonidentity spatial observation dependence
passed in 95.55 s (bounded run 96.83 s / 341 MB). The independent oracle uses a
small dense Hessian solve, not a second PCG solve. Theta discrepancy was 2.27e-9
within a 6.87e-8 residual/roundoff bound; observation discrepancy was 9.16e-10
within 6.79e-8. The bound propagates the true adjoint residual through H^-1 J_cp;
it does not loosen the product solve or stationarity threshold. Polished central
reanalysis checks both observation and parameter directions.

A separate partial-observation callback test passed in 19.33 s: changing an
excluded finite value leaves y/theta sensitivities bitwise unchanged, its own
sensitivity is zero, and valid-observation/theta derivatives remain nonzero.
The default frozen/first-observation modes have previously recorded bitwise
continuity across GN/exact modes on the small fixture; this is not a 240-grid
rerun. No new learned-prior precision or held-out improvement is claimed.

Affected existing FV residual/analysis/forecast/default-probe tests: 32 passed
in 21.94 s. Pinned basedpyright 1.39.9 on both changed product modules reported
0 errors, 0 warnings. The shared environment and locks were not modified.

## Finite impact display

The existing 240-grid evidence is reused at measured revision ce6e36a. The
publisher verifies that revision's source hashes and the saved tensor/checkpoint
hashes; it does not claim the evolving source was remeasured.

Both signs appear in the original HTML. Errors relative to actual score change
are 57.94%, 40.78%, 221.37% at +0.001, +0.0005, -0.0005 dBZ. The denominator
is now explicit. A reliable finite-impact amplitude range remains unverified.
Original HTML forecast-array hashes are unchanged. Aside DOM and screenshots
confirm the rendered values and limitations; see fv_impact_relative_error.png.

## Three-hour prescribed-flow check

The existing probe accepts --leads (1–18), retaining the 3-lead default. The
new run covers 32/64/128 grids over 180 minutes. This fills a missing measurement,
not a requirement for high accuracy or complete assimilation.

| Flow | 128-grid echo relative L2 error | 128-grid JVP relative error |
| --- | ---: | ---: |
| Translation | 15.25% | 31.11% |
| Rotation | 9.61% | 20.05% |
| Area-preserving strain | 3.37% | 13.45% |

All field/JVP errors decrease with grid refinement. Maximum transformed budget
residual is 7.06e-16. The donor-cell discretization retains substantial
long-horizon error; discrete derivative consistency does not remove numerical
diffusion. The prescribed flows are not estimated here. Known-zero exterior
and a known initial field remain the declared boundary/initial conditions.

Elapsed 35.31 s, sampled peak RSS 327,385,088 bytes; limits 120 s / 2 GiB.
See fv_grid_convergence_180min.json and its .run.json. This cost does not include
analysis, adjoint, reanalysis or learning and is not whole-chain D7 evidence.
Default probe regressions: 3 passed in 4.51 s. The original HTML now exposes
the long-horizon table; see fv_long_horizon.png.

## Remaining scope

Typed background mean/precision/support integration, general partial support,
finite-impact validity domains, and full D7 remain open. Do not tune held-out
cases, relax stationarity gates or convert checklist counts into scientific
completion percentages. See POST_MERGE_REVIEW_CHECKLIST.md.
