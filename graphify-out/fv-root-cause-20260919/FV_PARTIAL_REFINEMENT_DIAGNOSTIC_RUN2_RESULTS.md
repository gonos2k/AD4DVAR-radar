# G3b nominal-only diagnostic, attempt 2

The user approved the separate 240-second / sampled 1-GiB nominal-only
diagnostic in `FV_PARTIAL_REFINEMENT_DIAGNOSTIC_PLAN.md`. It used the same
fixed 4×5 partial-observation problem and all nine recorded problem/input
identity fields as run 1. The diagnostic source changed only to expose
backtrack refusal categories and stop before response work. The six raw
outputs remain in `partial_refinement_diagnostic_attempt2/`, with hashes and
source identities in `fv_partial_refinement_diagnostic_run2_manifest.json`.

| Axis | Measured outcome |
|---|---|
| Guarded preflight | exit 0; 2.073 s; sampled peak RSS 316,391,424 bytes |
| Numerical child | exit 1; 136.317 s; sampled peak RSS 347,701,248 bytes; no resource termination |
| GN | same 4 outer / 83 PCG iterations and final objective 0.02076475483611489 as run 1 |
| Nominal Newton | four 29-iteration PCG solves; fourth line search refused |
| Final backtracks | **16 branch refusals; 0 nonfinite candidates; 0 finite Armijo refusals** |
| Nominal / response status | `refused` / `not_established`; no tangent, adjoint, signed endpoint or reanalysis |

The failure is now localized to the callback's branch admissibility policy.
It is **not** an overflow or a finite Armijo-decrease failure at the final
Newton iteration. The callback combines strict minmod branch eligibility,
quantitative slope and face-flux margins, and an exact selector/face-sign
match to the GN seed. The run did not record which of these checks rejected
each candidate, so it does not justify deleting any one guard yet.

This does not prove the partial objective lacks a stationary point. The
nominal Newton search may need to move between smooth limiter sectors even
though the final point used for an implicit derivative must itself have a
qualified local branch. That policy question will be decided against the
discrete equations and a bounded counterexample, then tested without
relaxing the final stationarity or response gates. G3b remains **open**.
