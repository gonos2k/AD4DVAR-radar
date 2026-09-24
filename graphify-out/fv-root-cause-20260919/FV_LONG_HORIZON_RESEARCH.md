# G6 bounded regular three-hour FV forward horizon

The shared `FVResearchProblem` already accepted an arbitrary positive regular
lead count tied to `NowcastConfig`. This slice supplies a physically
consistent **10-minute, 18-lead** 4×5 research fixture and makes verification
times explicit. Observations are at 0, 10 and 20 minutes. Forecast frames are
at 30, 40, ..., 200 minutes absolute, so the final frame is **180 minutes
after the last observation**, not 200 minutes after analysis.

The fixed five-mode coefficient box was valid for 1-minute intervals with
9 substeps. Merely changing the interval to 10 minutes while retaining
9 substeps fails the existing `bounded_fv_coefficients` CFL-domain check.
The new fixture uses 90 substeps per interval, preserving the 6⅔-second
substep and the coefficient-box CFL contract. It supplies 180 analysis and
1620 future boundary entries, each with two SSPRK stage traces. The expected
stage count is `2 * (2 + 18) * 90 = 3600`. The boundary is fully known and
constant in time in this synthetic case; a separate nonzero-flow regression
varies future traces by lead and verifies that reversing their order changes
the forecast.

The 10-minute log growth is `0.08` per interval. The integrator divides it
across 90 substeps; it does not apply the full value 90 times. In zero flow,
the analytic forecast after lead `k` is the analysis-end echo multiplied by
`exp(0.08*k)`. The fixture also regenerates the three observations and
future truth with the **same discrete FV operator** under nonzero flow.
The bounded forward probe reports:

| Case | Max absolute forecast difference from same-operator truth | Score against that truth |
|---|---:|---:|
| Zero flow | `1.72591e-11` dBZ | `1.39939e-22` |
| Nonzero flow | `1.28182e-11` dBZ | `2.36228e-23` |

The probe completed in about 9.56 seconds with macOS peak RSS
221,200,384 bytes. It uses a 60-second alarm and reports build and forecast
times separately. These tiny differences are a **time wiring and forward
replay check**, not independent forecast accuracy: truth and forecast use the
same discrete model, constant boundaries, stationary flow and stationary
interval growth. No new GN analysis, stationary refinement, adjoint, signed
reanalysis or physical validation was performed. The strict minmod branch
tracer refused both forward inputs, including the structurally zero-flow
case; no local response is claimed for this profile.

`verification_times_seconds` is an optional declared tuple on the shared
problem. When present, it must exactly equal the configured forecast lead
times; shifted/reversed labels are rejected, and the declaration is included
in fixed-problem identity. Legacy one- and two-lead callers may continue
using implicit lead order. The new identity field and support description
change fixed-problem hashes even for those legacy cases, so old caches must
be rebound. This check binds a **declared** timestamp to each verification
frame; it cannot prove that externally supplied pixel values were captured
at that time. The synthetic case constructs them in the declared order.

The affected shared-problem and lead tests report **33 passed**, with 18
existing TorchScript warnings. Pinned typecheck on the changed module, case,
probe and tests reports 0 errors/warnings/notes. The commands, source hashes
and complete probe values are in `fv_long_horizon_manifest.json` and
`fv_long_horizon_metrics.json`.

This closes a bounded **regular three-hour forward execution and time/CFL
contract**. Irregular observation times, time-varying flow/growth, uncertain
future boundaries, long-horizon local-response eligibility, independent
weather truth and useful forecast skill remain open G6/G10 work.
