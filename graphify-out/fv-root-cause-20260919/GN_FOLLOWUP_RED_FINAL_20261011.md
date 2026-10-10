# RED final review: GN follow-up from 29b52 — 2026-10-11

## Disposition

The frozen run closes one fresh current-point GN build at 29b52, one first-passing candidate, one independent P2 repeat, and one commit. No event gradient/JVP or event projection was reused or computed; the event selector is recorded as a trace snapshot only. No future-point readiness or second candidate was run.

Plan SHA `e9d8c9b356c9bb97ebc1b74bfd69ee3c2bbd6aea7cfb8020440b656cfb4a422d` pins 162 sources and 223 archives, all present and matching. At base `29b52a1377caa7fb52431bdbea25124fa827ac548eeefb9b39ce0e5f240d97c4`, the run freshly qualified J, full mixed residual, θ, side-gradient pair, strict traces, fixed input, and runtime, then completed 24/24 row VJPs, one dense solve, and two HVPs linked to the current direction. Event gradient and JVP counts are both zero.

The prepared GN search passed both original actual-J and actual-R Armijo checks at candidate 8 of 24: α=`0.006481251126157063`, J=`0.061035034667252686`, R=`0.003157465741517131`, and θ=`0.6863690764427828`. The Euclidean displacement recomputed from the raw 26 control coordinates is `0.00029597070591344505`; the saved candidate movement is `0.0002959707059134451`, a `5.42e-20` difference within the 128ε relative roundoff allowance. The record reports eight evaluated slots.

The independent P2 repeat matched J, R, θ, side gradients, and branch traces, and passed the face, paired-branch, input, source, runtime, and deadline checks. The committed control SHA is `bef45912ae6253673c6bd743670b374bdde76a02ba6199c464d29f6a1c863e5c`. Relative to 29b52, J decreased by `1.3907e-5` (about `0.0228%`) and R decreased by `0.0001240` (about `3.78%`). The run made exactly one commit and records no postcommit readiness or later candidate.

The base trace at 29b52 and the accepted/P2 trace at bef459 both show negative increments with the right selector at stage 124, x, interior cell `(1,2)`. This is endpoint snapshot evidence only; the run did not constrain the selector or compute event derivatives. The single step does not establish convergence, a minimum, response, forecast value, or closure of the broader 70% FV completion/P3 milestone.

## Receipts

The child exited 0 in `62.8090 s`, with no resource termination or monitor error and sampled peak RSS `417,497,088` bytes below the 2-GiB limit. Raw SHA is `7b02ad8cff621f13cfeecd7a95e3990ea7e0a65e0e31c29c1e1e456604fcc33a`; gzip round-trip is lossless. Prelaunch checks report 13 tests passed and zero type errors/warnings; those checks are separate from the guarded run.

- Producer and plan: `examples/weather_scenarios/fv_point_3h_gn_followup.py`, `GN_FOLLOWUP_PLAN_20261011.json`
- Run record, gzip, run/resource receipts, and archive manifest: `gn_followup_20261011_attempt1/step.json`, `step.json.gz`, `step.run.json`, `step.resource.json`, and `GN_FOLLOWUP_ARCHIVE_20261011.json`

Root, minimum, score, and response claims remain false.
