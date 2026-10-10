# RED final review: three-point GN continuation — 2026-10-11

## Disposition

The frozen run completed its three-step cap as three sequential P2-closed GN commits. Each point rebuilt its own full mixed residual, 24 row VJPs, one dense solve, and two HVPs; each step committed only its first passing candidate. No event gradients/JVPs, event projection, postcommit readiness, or fourth point were performed. The final cap status is not a stationarity or optimum claim.

Plan SHA `319bb7798348643c0198fd09ae9bbe19ddfb88ae7bcfc94c50a80745b29c2d65` pins 164 sources and 228 archives, all present and matching. The P2 chain is `bef45912…` → `6eaee66c…` → `1825954e…` → `454396e7…`; each accepted point becomes the next point’s fresh base before another solve/search.

| Step | Base control | Direction SHA | α | Raw-control Δ₂ | P2 objective | P2 R |
|---:|---|---|---:|---:|---:|---:|
| 1 | `bef45912…` | `0794ce12…` | `0.0031256268253308304` | `0.000132279893373498319` | `0.06102909648828514` | `0.0031335040350494955` |
| 2 | `6eaee66c…` | `07f55bbc…` | `0.001546937476496079` | `0.000063535597686301853` | `0.061026299351646204` | `0.0031277557240223363` |
| 3 | `1825954e…` | `172ac6c3…` | `0.0007705199346405181` | `0.0000312019054380250068` | `0.06102493753540203` | `0.003127039112866745` |

Each raw-control displacement was recomputed from that step’s saved proposal and its current base; it agrees with the saved `actual_path_norm` to roundoff. All were below the `0.05` radius. Per-step first-passing search counts were 9, 10, and 11, for 30 of 72 allowed slots. Each selected candidate passed original actual-J and actual-R Armijo, then its independent P2 repeat matched objective, full R, θ, side gradients, trace, pair, face, input, runtime, source, and deadline.

The run applied three optimizer steps, with 72/72 row VJPs, 3/3 dense solves, and 6/6 HVPs. Final J/R were `0.06102493753540203`/`0.003127039112866745`, down from `0.061035034667252686`/`0.003157465741517131` at bef459. The run does not establish convergence, a minimum, response, or forecast value. The final point has no GN readiness operators; any further step must rebuild them.

## Receipts

The child completed in `331.8552 s` with exit 0, no resource termination or monitor error, and sampled peak RSS `476,413,952` bytes under the 2-GiB limit. Raw SHA `f3cd137e9f3e348d9ca679a7b701bad9294ed5c4d2b7693c0c6f150dbb716242` and gzip SHA `3fdc6f32d5a58b5e929e31a4e93be698d9da9738e913a99e05ae77534b86239c` match the archive manifest; gzip round-trip is lossless. Source, fixed-input, runtime, and deadline closure flags are true. Prelaunch checks report 14 tests passed with zero type errors and warnings; those checks are separate from the guarded run.

- Producer and plan: `examples/weather_scenarios/fv_point_3h_gn_continuation.py`, `GN_CONTINUATION_PLAN_20261011.json`
- Raw record, gzip, run/resource receipts, and archive: `gn_continuation_20261011_attempt1/step.json`, `step.json.gz`, `step.run.json`, `step.resource.json`, and `GN_CONTINUATION_ARCHIVE_20261011.json`

Root, minimum, score, and response claims are false.
