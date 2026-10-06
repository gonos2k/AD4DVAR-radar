# PR247 records and execution audit: RED

**Disposition: one non-blocking validation gap; the archived attempt itself passes the recorded resource and integrity checks.** This review was read-only and did not run the FV objective, an HVP, or a response calculation.

## Finding: resource status accepts an internally contradictory cap report

`fv_point_3h_bounded_coupled_step.execution_status()` checks elapsed time for finiteness and termination flags, but never compares `elapsed_seconds` with the report's `wall_limit_seconds` or `sampled_peak_rss_bytes` with `rss_limit_bytes` ([helper](../../examples/weather_scenarios/fv_point_3h_bounded_coupled_step.py:469)). PR247 calls this helper for both the archived curvature resource in `load_base()` ([current Newton step](../../examples/weather_scenarios/fv_point_3h_current_newton_step.py:48)) and the current guarded launch ([current Newton step](../../examples/weather_scenarios/fv_point_3h_current_newton_step.py:221)). Thus a resource JSON with `resource_termination: null`, exit 0, and a sampled peak above its declared cap is labeled `completed` if its other fields pass.

Minimal in-memory reproduction using the repository `.venv` returned `completed` for `sampled_peak_rss_bytes = 2 * 1024**3`, `rss_limit_bytes = 1024**3`, `elapsed_seconds = 18.66`, and `wall_limit_seconds = 1`. The helper should reject contradictory reports (at least elapsed above the declared wall limit and sampled peak above the declared RSS limit); the base loader should also compare the archived report's declared limits to the reviewed policy. This is a validator gap, not evidence that the actual archived run exceeded its caps.

For the actual attempt, `step.resource.json` and the embedded `step.run.json.resource` agree: exit 0, 18.663 s elapsed against 300 s, sampled peak 358,252,544 bytes against 1 GiB, 70 samples, no termination, no SIGTERM, no monitor error, and no cleanup error. The child hash in the parent matches `step.json`; the parent reports completed and `one_original_J_step_accepted`.

## Provenance and result checks

- All 80 source pins and 4 archive pins in `F82C_NEWTON_PLAN_20261005.json` match. All 17 files in `F82C_NEWTON_MANIFEST_20261005.json` match, including the three attempt records, final GREEN/RED notes, plan, and scientific result.
- `load_base()` requires the curvature raw audit, run record, resource record, and checkpoint to be plan-pinned. It checks the archived raw audit's completed phase/status, source and fixed-input invariants, control/parameter hashes, curvature/checkpoint Hessian equality, HVP counts, minimum-eigenpair audit, and exact reviewed policy. The actual base resource and parent records satisfy the tested status checks.
- The Newton child pins the plan, sources, and archives before work and compares them again after work. The accepted point has a new control hash and matching trial/full-signature partition. The source maps are identical before/after; fixed-input and runtime identities pass. Cancellation is represented as failed execution by the guard status helper; no cancellation occurred in this run.
- The scientific result accurately limits claims to one accepted unshifted step. It records no new HVP, score, response, accepted-point curvature, stationarity, or physical validation. Candidate `J`, `Phi`, and gradient norm decreased, but the candidate remains nonstationary; endpoint branch evidence does not certify the path. These limits are explicit in the plan and result.

## Evidence limits

The run's RSS protection is sampled child RSS every 0.25 s, explicitly labeled as not an OS hard allocation limit. The observed sample stayed well below 1 GiB. The helper gap matters for checking archived or corrupted/inconsistent resource receipts and for defense in depth; it does not invalidate the actual attempt's evidence. No independent weather skill or finite observational influence evidence is part of this one-step record, and the plan correctly leaves those questions open.
