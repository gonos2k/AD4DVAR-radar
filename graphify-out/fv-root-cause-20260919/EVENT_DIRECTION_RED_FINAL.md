# RED final review: event-orthogonal direction comparison — 2026-10-10

## Disposition

The frozen source plan and completed saved run close a two-arm, same-point comparison, one selected candidate, one independent P2 repeat, and one commit. I found no actionable receipt, candidate-selection, HVP-linkage, or point-state inconsistency. The new point has no readiness operators; the run makes no convergence, minimum, response, or forecast claim.

## Frozen plan and resource closure

Plan SHA: `14773f392f41eaf9cf0fdba2b59bbf3427f8d3fdf0d0e7cc92b266704928ea29`. All 156 source and 207 archive pins exist and match. The policy freezes one guarded launch, two arms with at most 24 candidates each, four fresh HVPs, one commit at most, a 0.05 actual-radius cap, and no row VJPs, dense solves, or postcommit readiness.

The raw, gzip, run, and resource hashes match the archive manifest, and gzip decompression reproduces the raw record byte for byte. The child completed with exit 0 in `118.6098 s`, no monitor/resource termination, and sampled peak RSS `565,428,224` bytes under the 2-GiB sampled limit. All numeric leaves in the saved record are finite.

The frozen producer also checks the deadline around plan/base loading, fixed-input preparation, current-point observation, each base and accepted-candidate event capture, both HVPs, candidate observations, and P2. The 118.61-second child duration is the saved execution time, within the 600-second internal budget.

## Direction arms and first-pass selection

The base is PR280 control `027e83cf77f7f9cf97c0c72da9f737b5627d312d6375b698bb5b7f7ae94c6769`, freshly matched to its saved J, R, theta, paired gradients, traces, input, and runtime. The reused prepared GN direction has hash `ab3a2635…`. The event-orthogonal direction has hash `c9ac3946…`; both saved one-sided event gradients were projected to the current face tangent, checked for componentwise agreement, then averaged. The projection retained `99.386%` of squared direction norm. Directions were not rescaled, and each arm used its own model α₀ under the same actual-radius cap.

The prepared GN arm evaluated 22 dyadic endpoints and first passed both original actual-J and actual-R Armijo gates at index 22: J=`0.061192476458468474`, R=`0.004478418676506658`, actual movement `1.6183e-8`. The event-orthogonal arm passed at its first endpoint: J=`0.06118411701571423`, R=`0.0043732197531327595`, actual movement `1.5936e-4`. It had lower actual R and lower J, so it won without invoking tie rules. Because the two arms had very different actual movements, this result does not support a same-distance efficiency claim.

Both arms completed their two fresh HVPs. All four HVP receipts bind to the base control and the corresponding arm direction hash, with sides −1/+1 and the selected-face extension operator. The record shows 4/4 started/completed, 0 row VJPs, and 0 dense solves.

## P2 closure and committed state

The selected event-orthogonal endpoint received the sole independent P2 repeat. It matched J, R, theta, side gradients, and branch trace, and passed paired-branch, face, side/native parity, source, fixed-input, runtime, and deadline checks. The committed control hash is `48b0f95c5aa485a6a17046d4497a2267a389b1cfb2eeedb75d79e7ba038d5729`; theta is `0.48584442402924255`. Relative to 027, J decreased by `8.3603e-6` (`0.01366%`) and R decreased by `1.0520e-4` (`2.3491%`).

The run reports exactly one commit, `candidate_committed=true`, no second candidate, `postcommit_readiness_performed=false`, and `readiness_complete=false`. The current point is therefore committed and closed by P2, with future-point operators intentionally absent. All root, minimum, score, and response claims are false.

## Evidence locations

- Frozen plan: `EVENT_DIRECTION_COMPARISON_PLAN_20261010.json`
- Producer and tests: `examples/weather_scenarios/fv_point_3h_event_direction_comparison.py`, `tests/test_fv_point_3h_event_direction_comparison.py`
- Raw record, run receipt, resource receipt, and gzip: `event_direction_comparison_20261010_attempt1/step.json`, `step.run.json`, `step.resource.json`, and `step.json.gz`
- Archive and verification manifests: `EVENT_DIRECTION_ARCHIVE_20261010.json`, `EVENT_DIRECTION_VERIFICATION_20261010.json`

Prelaunch validation reports 18 tests passed and zero type errors; the saved production run is separate evidence from those checks.
