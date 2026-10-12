# Residual direction work record — 2026-10-12

## Scope

Implement the pure residual-direction helper, focused mathematical tests, and
RED design record. The caller owns fresh derivative-action evaluation and all
bounded trial / FV acceptance work.

## Progress

- Consulted the repository graph nodes for the existing tangent continuation,
  chart override validator, and minimum-mixture model. The graph confirms the
  shared `_validate_direction_override` contract is the current chart authority.
- Added `build_residual_direction` with strict common-face checks, the Euclidean
  normal projector, current chart/pivot validation, raw and equal-norm-scaled
  directions, and actual side/merit descent diagnostics.
- Added a 26-variable curved-face direct-autodiff regression and focused tests
  for normal scaling, unsupported common-face jumps, unresolved jumps, chart
  mismatch, and adverse curvature.

## Verification

`.venv/bin/python -m pytest -q tests/test_fv_point_3h_residual_direction.py`
completed successfully: 4 passed. This is local synthetic derivative evidence;
it does not exercise FV, HVP production wiring, candidate acceptance, or forecast
quality.

## Remaining integration checks

- The configured trial call chain resolves the derivative action as follows:
  `fv_point_3h_netflux_subspace_trial._run_child` computes a fresh paired HVP
  and calls `prepared.mixing.minimum_tangent_model` at lines 243–244;
  `fv_point_3h_netflux_continuation.prepared` aliases
  `fv_point_3h_gn_prepared_search`, which aliases
  `fv_point_3h_tangent_mixing_minimum_probe` as `mixing`. Its `theta_prime`
  uses `-(jump·Hmix + mixed·(Hplus-Hminus))/(jump·jump)`, the derivative of
  the unconstrained minimum-mixture stationarity relation. This is the active
  formula; the older `fv_point_3h_tangent_continuation` builder is not called.
  `direction_trial.next_direction` passes the first qualified model's control
  block of `residual_direction` to the helper only after checking that the
  requested action vector exactly matches the first HVP direction. The shared
  trial then obtains a fresh paired HVP along the returned residual direction.
- `tests/test_fv_point_3h_tangent_mixing_minimum_probe.py` directly compares the
  minimum-mixture residual derivative and `theta_prime` against autodiff JVP.
  `tests/test_fv_point_3h_residual_direction_trial.py` verifies the production
  ordering: two baseline HVPs, the derivative-action callback, two residual
  HVPs, no GN rows/solves, and two independent repeats. A failed second-arm
  repeat leaves the base uncommitted and cannot fall back to the first arm.
- These checks resolve the concern that the older builder's fixed-normal
  `delta_theta` might be used on this path. Final source freezing and the
  separate terminal-score-only preflight are recorded below.

## Terminal-score-only preflight

`observe_current.py` ran under frozen plan
`5d7215ce2796c2af4473eef8a64295fc3ce90a90bdd5dc7d40f3f2f6dade4276`.
Its independent `load_plan` check and receipt audit verified the 233 source
pins, 302 archive pins, child hash, and exact source-before/source-after map.
The run evaluated only terminal `E` at the archived 7839 and final 2a4d
controls: `E(7839)=7.386235570851804`, `E(2a4d)=7.384565484459929`, with
`delta E=-0.0016700863918748254`. It made zero optimizer/candidate calls and
marked `selection_use=false`; this comparison does not choose or commit a
control. The guarded run took 7.1119 seconds and sampled 810,139,648 bytes of
RSS under the 660-second/2-GiB caps. Source, fixed input, and runtime remained
unchanged during the child.
