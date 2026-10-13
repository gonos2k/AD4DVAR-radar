# Qualified active-face response phase

`active_face_response.py` is a separate, inert driver for the single
`qy[4,3] = 0` face. It uses the existing `FVFaceFluxCoordinateChart` to
parameterize the 25 free controls. The smooth full-control runner does not
handle active-face events.

The driver first requires a RED `singleface_local_minimum_qualified` receipt
for the exact control and parameter hashes. It checks `P2_passed`, the 360
stage selector signature, the exact chart/pivot and real face event, both
tangential gradients, strict negative/positive one-sided normal slopes, the
mixture weight, and two exact 25×25 positive reduced-Hessian matrices from
that same point. No H matrix is recomputed by this driver. Its approximate
metric is the fixed mixture `(1-theta) H_minus + theta H_plus`.

Before any objective or score call, it also requires a separately frozen phase
plan with `root_launch_clearance: true`, the continuation plan pin, candidate
and parameter hashes, the base qualification digest, runner/common-helper
hashes, and an explicit cumulative budget. The budget must carry the original
3,600-second cap and charged time before this phase, plus the cumulative native
J-success count before the phase and the phase allowance. The driver never
resets those counters or creates a time grant.

The phase-plan object uses `phase_kind: active_face_reoptimized_parameter_response`,
`status: frozen_active_face_response_plan`, `fixed_problem_sha256`,
`continuation_plan_sha256`, `base_control_sha256`, `base_parameters_sha256`,
`qualification_sha256` (canonical JSON digest), `runner_sha256`,
`common_runner_sha256`, and positive `armijo_iterations` / `armijo_backtracks`.
Its `budget` object contains `charged_seconds_before`, `phase_seconds`,
`native_J_successes_before`, and `phase_native_J_limit`; charged time plus phase
allowance must fit within 3,600 seconds. The phase evaluation count starts at
the supplied cumulative success count, so the output records both before and
after totals.

For each `p±h` and `p±h/2`, the driver starts from the qualified base chart
coordinates and runs actual native-J Armijo steps using the stored base metric
as a fixed preconditioner. It does not substitute a linear predictor for an
optimized point. At every final point, the built-in `qualify_active_point`
helper (or an explicit replacement) must audit
the actual 360-stage branch and report the exact face/chart, tangent gradient,
mixture weight, both one-sided normal slopes, matching selector signature, and
native objective. Only then does the driver evaluate the production terminal
score. It reports the exact active-face adjoint and the two central-difference
estimates; future scores never select a point. It does not claim perturbed
curvature certification without new Hessians.

An override callback receives `(control, parameters, budget)` and returns
`singleface_first_order_qualified` with `event_control`, its digest,
`input_parameters_sha256`, `native_J`, `tangent_gradient_inf`,
`full_mixed_gradient_inf`, `minimum_mixture_theta`, `same_actual_branch: true`,
both full side-gradient norms, and the `event_face`, `event_branch`,
`one_sided_normal_slopes`, and `one_sided_normal_roundoff_budgets` fields above.
The built-in callback audits the actual branch through the existing RED
qualifier and evaluates the original objective with an explicit parameter
argument. It updates the shared cumulative native-J counters; an override must
preserve that contract. A thin CLI accepts the fixed qualification receipt and
sealed phase plan.

The current stored A candidate has not passed this active-face base gate, and
there is no frozen phase plan or launch clearance, so no active response has
run. Tests cover this fail-closed path and verify the explicit parameter
argument in the adjoint derivative helper without invoking an FV optimizer.
