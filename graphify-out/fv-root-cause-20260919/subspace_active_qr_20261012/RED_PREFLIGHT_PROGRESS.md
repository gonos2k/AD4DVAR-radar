# RED preflight: cached-gradient original-J/R progress driver

## Scope and evidence reviewed

Reviewed `cached_gradient_progress.py`, the current diff in
`fv_point_3h_netflux_continuation.py`, the shared dyadic search callback
semantics, and the PR302 cache-admission path. I did not run FV/HVP, candidate,
repeat, or score work.

`cached_operators.admit()` completed as an offline read-only check. It verified
the pinned PR302 source bundle (222 members), archive set (291 pins), and
receipt bundle (5 members), and reported current source drift in exactly the
two intended numerical files: `fv_point_3h_subspace_model.py` and
`fv_point_3h_netflux_continuation.py`. The cached closed endpoint is control
SHA `7839625d…c7c1a`, with stored `hvp_started=hvp_completed=4` and model order
`baseline_tangent`, `robust_gn_coupled`. The new `PROGRESS_PLAN.json` is not yet
present, so this preflight cannot validate its final source/plan hash closure.

The intended resource limits are internally consistent: 19 maximum commits,
24 total candidate evaluations, six per point, 36 fresh HVPs, plus four
historical calls for a total cap of 40. The point-zero cached baseline pair is
reused only after fresh native pair/P2, control, theta, direction, G, input,
runtime, and cached-DGd/model closure checks. Later points have no cached
products and execute fresh two-sided HVP pairs. The driver does not construct
new GN directions. The prior 79.7018435 seconds are subtracted from both
internal and outer limits; the radius stays at two times the pinned base
radius. Rolling comparison is recorded with no technique-selection stop.

## Launch blocker

The current chart contract can contract a requested alpha to
`actual_alpha=requested_alpha*factor`. `candidate_chart` records that mapping,
and the callback has now been corrected to pass the charted `candidate` into
`evaluate_candidate`. Preserve that behavior.

The receipt lookup still has an alpha-key mismatch when `factor < 1`. The
progress wrapper stores `receipts[requested_alpha]`, while
`backend.evaluate(..., actual_alpha)` returns facts with `alpha=actual_alpha`.
`policy_search.search` starts a trial with requested alpha and then
`trial.update(facts)` overwrites `trial["alpha"]` with the actual alpha. After
selection, the continuation indexes `receipts[selected["alpha"]]`, which will
miss whenever the chart safeguard contracts. Carry distinct
`requested_alpha` and `actual_alpha` fields through search facts and index the
receipt by the requested value (or store the selected receipt path directly in
the trial). The actual alpha must continue to reach the backend Armijo/model
calculation and the persisted proposal.

RED recommendation: do not launch the progress run until this receipt mapping
is fixed and covered by a targeted chart-contraction test. Afterward, verify
the generated plan pins exactly the two allowed numerical source transitions
and the current new-file set before any FV work begins.

## Main-thread fix observed; launch gate still pending

The shared driver now passes the charted candidate to `evaluate_candidate`.
Receipt insertion now uses `facts["alpha"]` (the actual alpha sent to the
backend), matching the selected trial after its facts overwrite the requested
alpha. This resolves the code-path mismatch found above. GREEN coverage for
cache reuse at point zero, fresh HVPs at later points, and a contracted-alpha
receipt lookup is still pending, as is final plan/source-pin closure. I have
not run the driver or any physics workload.
