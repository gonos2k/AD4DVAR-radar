# R2 core-strict fixed-branch root diagnostic: signature refusal

This is a new root-only run from the single locked attempt-2 terminal
control; it does not relabel attempt 2's failed `1e-4` search handoff or
the previous fixed-margin Newton refusal. The same correlated point
objective, parameters, frozen verification, seed control and full 54-stage
signature were pinned by hash. Newton candidate admission used the existing
roundoff-aware minmod tracer and required the **same full signature** plus
finite positive measured margins. The final `>1e-4` slope/face requirement
for any later response remained unchanged. No adjoint or reanalysis ran.

The seed exact Hessian again passed its pointwise SPD/symmetry gate:
`lambda_min=0.6942730154315004`, `lambda_max=4453.220758903886`, ratio
`0.0001559035702515652`, relative antisymmetry
`4.038200716034779e-16`. Five PCG solves converged (20, 21, 22, 21 and
22 iterations). Four Newton corrections were accepted under the existing
gradient-merit Armijo rule. Their pointwise face-flux margin fell from
`3.5249320414798194e-4` at the locked seed to
`2.565503880514295e-7`; the maximum control-gradient component changed
only from `0.01080526397720594` to `0.010787464455136356`.

At **Newton iteration 5**, all 16 backtracking candidates passed the core
strict trace but had a **different full limiter/face-sign signature**.
Each was evaluated separately for diagnostic objective and gradient before
being refused by the unchanged refiner's fixed-signature policy. All 16
diagnostic objectives were below the current value
`0.0010925189095366723` (range
`0.0010820067515604666`–`0.0010925181626788467`), but only one had a
smaller Euclidean gradient norm than the current
`0.012945626174262087`. These diagnostic values are **not accepted
Newton iterates** and do not establish a smooth path across the branch
transition. The refiner performed no Armijo merit evaluation on them.
All 16 did have a slightly smaller maximum gradient component than the
current `0.010787464455136356` (range
`0.007117637576435666`–`0.010787357079420825`). The report retains each
trial's control values/hash and diagnostics, but the refiner does not
return its internal Armijo history after raising, so candidate Armijo
ratios and earlier accepted-step history are unavailable.

The guarded child exited 2 with `numerical_status=root_refused`; the outer
execution completed after **132.666 seconds**, with 499 RSS samples and
sampled peak **347,062,272 B** under the 600-second / sampled 1-GiB cap.
Source, input, seed control, plan, Stage A plan/report, attempt-2 report
and prior terminal report hashes stayed stable. Raw child/resource/runner
records and exact execution-time new probe/runner/plan snapshots are
preserved in `point_core_strict_root_attempt1/` with a seven-file SHA
manifest. `response_validation=not_performed`.

The result supports a narrow conclusion: this **same-signature Newton
correction policy** cannot continue from the observed point within 16
backtracks, even when candidate margins below `1e-4` are permitted for
root search. It does not prove a stationary point is absent. A new
branch-aware search would have to check the actual objective or
stationarity residual after any accepted signature change, then form a
new local gradient/Hessian there. The old branch's Newton-slope merit
prediction cannot be treated as a smooth-path guarantee across a minmod
kink. Any eventual root still needs fresh final stationarity, SPD,
branch-margin, adjoint and signed-endpoint gates before an R2 response
claim.
