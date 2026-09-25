# R2 nominal point-control refinement diagnostic

The guarded no-solver result in
`point_response_preflight_attempt1/point_response_preflight.json` has SHA256
`d8377046433ad8d63bf32f3f60b81085061d4be4d7ccf6591e5df69f9a3dfbd5`.
It confirms a full-valid, correlated 4×5 point problem with 26 controls and
13 parameters, 54 strict minmod stages, a minimum scaled slope margin of
`0.00076156935`, a minimum scaled face-flux margin of `0.00775273489`, and
a warm maximum gradient of `15.15045066`. This point is not stationary.

Run **one** warm-start `refine_stationary` feasibility attempt, in a fresh
Python child under a 240-second wall cap and sampled 1-GiB child RSS cap.
The cap is sampled every 250 ms and is not a hard OS memory bound. No GN,
basin finder, adjoint or signed endpoint reanalysis is part of this run.

Reconstruct exactly the preflight `fixed_problem` and require the same
problem, warm-control, parameter, verification, correlation and direction
hashes. Require the current R2 preflight plan SHA to equal the SHA recorded
inside the pinned Stage A artifact. Pin the entire 54-stage warm limiter
choices and face signs.
Every evaluated candidate must pass strict branch tracing with exactly 54
stages, minimum scaled slope margin `>1e-4`, minimum over stages of
`min(abs(face flux))/max(abs(face flux)) >1e-4`, and exact nominal branch
signature. The point objective and input parameters remain unchanged.

Use the existing exact-HVP Newton–PCG refiner unchanged: at most 8 Newton
iterations, 16 backtracks per step and 104 PCG iterations per linear solve.
The refiner checks its actual Euclidean Newton residual relative to the
gradient norm at `<=1e-10`; success additionally requires
`||grad_c J||_inf <1e-10` and a final repeat of the pinned branch/margin
checks. Save candidate refusal reasons and resource/source/input identity
checks. A finite failure or branch/curvature/PCG refusal remains a refusal;
do not loosen limits or change the starting control in this attempt.

Even if refinement succeeds, label the output only `nominal_stationarity`
until the dense local Hessian/SPD, fresh adjoint, full VJP and signed
nonlinear reanalysis gates in `FV_POINT_STATIONARY_RESPONSE_PLAN.md` run
separately. A failed direct warm refinement requires a separately
predeclared point-objective basin-search policy, not an ad hoc retry.
The outer runner must reject an elapsed time above 240 seconds even if its
polling monitor did not mark a wall termination before the child exited.
