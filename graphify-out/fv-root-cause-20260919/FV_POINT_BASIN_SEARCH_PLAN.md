# R2 branch-aware point-objective basin search plan

This is one new deterministic numerical policy for the **same** 4×5
full-valid, correlated 13-parameter point problem and frozen synthetic score
from `FV_POINT_STATIONARY_RESPONSE_PLAN.md`. The exact pinned Stage A
preflight and direct warm Newton refusal remain separate evidence. Start
only at the archived warm control. No observation, correlation, prior,
boundary, verification field or response tolerance changes are allowed.

The warm exact-HVP Newton–PCG attempt refused before a candidate step because
its operator was not positive definite. Keep `refine_stationary` unchanged.
First seek a smooth local basin by reducing the original point objective
with deterministic limited-memory BFGS (memory 10) and actual-objective
Armijo backtracking:

- At each accepted iterate compute the current point objective and exact
  control gradient. Require finite values and a strict 54-stage minmod trace
  with minimum scaled slope and face-flux margins `>1e-4`.
- Form an L-BFGS descent direction from positive curvature pairs. If
  `g.T d` is not strictly negative at a scale of
  `1e-12 ||g||_2 ||d||_2`, clear history and use `-g`. Limit the proposed
  control-space step norm to 1 before trying `alpha=1,1/2,...,2^-15`.
  The usual last-pair L-BFGS initial scale is clipped to `[1e-6,1e6]`;
  invalid two-loop arithmetic clears history and falls back to finite `-g`.
- For each trial, first require a strict 54-stage pointwise branch and both
  margins `>1e-4`. The full signature may differ from the current signature
  during this **search**, but a tie or too-small margin is refused. Accept
  only if the actual finite unchanged objective satisfies
  `J_new <= J_old + 1e-4 alpha g.T d` and decreases by more than
  `64 eps64 max(|J_old|, tiny64)`. A branch change clears all L-BFGS history
  before the next local model. There is no claim that the segment connecting
  accepted iterates stayed in one minmod branch.
- On the same branch, retain a pair only if
  `s.T y > 1e-10 ||s||_2 ||y||_2`; `s` and `y` are differences of consecutive
  accepted controls and finite gradients with endpoint-matching signatures.
  This is a heuristic curvature pair, not a certificate that the segment
  avoided an even number of branch crossings. Otherwise
  clear history. Require the trial gradient finite **before** accepting it.
  If no trial passes 16 backtracks, report a search
  refusal, not a stationary result.

Stop the search after at most 100 accepted steps, 1600 trials, or 600 seconds
of its own elapsed time, counted from before the initial branch/objective
evaluation and checked again after expensive trial evaluations and
checkpoint writes. Every accepted-iterate checkpoint includes its FP64
control values. The sole candidate gate is
`||grad_c J||_inf <=1e-4`; otherwise budget exhaustion is `basin_incomplete`.
This 1e-4 is only a handoff threshold, not the final stationarity criterion.
At the candidate, recheck the strict branch and build the fresh exact
26×26 Hessian from gradient JVP columns. Require all entries finite,
relative antisymmetry `||H-H.T||_F/max(||H||_F,tiny64)<=1e-10`, and for the
symmetrized Hessian finite `lambda_max>0`, `lambda_min>0`,
`lambda_min/lambda_max>sqrt(eps64)`. A failed curvature gate is
`curvature_refused`, not a reason to change seed or thresholds. A basin search
that ends at a phase/step/trial/line-search limit is `basin_incomplete` with
its more specific `basin_status` preserved.

Only then call the existing exact-HVP `refine_stationary` at that candidate,
pinning its **terminal search** 54-stage choices and face signs and both
`>1e-4` margins. Retain its 8 Newton /16 backtrack /104 PCG limits and
actual linear residual gate, and require final
`||grad_c J||_inf <1e-10`. Repeat the same exact Hessian symmetry/SPD
diagnostic at the polished point before classifying nominal stationarity.
Persist and parent-verify the final 54-stage branch signature and both
`>1e-4` margins, as well as the candidate and final SPD reports and final
gradient; a mere child success label is insufficient.
A point found by the L-BFGS search alone is not
stationary or response-eligible.

One serial child is capped at 1200 seconds and sampled 1 GiB RSS; the
search's 600-second phase cap leaves time for Hessian and exact refinement.
The runner must enforce the outer elapsed cap even if polling misses a
just-over-limit exit. The sampled RSS cap does not bound between-sample
spikes or parent memory. Checkpoint every accepted iterate and refusal
category atomically; bind the exact preflight/problem/source/input hashes
before and after. Keep execution, basin search, curvature, exact
stationarity, local response and signed validation as separate statuses.
No adjoint or endpoint reanalysis runs in this basin-search attempt.
