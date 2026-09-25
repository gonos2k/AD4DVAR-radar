# R2 one-start gradient-merit sector-root feasibility plan

This is a new bounded experiment after PR #198, not a reinterpretation or
threshold change in `FV_POINT_SECTOR_ROOT_ATTEMPT1_RESULTS.md`. The prior
child report SHA256 is
`c32d148cd08523661f4838ceebd518f57c26e6b92f0ebcf25d6493a4c1658aab`.
Select **only** its sixth accepted candidate (Newton iteration 6), with
FP64 control SHA256
`17fe09298738471223846280f77095f7f3bcb96d9e1d527dacf71a62b0628a8f`.
Its archived objective is `0.0010782540255775742`, gradient 2-norm
`0.01161246395765124`, and maximum gradient `0.006432750851827679`.
The fresh 54-stage core trace must match archived full signature digest
`ae447d2aeb384f7c39f7c97ba9b034fe3cdd6b7222f5d58d2b8587b2972b1da5`;
archived scaled slope/face margins are `0.0007538289845848797` and
`2.540342154109891e-06`. This is a search seed below the final face-margin
gate, not an eligible response point.

Preserve the same fixed correlated 4×5 point objective, parameters,
boundaries, score and 26-control/13-parameter layout. The numerical target
is the stationarity equation `F(c)=grad_c J(c,p)=0`. The gradient-merit
function is `Phi(c)=0.5*||F(c)||_2**2`. Build the exact Hessian-vector
operator with gradient JVP, solve each Newton equation by the existing PCG
at true relative residual `<=1e-10`, and relinearize after every accepted
step. First audit the seed's 26-column exact Hessian for finite symmetry
and positive eigenvalues. Preserve 8 Newton, 16 backtrack and 104 PCG
iteration limits.

Every candidate must pass the existing roundoff-aware 54-stage core minmod
oracle with finite positive slope and face margins. On the current full
limiter/face-sign signature, retain the refiner's unchanged normalized
gradient-merit Armijo decision. For a **changed** full signature, do not
apply a derivative computed on the old sector. Accept only if both merits
are finite and the measured decrease satisfies

`Phi_old-Phi_new > 128*eps64*max(abs(Phi_old),abs(Phi_new),tiny64)`.

Do not require `J` or `||F||_inf` to decrease at every intermediate sector
switch. Record both anyway; the prior run showed why these three measures
must not be silently conflated. This is a root-finding policy, not a
monotone physical-analysis objective policy. It may move to a different
local stationary point. No continuity along a candidate segment or global
convergence is asserted.

Only a **fresh final** `||F||_inf<1e-10`, a core-strict 54-stage final
trace, and finite/symmetric/SPD exact final Hessian can be called a
branch-local root. Keep the final `>1e-4` slope and face margins as the
separate response-eligibility gate. Label a root below either margin
`merit_root_low_margin`, not response-eligible. Even a
`merit_root_margin_qualified` status does not establish an adjoint,
full VJP or signed nonlinear reanalysis. In all statuses set
`response_validation=not_performed`.

Run exactly one serial child with a 600-second wall and sampled 1-GiB
child-RSS guard. Bind source, current input, plan, prior report, seed
control, and prior control trace by SHA256 before/after. Record PCG,
branch, candidate, acceptance and resource statuses atomically. The
parent must independently verify PID, exit/monitor/resource status,
elapsed time, sampled RSS, source/input/plan hashes and, for a reported
root, final control hash, fresh final branch trace, stationarity and
curvature gates. A declared numerical refusal stays a refusal; unknown
callback/invariant errors are execution failures.
