# One fixed three-hour point seed stationarity and linear-feasibility preflight

Use the exact PR #218 alternate-control seed (`control[21]` flow
fraction `-0.59`) with the unchanged PR #204 4×5 point-observation
problem, 13 parameters, verification target, 0/10/20-minute
observations, 18 forecast leads, boundaries and 3,600-stage layout.
The seed passed the pointwise and full strict branch checks but has
**not** been shown stationary. Pin its control/input/source/plan
identities and do not tune another seed after seeing this result.

In one guarded child, rebuild the fixed fixture. Run a separate
no-grad forecast pass with a detached observer to record compact
minimum scale-normalized margins over the 3,600 stages, then recheck
the full strict branch at that seed. This order lets the collector
observe a failing stage before the nested strict oracle could raise.
If either pass stops early, mark margin coverage incomplete rather
than reporting a 3,600-stage minimum. Record:
interior left/right slopes and active-limiter operand gaps divided by
`max(abs(q))`, and absolute face flux divided by the global maximum
absolute face flux. Preserve the original `128*eps` strict predicate;
these margins are **diagnostics**, not a new acceptance threshold or
finite-perturbation certificate. Record the stage/phase of each minimum.

Compute the **analysis objective** `J(c,p)` and exact control gradient
`g=grad_c J` at the unchanged seed. Check finite CPU FP64 shape,
`||g||_inf` against the existing `<1e-10` criterion and report the
field-control and dynamics-control gradient norms separately. Do not
use the terminal score `E` as the stationarity target or evaluate it
merely to claim progress. The objective includes the two analysis
intervals (360 SSPRK stages); the terminal score forecast includes
18 future intervals and is not part of this Newton equation.

If `||g||_inf >= 1e-10`, attempt **one read-only linear solve**
`H s=-g` with exact `H v=JVP_c(grad_c J;v)` and existing PCG at
relative tolerance `1e-10`, maximum 104 iterations, no preconditioner.
Recompute `||H s+g||_2/||g||_2` from one additional exact HVP, record
iterations, all HVP calls and any non-SPD/iteration-budget refusal.
Do not apply `s`, run line search/refinement, issue adjoint/VJP, or
perform nonlinear reanalysis. PCG's visited positive directions do
not prove the 26×26 Hessian globally SPD; even a converged solve is
only this RHS's linear feasibility. If the seed already meets the
gradient criterion, skip the zero-RHS PCG and leave curvature
unverified.

Run only once in a fresh `point_3h_seed_linear_attempt1` directory.
Use a **600-second wall-limit trigger** plus reap grace and **sampled
1 GiB child RSS**; report parent and child phases/costs separately.
The parent must not rebuild the full FV fixture outside the child
guard. Distinguish process/resource failure, branch refusal,
nonfinite derivative, PCG refusal, linear pass and stationarity pass.
Do not retry if any budget or mathematical gate fails. Preserve raw,
resource, parent and source/input evidence. Run focused/fake-guard and
affected tests, typecheck, incremental Graphify update and GREEN/RED
prelaunch review before the numerical launch.

This preflight cannot close R5-R-R. A separate branch-aware stationary
search, final gradient and curvature checks, true adjoint residual,
full 13-vector VJP and two signed endpoint sizes would still be
required. It makes no finite-influence or physical forecast claim.
