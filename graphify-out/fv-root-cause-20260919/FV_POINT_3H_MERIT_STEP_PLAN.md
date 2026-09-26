# One branch-aware joint-descent trial from the indefinite three-hour seed

PR #220 verified one negative eigenvalue in the analysis-objective
Hessian at the fixed PR #218 shifted control. The seed remains
nonstationary (`||g||_inf=2.254729681485403`) even though its full
3,600-stage minmod branch passes. Do not rerun unshifted SPD Newton
from this seed or modify its 13 observation/background parameters,
terminal score definition/target, 0/10/20-minute observations, 18
forecast leads or boundary schedule.

This is **one exploratory root-search step**, not an optimizer or
response. In one guarded child, rebuild the fixed seed and strict
branch, evaluate the unchanged analysis objective `J(c,p)`, its
control gradient `g`, and **one** exact `Hg=JVP_c(grad_c J;g)`.
Define stationarity merit `Phi=||g||_2²/2` and the candidate direction
`d=-Hg/||Hg||_2` in the existing 26 latent control coordinates. Do
not silently use another direction. Require finite nonzero `||Hg||`
and `gᵀHg > 128*eps*||g||_2*||Hg||_2`; otherwise return an explicit
`no_joint_descent_direction` refusal. With the seed's measured
symmetric-Hessian condition (PR #220 evidence SHA256
`77bf193a2f891748199c12c0a318bd47fcb0f199a23da3f0e926d7fa4ebf8ed0`,
relative symmetry defect `1.0949379174674463e-15`), this gives local negative slopes
`D J[d]=gᵀd<0` and `D Phi[d]=(Hg)ᵀd<0`, but not a finite step.
Record both slopes, direction hash, field/dynamics block norms and
maximum component. Euclidean normalization is a declared latent-space
coordinate policy, **not** a scale-invariant physical trust region.

Try only `alpha=0.02*2^{-k}` for `k=0,...,7` (one candidate per
backtrack, at most eight). Require finite candidate control, the
**original complete 3,600-stage strict branch oracle** at each
endpoint, then freshly evaluate J, g, Phi and gradient maximum using
the changed control but unchanged p. A candidate is accepted only if
both J and Phi have strictly decreased by more than their own
`128*eps*max(abs(seed_value),tiny)` roundoff floors. For an endpoint
with the seed signature, also require both Armijo tests with
`c1=1e-4` and the measured seed slopes. For a changed strict
signature, use only actual two-scalar decrease, label
`branch_changed=true`, and stop after that single accepted endpoint;
the seed's derivative does not certify the crossing path. Equal
endpoint signatures also do not certify the segment between them.
Preserve
every refusal reason and costs, do not retry with a different alpha
grid or direction.

This one step returns at most a **candidate** with its 26 control
values/hash, branch signature, J/Phi/gmax and source/input evidence.
Even if gradient happens to be small, do not label it a qualified
root or issue any adjoint/VJP/reanalysis. A later separately budgeted
handoff must recheck final branch, `<1e-10` gradient, exact unshifted
Hessian suitability and original-H true adjoint residual before an
implicit response. No damping or modified Hessian enters that final
response.

Use one fresh `point_3h_merit_step_attempt1` directory and one child
under a **300-second wall-limit trigger** plus reap grace and
**sampled 1 GiB child RSS**. The parent only checks archived identity,
source/plan/resource/child records and must not build the FV fixture.
Return a completed numerical refusal if no candidate meets the
declared direction, branch or merit gates; never infer that no root
exists. Run focused/fake-guard and affected tests, typecheck,
incremental Graphify refresh and GREEN/RED prelaunch review before
launch. Preserve raw/guard/parent artifacts and phase costs. No
physical-skill or finite-observation-impact claim follows.
