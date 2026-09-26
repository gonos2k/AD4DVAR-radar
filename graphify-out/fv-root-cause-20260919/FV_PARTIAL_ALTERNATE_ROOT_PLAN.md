# R4-R one exploratory alternate-sector root-only refinement

Start **directly** from the post-hoc PR #212 seed, iteration
5/backtrack 3 of PR #209's rejected trial archive: control SHA256
`125e0200fdf85f996715ba1bc04aa3f631333614bf15cf2c16630b94c1897421`,
full branch signature `50d3b1a4bad6761b14806c62708400d87d16e506ab08ef545f7ba9d870bece6d`.
Pin PR #212's seed-gate manifest SHA256
`959ab714de6f862a85fb0ee6ef8353fbfc45098916217a83a0d6acb7ef8f8fa5`
and its child/parent/resource records. Require the same two-hole
4×5 observations, prior, FV transport, boundary and fixed
verification; compare the PR #209 raw candidate hash and current
input/source identities. This is **one exploratory restart chosen
after seeing the earlier trial table**, not independent basin
validation. The seed was previously `merit_switch_refused` on that
path and is not a stationary point.

PR #212 measured this exact seed at `||grad_c J||∞=0.0068481175`,
54 strict stages, slope/face margins `4.0606113e-4` /
`1.1662714e-4`, and an exact 26-column SPD Hessian with minimum
eigenvalue `1.0167708533`. Reuse that seed curvature only after
verifying its manifest, child `seed_locally_spd`, parent `completed`,
child exit 0 without resource termination, selected-control **and
curvature-control** SHA256s, exact 26-column SPD predicate, and
source/input hashes. Match the recorded Python/PyTorch/CPU FP64
environment; if it differs, fail this one-shot rather than assuming
the old curvature applies. Recompute the current seed objective,
gradient 2-norm and maximum, full branch signature and both margins
and compare every scalar to the PR #212 record at
`rtol=1e-10, atol=1e-12`; any discrepancy is an identity/evidence
failure, not a reason to choose another seed. Do not invoke product GN or substitute a
different candidate if any gate fails. A separate final Hessian
audit is still mandatory if a root is found.

Solve the unchanged original partial-observation stationarity
equation (F(c)=\nabla_cJ(c,p)=0) with the existing exact
gradient-JVP Newton–PCG refiner, maximum **8 Newton iterations**,
**16 backtracks** per iteration, **104 PCG iterations** per solve,
and independently checked actual linear relative residual
`<=1e-10`. Each candidate must pass the full 54-stage roundoff-aware
minmod/upwind oracle and have finite **positive** scaled slope/face
margins. Within an unchanged full limiter/face-sign signature use
normalized gradient-merit Armijo. For a changed signature, accept
only if the **measured** (\Phi=\tfrac12\|F\|_2^2) decrease exceeds
`128*eps64*max(|Phi_old|,|Phi_new|,tiny64)`; relinearize after an
accepted switch. This endpoint policy does not certify a smooth
path through a switch or global convergence. Even matching full
signatures at the endpoints cannot prove the intervening step stayed
within that sector.

Record source/input/plan before and after, the exact archived seed,
fresh seed objective/gradient/branch, every candidate hash and
J/gradient/branch/policy decision, PCG iterations/HVPs and true
linear residuals. A qualified classical root requires a fresh
`||grad_c J||∞<1e-10`, full exact finite/symmetric/SPD Hessian
audit (relative asymmetry `<=1e-10`, positive minimum eigenvalue,
eigenvalue ratio `>sqrt(eps64)`), and a fresh 54-stage branch with
both scaled margins **strictly >1e-4**. A stationary point below
that final margin is `root_low_margin` **only after** the gradient
and exact final Hessian gates pass; it remains response-ineligible.
Only full stationarity, exact final SPD and both strict final margins
produce `root_margin_qualified`. Failure of the exact final curvature
gate is `root_refused` at `final_curvature`, not a low-margin root.
Numerical line-search/PCG/iteration refusal, resource termination,
and callback/identity execution errors are separate. None proves
root absence. Do **not** compute adjoint, VJP, tangent or signed
nonlinear reanalysis in this experiment.

Launch exactly one serial child under a **600-second wall** and
**sampled 1-GiB child-RSS** ceiling. Pin reviewed plan, new probe
and guarded runner hashes at launch; preserve PR #209/#212 raw
archives. No post-hoc budget, seed, margin or objective change is
allowed within this attempt. A successful root would authorize only
a separately planned response/endpoint validation.
