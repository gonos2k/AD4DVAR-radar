# One exact 26-column Hessian audit at the fixed three-hour point seed

PR #219 found a strict 3,600-stage branch at the PR #218 shifted
control (`control[21]` fraction `-0.59`), but the analysis objective
gradient maximum was `2.254729681485403` and an exact-HVP PCG solve
for `H s=-g` refused its SPD premise after 15 HVP calls. That Krylov
event alone does not establish the full Hessian's inertia. This audit
tests **that one seed only**, with the unchanged PR #204 problem,
observations, background parameter, target, time and boundaries.

In one guarded child, reconstruct the fixed seed and independently
confirm its original strict branch (3,600 stages/signature), then
compute the exact analysis objective control gradient
`g=grad_c J(c,p)`. Build the 26×26 matrix column by column using
`H e_j=JVP_c(grad_c J;e_j)` for every standard basis vector. No
Gauss–Newton or finite differences. Record every HVP call, finite
shape/dtype/device, `||H-H.T||_F/||H||_F`, eigenvalues of the symmetric
part, and `lambda_min/lambda_max`. Predeclare relative symmetry gate
`1e-8` for CPU FP64; if `||H||_F=0` or symmetry fails, do not assign
a Hessian inertia. If symmetry passes, apply one additional **fresh**
exact HVP to the unit minimum-eigenvector and require
`||Hv_min-lambda_min v_min||_2/||H||_F <=1e-8` before interpreting
curvature. Use the conservative absolute sign uncertainty
`delta = max(1e-8*max_j|lambda_j|,
           ||H-H.T||_F/2 + ||Hv_min-lambda_min v_min||_2
             + 128*eps*||H||_F)`.
Classify measured negative curvature only when `lambda_min < -delta`,
positive curvature only when `lambda_min > delta`, and the intervening
band as near-zero/inconclusive. This is a numerical audit of the
measured symmetric part, not a rigorous spectral error bound. If the
spectral scale is zero, classify as
inconclusive. Emit `lambda_min/lambda_max` as null whenever
`lambda_max` is zero or nonfinite, even if another eigenvalue makes
the spectral scale nonzero. Before hashing
the minimum-eigenvector, make its largest-absolute component positive
(first index breaks ties); this removes the arbitrary eigensolver sign.
A negative eigenvalue with a verified residual is evidence
of local Hessian indefiniteness at this point, not of root absence or
global nonconvexity. Record the Hessian SHA256 and eigenvector SHA256
without publishing raw matrices as a general API.

Do not take a Newton step, retry PCG, change the seed, evaluate the
terminal score, compute an adjoint/VJP or signed reanalysis. Source,
plan, archived input and candidate control identities must match the
PR #218/#219 records before and after. The parent must not rebuild the
FV fixture. Distinguish process/resource success, strict branch,
gradient, symmetry/eigenpair qualification and curvature verdict.

Run once in fresh `point_3h_seed_hessian_attempt1` under a
**600-second wall-limit trigger** plus reap grace and **sampled 1 GiB
child RSS**. Preserve raw/guard/parent artifacts and phase costs; do
not retry if any gate or resource bound fails. Run focused/fake-guard
and affected tests, targeted typecheck, incremental Graphify refresh,
and GREEN/RED prelaunch review before launch.

This is a small dense **audit** of the existing exact HVP, not a
dense production solve or a stationary response. R5-R-R remains open
until a separately declared numerical policy reaches and validates a
strict stationary point and its full response/endpoints.
