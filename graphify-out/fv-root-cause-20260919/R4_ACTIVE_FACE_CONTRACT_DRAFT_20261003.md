# R4 active-face contract draft: not yet qualified

Status: mathematical and source feasibility only. PR229 closes the fixed-slice scalar-J veto, but original full stationarity remains unresolved. This draft defines a possible selected local optimizer response for the unchanged nonsmooth objective; it is not an issued response, a proved cusp, or an implementation result.

## Coordinate family and unchanged objective

Let alpha_k=L_k*tanh(u_k), Q=w^T alpha, w=(0,-1,-2,-1/2,-2), pivot p=1, eta=Q/0.08. For eta=0, alpha_p=-sum_{k!=p} w_k alpha_k/w_p and u_p=atanh(alpha_p/L_p). Keep the other latent controls unchanged. Reject |alpha_p/L_p|>=1 without clipping. The independent nonpivot box allows |alpha_p| up to0.22, while L_p=0.08; rank/CFL validity does not replace this domain guard.

The equivalent streamfunction basis on this family is B'_k=B_k-(w_k/w_p)B_p, k=(0,2,3,4). For the captured basis all four projected modes have exactly equal vertices [2,0] and[2,1], with entries(2,0,-2,0), so Qy[2,0] is structurally zero. Deriving both face-flux arrays from their common streamfunction retains the discrete divergence identity. The projected matrix has rank4 and its independent box passes the existing CFL validator (reported bound about0.15533<0.5). Original coefficient/pivot constraints remain authoritative.

A native4-mode representation changes floating operation ordering, even though it is the same real linear streamfunction on the constraint. Validate primal fields, gradient and HVP against the original5-mode lift; use an independent precision reference where scalar cancellation obscures agreement. Do not merely overwrite a computed face flux or erase its uncertainty based on its small value.

For tangent vector t in R25 and unchanged p in R61, compute the reduced observation residual with a dataclass-replaced4-mode FrozenOuterState, but apply the original cost/prior to C26(t,0):

    j(t,p) = robust_from_residual(C26(t,0), residual4(t,p), obs_p, frozen_original(p)).

The prior remains .5||C26||², including the recovered pivot latent and original field smoothness. Replacing it with .5||t||² changes the statistical problem. Observation values/masks, symmetric whitening, initial background y0+theta*P, growth/time schedule and future boundaries remain unchanged. Score must use the same reduced forecast family and the original fixed verification field. Existing FVAnalysisTransport, FrozenOuterState and native trajectory/forecast functions suffice; no new FV solver is needed.

## Sufficient local theorem (conditional)

Assume C(t,eta) is a regular chart near(t0,0,p0). Assume j admits continuous branchwise C2 extensions j_- andj_+ for eta<0 andeta>0, with both agreeing on the active face and all other limiter/upwind choices locally fixed. If

    j_t(t0,0,p0)=0,
    H_tt=j_tt(t0,0,p0) is positive definite,
    sigma_- = partial_eta j_-(t0,0,p0) < 0,
    sigma_+ = partial_eta j_+(t0,0,p0) > 0,

then continuity keeps the normal slopes strictly oriented toward eta=0 in a sufficiently small neighborhood. At any nearby t, eta=0 locally minimizes in the normal direction. Positive tangent curvature gives an isolated strict tangent minimum. Together these conditions give a selected local minimum of the original unconstrained nonsmooth problem, not merely a minimum of a different constrained objective. This is a local statement; it does not rule out other minima or establish global convergence.

Strict gaps, regularity and branch continuity are hypotheses to verify, not consequences of finite-offset signs. The current four finite slices do not prove them. Native max/min at Q=0 uses a library subgradient; it does not supply both one-sided normal derivatives. Tangent Hessian must include chart curvature, including sum_i J_c_i C_i,tt if computed through the original coordinates.

If these conditions persist under p, the constrained IFT gives

    H_tt dt*/dp = -j_tp,
    H_tt^T lambda = E_t,
    g_p = E_p - j_tp^T lambda.

Here Q_p=0 because the fixed flow basis/limits do not depend on the observation/background parameters. Otherwise constraint parameter derivatives must be included. This is a response of the explicitly selected active-face local optimizer; never route it through the old smooth26-control stationarity gate or label it a general minmod response. Theta/direct background dependence and inactive observation slots must be retained.

## Required packet before issuing a response

- [x] Establish projected-basis rank, exact vertex equality and native CFL-box feasibility with the captured fixed inputs; no optimizer was run for these primitive checks. Fixture creation includes its existing fixed forecast setup.
- [ ] Build a thin family binding preserving the original26-control prior and all data/time/boundary dependencies; independently check operators/derivatives and outside-domain refusal.
- [ ] Obtain fresh tangent stationarity and full25-column Hessian symmetry/positive-curvature evidence; record inverse-curvature/error scales rather than only a gradient threshold.
- [ ] Certify all other face and limiter margins; exempt only the named structurally zero face.
- [ ] Compute correctly signed one-sided normal derivatives at eta=0 with resolved error bounds. Finite samples or averaged subgradients are insufficient.
- [ ] Define the parameter neighborhood and preserve the active/other branches there; reoptimize signed parameter endpoints at two adjacent sizes.
- [ ] Only then compute and verify tangent adjoint/full61-vector response and direct/indirect/total score sensitivity.

Q=0 is an interior no-normal-volume-flow face, not clear sky or missing data. Prior-stabilized positive curvature is not observational flow identifiability. This synthetic short-lead contract does not close the independent physical/learning or3-hour response milestones. GREEN/RED reviewed feasibility/theory; unresolved conditions remain explicit.

Source anchors: variational.py FVAnalysisTransport/FrozenOuterState(3096ff), _analysis_trajectory(4885ff), _control_prior_residual(5476ff), robust_from_residual(6527ff), _validate_control(9928ff); fv_research_problem.py contract/objective/forecast(157ff); transport.py face_volume_fluxes(368ff), bounded_fv_coefficients(702ff).
