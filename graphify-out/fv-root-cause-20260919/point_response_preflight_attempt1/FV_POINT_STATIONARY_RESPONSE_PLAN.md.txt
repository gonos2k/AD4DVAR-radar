# R2 bounded full-valid point-observation stationary response plan

This plan is for one 4×5 CPU FP64 synthetic point problem. The initial
control from `fv_point_research_case.make_case()` is a **warm start**, not a
stationary analysis for this objective. Its prior measured maximum control
gradient is 14.327. The fixed verification field is synthesized and detached
before any optimization; this is a local numerical response test, not weather
forecast skill.

The first profile keeps all 12 off-grid dBZ point observations detected,
the four fixed interior coordinates, the exogenous 4×5 background and
fully known boundaries. The same-time four-point correlation is fixed to
the identity except `C[0,2]=C[2,0]=0.3` and
`C[1,3]=C[3,1]=-0.2`; its symmetric principal inverse square root and
the existing pseudo-Huber cost are unchanged. The response direction is
`d[4:8]=1` dBZ at the middle time, with all other observation and theta
components zero. The parameter layout is 12 observations plus theta.

Run a no-solver preflight first, capped at 120 seconds and sampled 1 GiB
child RSS. It must record source and archived-input SHA256, current problem
and tensor identities, 54-stage strict branch signature, slope and face-flux
margins, objective, score and warm gradient. The warm trace requires both
the minimum scaled minmod-slope margin from `inspect_branches` and the
minimum over all 54 stages of `min(abs(face flux))/max(abs(face flux))` to
exceed `1e-4`; the face denominator must be finite and positive. If this
preflight fails, no
optimizer/adjoint is launched. The source and input identities must be
rechecked before numerical publication.

One serial numerical child is capped at 1200 seconds and sampled 1 GiB RSS.
It first attempts `refine_stationary` on the warm point objective with the
fixed initial branch; 8 Newton steps, 16 backtracks, default PCG cap 104,
actual Euclidean relative Newton residual
`||H s + grad J||_2 / ||grad J||_2 <= 1e-10` and maximum control-gradient
component strictly below `1e-10` in the fixed current control/objective
units. Every candidate is checked against the same full limiter/face-sign
signature and `1e-4` scaled slope/face margins before Armijo evaluation.
A failed warm correction is recorded as a refusal;
no tolerance or strict branch gate is relaxed within that attempt. A
separately specified point-objective basin search would then be needed.
**The direct warm-root attempt is only a bounded feasibility diagnostic;
successful response validation is not expected or assumed from it.**

If a qualified point is obtained, compute a fresh dense 26×26 Hessian from
the exact gradient-JVP operator for a local curvature/symmetry diagnostic.
Require all entries finite and
`||H-H.T||_F/max(||H||_F,tiny64) <= 1e-10`; eigensolve
`(H+H.T)/2`, require finite `lambda_max>0`, `lambda_min>0`, and
`lambda_min/lambda_max > sqrt(eps64)`
before using the SPD PCG response. Then run `compute_local_response` with
the point branch pinned at the actual nominal solution; require a true
transpose-adjoint relative residual at most 1e-10, complete 13-component
direct/indirect/total gradients, and equality of full-gradient projection
with the direction-specific JVP response to relative 1e-6. If the total is
near zero, require absolute difference at most
`1024 eps64 (||g_p||_2 ||d||_2 + |s_d| + tiny64)` instead. For this fixed score, the
observation-value direct term should be zero.

Only after those gates, try signed nonlinear endpoint reanalyses at
`h=0.001*2^-j` dBZ per middle-time point, `j=0,...,5`, in that order. Use
the independent tangent `H c_dot = -J_cp d`, with its own actual Euclidean
relative residual `||H c_dot + J_cp d||_2/||J_cp d||_2 <= 1e-10`, to form and record signed
predictors `c* ± h c_dot`; refine each actual `p ± h d` to the same
stationarity gate, with the frozen verification field. Require strict
54-stage branch signatures equal to the nominal signature, positive declared
minmod/face margins above `1e-4`, and record the endpoint distance from the
nominal point and tangent predictor as local-continuation diagnostics.
Only endpoints whose distance from nominal is at most
`2h*max(||c_dot||_2,1)` count as a local comparison; this gate is not a
uniqueness or full-path proof. A signed pair is atomic: both endpoint
refinements and gates must pass. At least two accepted **adjacent j values**
must have central score slope `D_h=(E_+-E_-)/(2h)` versus the nonzero adjoint
direction `s_d` satisfying `|D_h-s_d|/|s_d|<=1e-4`; any rejected pair resets
the consecutive count. Reject any h whose expected signed
score separation is at or below `1024 eps64 max(|E_nom|, tiny64)`; retain
all refusals and both signed endpoint values. Endpoint agreement alone is
not a certificate that every point on the nonlinear solution path retains
the limiter branch, so no finite-impact range is claimed.

Execution exit, numerical stationarity, local-response eligibility,
independent signed validation, and physical skill remain separate fields.
Any failure before the signed comparison leaves `response_validation`
`not_established`; it must not be turned into a pass by changing the
predeclared thresholds. The resource guard samples child RSS every 250 ms
and terminates on an observed cap violation or wall timeout; it is not an
OS hard-memory limit and can miss between-sample spikes. Preserve old point
and collocated reports untouched.
