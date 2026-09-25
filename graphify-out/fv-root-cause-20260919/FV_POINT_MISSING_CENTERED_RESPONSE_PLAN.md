# R3 one missing middle-time point: stationary response plan

This is a **new incomplete-observation profile** derived from the
constructed fixed-centered-prior problem in PR #200. It is not a rerun or
solution of the original zero-centered-prior point input. Preserve the
4×5 FV state, complete initial and boundary support, fixed theta,
fixed nonzero six-component dynamics prior mean, verification target,
time grid, point sampler, per-time symmetric correlation convention and
all numerical tolerances.

At the middle observation time, mark point index 1 (flattened parameter
index 5) as **genuinely missing** (`status=1`). Replace its stored dBZ
value by the required canonical inactive fill, not by a detected clear-sky
measurement. All other eleven observations retain the constructed
model-generated detected values. Rebuild `FVPointResearchProblem` so the
middle-time correlation whitening uses the principal submatrix for its
three detected points. The direction adds one dBZ to the other three
middle-time points, parameter indices 4, 6, and 7; the missing slot and
theta do not move. Include status, canonical fill, fixed prior mean,
actual observation values and direction in the input identity.

The constructed control is the same 26-vector as PR #200. At that
control, the eleven active observation residuals and centered prior
components should be zero. Verify fresh gradient maximum `<1e-10`,
finite/symmetric/SPD exact 26-column Hessian, and the full 54-stage
nominal minmod signature with slope and face margins `>1e-4`.

Reuse the existing matrix-free `compute_local_response` and exact-HVP
PCG tangent; do not duplicate the objective, score or solver. Verify
the full 13-component direct/indirect/total VJP and independent true
adjoint residual `<=1e-10`. The missing observation's direct, indirect
and total gradient component (index 5) must be zero to absolute
`1e-12`; changing its stored parameter alone must leave the objective,
score and control gradient invariant at the nominal point. Preserve
the fixed mask and principal-submatrix whitening at every endpoint.

For each `h=0.001,0.0005` dBZ, reanalyze actual `p±h*d` from
`c0±h*c_dot`, using unchanged Newton/PCG/Armijo and fixed full nominal
branch plus `>1e-4` margins. Require all four final gradients `<1e-10`
and compare the signed score central difference to the adjoint response:
relative error `<1e-4` when `|s|>=1e-8`, otherwise absolute error
`<1e-10`, at both sizes; the smaller h must reduce absolute error.
This validates one local active-observation direction, not every
component's nonlinear response or any finite observation removal.

Run one guarded serial child with 600-second wall and sampled 1-GiB
child-RSS limit. Pin current implementation, plan, baseline archive,
derived input and parent source hashes before/after. The parent must
independently recompute the Hessian, adjoint/full VJP, tangent, endpoint
objective/score/gradient/branch, inactive gradient and signed central
differences before publishing `passed`. Keep execution success,
numerical eligibility and response validation separate. Unknown errors
are execution failures, not silent scientific refusals.
