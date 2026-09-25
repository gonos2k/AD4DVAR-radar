# R2 constructed centered-prior point response integration plan

The original full-valid correlated point problem remains unresolved after
seven bounded numerical attempts. This is a **different, explicitly
constructed statistical problem** for testing whether the existing point
FV observation path can connect a qualified stationary point, matrix-free
adjoint/full parameter VJP and signed nonlinear reanalysis. Its success
must not be reported as solving the original zero-centered-prior input,
arbitrary GN convergence or physical forecast skill.

Start from `fv_point_response_preflight.fixed_problem()` on the 4×5 grid,
with 26 controls, 12 fixed off-grid dBZ point observations over three
times, one theta parameter, the same fixed 4×4 correlation matrix and
complete model state/boundaries. Let `c0` have 20 zero initial-field
increments and the six nonzero dynamics controls from the archived warm
control. Freeze `mu=c0[-6:]` as a **fixed Gaussian dynamics-prior mean**;
it is not learned and is not included in differentiable parameters.
Generate the 12 synthetic observations from the model trajectory at
`c0` and the original theta. Generate a fixed verification field from
the model forecast at `c0` plus `0.1*background_pattern`. Reconstruct the
`FVPointResearchProblem` with these fixed observations and verification;
set `p=(flattened observations,theta)`.

The centered objective is explicitly

`J_mu(c,p)=J_point(c,p)-0.5*||c_dyn||²+0.5*||c_dyn-mu||²`.

For this profile the point problem has no learned neural prior, so its
original dynamics prior residual is exactly `c_dyn`. The wrapper changes
only that prior mean; the FV trajectory, point sampler, symmetric
whitening, pseudo-Huber data term, field regularization, boundary,
score and observation meaning remain unchanged. Include `mu` and the
wrapper equation/version in the input identity. The synthetic
construction makes residuals and all prior components zero at `c0`, but
the program must independently confirm fresh gradient maximum `<1e-10`,
finite/symmetric/SPD 26-column exact Hessian and the 54-stage core
minmod branch with **both** scaled slope and face margins `>1e-4`.

Compute one `compute_local_response` at `c0` with the full 13-component
direct/indirect/total parameter gradient and the predeclared direction
`d`: add `1 dBZ` to each of the four *middle-time* point observations,
zero elsewhere. Check the independent true transposed-adjoint relative
residual `<=1e-10` and the full-vector projection against the separate
directional JVP response to relative `1e-6` (or absolute `1e-12` when
the directional response is smaller). Record direct/indirect/total terms and any
cancellation; no dense Hessian may be used by the response solve.

For signed nonlinear validation, form the tangent by PCG of the same
exact HVP equation `H c_dot = -J_cp d`, with true relative residual
`<=1e-10`. At `h=0.001` and `0.0005 dBZ` per selected point, set
`p±=p±h*d`, start each endpoint from `c0±h*c_dot`, and use the existing
`refine_stationary` with unchanged `1e-10` stationarity, PCG and Armijo
criteria. Every candidate and final endpoint must pass the core oracle
and retain the exact nominal full limiter/face-sign signature and both
`>1e-4` margins. Compare the *actual reanalyzed* central score
difference with the adjoint response. Require signed relative error
`<1e-4` when `|s|>=1e-8`, otherwise absolute error `<1e-10`, at both
sizes; require the smaller size's absolute error to be lower. No
finite-amplitude validity or independent skill claim follows.

Use one guarded serial child, 600-second wall and sampled 1-GiB child RSS
limit. Pin source, input, plan and baseline archive SHA256 before and
after; record each endpoint's parameter/control hashes, objective,
gradient, exact branch/margins, score, PCG/refinement diagnostics and
stage-level timing. The parent checks child PID, exit, resource status,
elapsed/RSS and all identity gates before classifying completion.
Unknown errors are execution failures, not scientific refusals.
