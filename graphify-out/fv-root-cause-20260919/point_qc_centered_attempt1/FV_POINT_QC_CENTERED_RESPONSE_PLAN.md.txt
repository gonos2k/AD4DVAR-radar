# R3-Q one externally QC-excluded point: stationary response plan

This profile is distinct in **meaning and identity** from the status-1
genuinely missing point verified in PR #201. It reuses the constructed
fixed-centered-prior 4×5 FV problem, complete state and boundaries,
fixed theta and verification, exact point sampler, and fixed within-time
correlation convention. An external QC decision, not a QC algorithm
implemented here, marks middle-time point 1 / flattened parameter 5 as
status **2** (QC rejected). Its stored number is the canonical inactive
fill. Eleven other observations remain detected; this is neither a
clear-sky detection nor a genuinely missing status-1 record. The
middle-time correlation whitener must use the three detected points'
3×3 principal submatrix.

Keep the same fixed nonzero dynamics prior mean and constructed control
as PR #200/201. Rebuild the immutable point-problem status/cache and
include status-2 provenance in its input identity. The signed direction
adds one dBZ only to active middle-time indices 4, 6 and 7; index 5 and
theta remain unchanged. Fresh nominal maximum gradient must be
`<1e-10`; exact 26-column Hessian must be finite/symmetric/SPD; the
54-stage nominal minmod branch must retain slope and face margins
`>1e-4`.

Reuse the existing parameterized point-response child/parent, exact HVP,
matrix-free adjoint/full 13-component VJP and tangent PCG. The QC
excluded parameter's direct, indirect and total gradient component must
be zero to absolute `1e-12`; changing its stored parameter alone by
`+0.25` must leave nominal objective, score and control gradient
unchanged. Parent must independently recompute the same invariance,
full derivatives and nominal Hessian. Do not learn QC weights or treat
the external exclusion as a differentiable selection decision.

At `h=0.001,0.0005` dBZ per active middle-time point, reanalyze
`p±h*d` from tangent predictors with unchanged Newton/PCG/Armijo
criteria. All four endpoints need fresh gradient maximum `<1e-10`,
same full nominal signature and `>1e-4` slope/face margins. The signed
score central differences must match the adjoint direction to relative
`<1e-4` if `|s|>=1e-8` or absolute `<1e-10` otherwise, with smaller
absolute error at smaller h. Parent forms those differences from its
**own recomputed** endpoint scores. This is one local active-value
perturbation, not a finite QC decision impact or general QC algorithm.

Use one serial child under a 600-second wall and sampled 1-GiB child-RSS
guard. Pin current code, declared plan, archive, full input/mask and
source identities before/after. The parent checks PID/command/resource,
source/input/plan identity, exact numerical predicates and all endpoint
evidence before publishing. Keep execution status, numerical eligibility
and response validation separate. Original zero-centered-prior input,
whole-empty-time profile, long-horizon point sensitivity and physical
skill remain open regardless of this result.
