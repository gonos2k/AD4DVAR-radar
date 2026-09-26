# One alternate control passes the fixed three-hour strict branch

The predeclared single alternate control passed the full 3,600-stage
strict minmod branch oracle for the unchanged PR #204 point-observation
problem, terminal score definition and verification target. This
establishes a **smooth branch at this particular control**, which the
original coefficient-cancellation
control did not have. It does not establish an analysis stationary
point, local Hessian suitability, adjoint/VJP response or nonlinear
observation impact.

The streamfunction algebra makes the original `q_y[3,1]` zero a
coefficient cancellation, not a structural zero: for the five basis
coefficients,

`q_y[3,1] = -(a_X + 3 a_XY + 1.5 a_quadratic + 9 a_X2Y)`.

The original fractions `[0.7,-0.6,0.5,0.4,-0.3]` cancel at this
face. The only tested candidate changes the second flow fraction to
`-0.59`, through zero-based `control[21]=atanh(-0.59)`. The 13
parameter values, original problem/observation identity, verification
target, 0/10/20-minute observations, 18 ten-minute future leads and
boundary schedule are unchanged. The altered control has SHA256
`97ce3a636b25d79ca0aab208035e4299f9f4ad26ea00d557531e7a9eded2080b`.

The child used the production coefficient mapping and face-flux
operator. All 49 faces passed the static nonzero-flux check: minimum
absolute flux `0.0008000000000000229`, maximum `0.1592000000000001`,
strict threshold `4.5247361413203206e-15`, and zero near-zero faces.
The following trajectory observer accepted all 3,600 callbacks. The
separately invoked original `FVPointResearchProblem.branch_check`
also returned 3,600 limiter-choice and face-sign stages; the compact
signature SHA256 is
`80a1fac22b536bfedcedfda506e11d85f4b9f2a73c5610c73fa8dfdb210dfe51`.
These are pointwise branch checks at one control, not a finite-segment
certificate or a stationarity result.

One guarded child (PID **43857**) exited 0 with no resource termination
or monitor error. The resource guard measured **9.433 seconds** elapsed
and 35 child-RSS samples with peak **235,323,392 bytes** under the
180-second wall trigger and sampled 768 MiB limit; the raw child
recorded 8.144 seconds internal elapsed. The guarded child rebuilt
the original 3,600-stage synthetic truth fixture, then ran one
3,600-stage candidate forecast for pointwise checks and a second
3,600-stage forecast inside the original strict oracle. The parent did
not rebuild the FV fixture outside the guard. Source, archived input,
candidate parameters, plan and after-run identities matched. Both reports say
`stationarity_passed=not_tested`, `response_computed=false`,
`response_validation=not_performed`, and
`physical_validation=not_performed`.

The final affected suite passed **65 tests** with 18 existing
TorchScript deprecation warnings; 19 focused tests passed. Error-level
basedpyright 1.39.9 reported 0 errors/warnings/notes. GREEN/RED
reviewed the one-shot plan, fixed a parent input snapshot and
candidate-record consistency checks before launch, and found no
prelaunch blocker. Graphify's targeted code refresh produced a valid
graph with 8,051 nodes and 206,571 links. Exact file hashes and
commands are recorded in
`FV_POINT_3H_SHIFTED_BRANCH_EVIDENCE.json`.

This closes the **alternate-control strict-branch feasibility**
subitem only. R5-R-R remains open. A separately budgeted next step
must measure objective gradient and local curvature at this seed,
then apply a declared branch-aware stationary search without weakening
the final `<1e-10` gradient and true adjoint residual gates. Only a
qualified final branch can be used for a full 13-component VJP and
two adjacent signed reanalysis sizes. No physical forecast-skill
claim follows from this synthetic same-operator test.
