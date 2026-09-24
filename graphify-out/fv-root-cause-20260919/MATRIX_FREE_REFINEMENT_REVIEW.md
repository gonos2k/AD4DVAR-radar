# Matrix-free stationarity refinement — new milestone after PR172

Baseline: merged PR172, `cee7a387`. Its input isolation and current
GN → explicit refinement → whole response integration are closed.

## Numerical contract declared before measurement

For fixed parameters p, solve F(c)=grad_c J(c,p)=0 with exact gradient-JVP
products H(c)v. PCG solves Hs=-F without constructing a Hessian or applying
curvature damping. Recompute Hs independently and require
||Hs+F||/||F|| <= 1e-10. The merit function is Phi=||F||²/2, with slope
F^T Hs. Accept a caller-branch-checked trial only when the slope is finite,
negative, and the squared gradient norm satisfies Armijo (c1=1e-4).
Stationarity remains max|F| < 1e-10; no objective/gate/physics changes.

PCG assumes a smooth symmetric positive-definite local Hessian. Positive
curvature in its visited directions does not certify the whole spectrum.
A stationary return is not a positive-definite/global-minimum certificate.
The actual minmod adapter and downstream response gates remain mandatory.
Candidate-point checks do not certify the intervening finite path.

The refiner is a numerical root solver, not an unrolled differentiable
optimizer. Its workspace is detached; the response still computes actual
objective/score derivatives at the returned control with original parameters.

## Planned validation and cost boundary

1. Predetermined analytic nonlinear SPD cases, including 64 controls above the
   old dense oracle limit; refusal cases and actual linear-residual checks.
2. Reuse the saved 26-control product GN result and fixed score, mask, boundary,
   observation and branch contract. Run the new refiner and existing full
   response once; compare with archived dense correction/reference response.
3. Before execution: control distance <=1e-8, full-gradient and all three
   directional relative differences <=1e-6, actual adjoint relative residual
   <=1e-10. No new GN or nonlinear observation reanalysis is needed.

The FV bridge is limited to one CPU run with a 180-second process timeout;
previous dense correction plus response took about 87 seconds. Matrix-free
correction is not assumed faster: each PCG iteration costs an HVP, and a
26-control conditioned problem can need nearly as many iterations as basis
columns. Record measured cost instead of claiming acceleration in advance.
Larger FV grids/observation ensembles require a separate costed plan.

Results below use the fixed-source run. Historical artifacts were preserved.

## Implementation

`advar.local_refinement.refine_stationary` returns the refined control and
outer/inner iteration, HVP, branch and accepted/rejected-step diagnostics.
Defaults: 8 Newton iterations, 16 halvings, 4*N PCG iteration limit. The
normalized test compares ||F_trial||/||F|| with
sqrt(1 + 2*1e-4*alpha*(F/||F||)^T(Hs/||F||)); it avoids subtracting nearly equal
objective values or forming large squared norms. A stationary initial point
returns without a Hessian test and does not certify a minimum.

Objective and branch callbacks must be deterministic and must not mutate
tensor arguments or shared state. Numerical workspace copies preserve caller
inputs under that contract; this is not a callback sandbox. Branch ValueError
rejects a trial; unexpected errors propagate. PCG/outer-budget or line-search
failure raises rather than returning a partial stationary result. The existing
`prepare_response` wrapper then preserves the original control and returns
`ineligible` with `response=None`.

`fv_gn_response_probe.py --matrix-free-refine` selects this callback explicitly.
The default dense oracle and `--fresh-gn` behavior remain available. The new
measurement uses the archived actual GN control; its purpose is to validate the
changed refiner. PR172 already supplies the current-GN forwarding evidence.


## Measured 26-control FV result

`matrix_free_refined_response.json` is the new raw producer report;
`matrix_free_refined_response.run.json` and `.run.log` record the bounded child
process. CPU FP64, existing 4×5 full-support minmod case, unchanged fixed score.
Sources were unchanged between start and end, and were checked again afterward.

| Quantity | Result |
|---|---:|
| Starting GN max gradient | 1.3348760351817581e-4 |
| Refined max gradient | 6.962041511869577e-12 |
| Newton iterations | 2 |
| PCG iterations per Newton step | 25, 25 |
| Actual linear relative residuals | 7.0131645965e-11, 8.4415039536e-11 |
| Accepted step scales | 1, 1 |
| Refinement HVP calls | 54 |
| Distance from archived dense-refined control | 3.5361449570e-18 |
| Whole sensitivity relative difference | 0 |
| All three direction relative differences | 0 |
| Actual transpose adjoint relative residual | 9.8881638048e-11 |

The first Newton point has max gradient 1.1007383329e-10 and correctly requires
one further correction. The 54 refinement HVPs comprise 25 PCG iterations,
one PCG residual check, and one refiner residual check per Newton iteration.
They exclude the subsequent response HVP/VJP work. Three refiner branch records
(initial plus two candidates), and the workflow before/after records preserve
all 54 RK stage selectors and face signs. This is candidate-point evidence,
not a continuous-path certificate.

The final objective is 0.021720900386241804. The refined control differs from
the archived dense result by roundoff; the reported full response and three
direction responses match exactly in this environment. Every predeclared gate
passes. This bridges a changed numerical refiner to the existing validated
response; it is not a new observation-direction or forecast-skill experiment.

| Cost interval | Seconds |
|---|---:|
| Preparation | 0.1965 |
| Before/after assessment | 0.9889 |
| Matrix-free refinement | 46.4578 |
| Whole response | 22.4458 |
| Producer including comparisons and source checks | 70.1127 |
| Child process wrapper | 71.1324 |

Child maximum resident set size is 354,369,536 bytes (macOS `ru_maxrss`, about
354 MB / 338 MiB), measured for the entire producer rather than individual
stages. No timeout occurred. No fresh GN or perturbed nonlinear reanalysis was
run. The earlier dense refinement took 61.579 seconds in PR172; these are
separate executions, not a controlled timing benchmark or a scaling law.

## Verification and scope

- 99 unique affected tests passed; 18 existing `torch.jit.script` deprecation
  warnings. See `pr172_final_tests.log`. No full CPU/package CI was run.
- New module pinned basedpyright: 0 errors, 0 warnings, 0 notes;
  `pr172_typecheck.log`.
- Refiner tests include three predetermined starts of a smooth quartic SPD
  objective, a 64-control analytic stationary solution, exponential overshoot
  backtracking, exact tolerance boundary, invalid successful-PCG output,
  negative curvature, branch refusal, exhausted outer budget and nonfinite
  input/objective. These are analytic unit cases, not larger FV experiments.
- Actual refiner + response wrapper tests verify analytic sensitivity on
  success, and preservation of the original input/no sensitivity on refusal.
- Producer forwarding checks cover both selected refinement methods, and the
  publisher rejects changed residual evidence using the archived report hash.

The new refiner has no dense 32-control cap. That removes one implementation
constraint; it does not establish convergence, memory cost or speed on larger
FV grids. The existing public donor-cell gate is unchanged. General minmod
FSOI, different observation/support cases, preconditioning, typed prior
learning, finite impacts and whole-chain D7 remain separate milestones.

## Presentation and records

The original HTML demo now has a separate matrix-free-refinement panel.
Aside desktop DOM and screenshot (`matrix_free_refinement_desktop.png`) confirm
measured values and the distinction between the 26-control FV execution and
64-control analytic test. No new mobile check was performed. Original animation
frame-data SHA256 values are unchanged:

- template: `88bb5eb25dfb62cd519d98912d421407678f9957dc7b3a7503444d78a113c8e1`
- local index: `5fe7d42540b66e83ca40eb4943a070696402d332f91e4ecea54270ebfb28c31e`

Code-only Graphify refresh: 271 files, 6,718 nodes, 65,132 edges, 210
communities, no semantic LLM extraction. The existing KG/checklist records
preserve PR172 completion and keep larger FV scaling as the next open scope.

Final GREEN/RED review: no blockers. GREEN checked current/archived source and
input identities, measured gates and references. RED checked strict stationarity,
normalized Armijo, actual linear residuals, callback contract and refusal scope.
All recorded source hashes still match the reviewed implementation.
