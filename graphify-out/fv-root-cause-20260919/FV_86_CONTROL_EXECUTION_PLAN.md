# Proposed 86-control FV milestone — not executed

This is a separate costed scope after the PR173 refiner implementation.
The candidate-domain correction does not require repeating the old FV run.

## Fixed experiment contract

- Same 40 by 50 spatial units: 8 by 10 cells, spacing (5,5). Vertex coordinates
  Y=i/2, X=j/2 retain the old streamfunction coordinate system. Use the same five
  functions [Y,X,XY,(X²-Y²)/2,X²Y] and limits [.11,.08,.07,.04,.03]. Do not use
  raw fine-grid indices, which would change velocity amplitudes and derivatives.
- Preserve the declared synthetic echo polynomial (not dBZ), sampled at the
  same lower-left normalized cell coordinates as the old fixture, now in half
  increments. This retains the fixture's sampling convention; it is not a new
  exact cell-average or continuum-convergence claim. Boundary arrays remain
  the prescribed adjacent-edge echo values with full known support.
- Keep observations at 0,60,120 seconds and forecast at 180 seconds, truth
  coefficient multipliers [.7,-.6,.5,.4,-.3], growth .008, min dBZ -10, std .1
  dBZ, full detected masks, field-smoothness weight zero, and existing defaults.
- Use 18 substeps per interval: halved spacing approximately doubles the
  outgoing rate, so doubled time resolution maintains the declared CFL box.
  Validate the actual conservative coefficient-box CFL using
  `transport.bounded_fv_coefficients`, not this scaling argument alone. Two RK
  stages per substep give 108 stages across analysis plus forecast. Provide
  36 boundary stage-pairs for analysis and 18 for forecast.
- 80 initial-field + 5 flow + 1 growth controls = 86. Parameters are 240 dBZ
  observations plus one fixed-pattern mean-background coefficient = 241.
- Preserve the product objective exactly: sum of pseudo-Huber observation
  costs + half squared control-prior norm. Do not replace sum with mean or
  rescale J to pass the gradient gate. There are four times as many observations
  and field controls but still six dynamics controls: this changes statistical
  weighting relative to the dynamics prior. The experiment tests larger
  discrete inverse execution, not grid-independent inverse convergence.
- Define verification before GN from a prescribed synthetic truth forecast
  with a fixed spatial pattern. Keep the same verification for both starts.
  It is a conditional synthetic score, not independent meteorological truth.
- Two predetermined starts: zero field controls with dynamics
  [.1,-.08,.06,.04,-.03,.008], and the same zero field with half those dynamics.
  No successful result will be selected by changing seeds after measurement.
- Evaluate the GN endpoint's strict branch contract. An invalid endpoint is a
  recorded refusal. For a valid endpoint, hold its cellwise selectors and face
  signs fixed during refinement and response; compare signatures for every
  RK stage. Do not require the two initializations to choose the same branch.

The current research adapter hard-codes 4x5/26/61/54. A separate explicitly
configured scaled adapter is needed; merely removing shape checks is invalid.
The current response PCG cap is 104, whereas refinement defaults to 4*N=344.
Keep the response cap in the first proposal and record refusal if insufficient;
any larger-cap/preconditioner study must be declared separately before a run.
Final gradient <1e-10 and actual linear/adjoint residual <=1e-10 remain fixed.

## Cost estimate and proposed execution budget

Archived 26-control refinement: 46.46s for 54 HVPs. Response: 22.45s for 26 HVPs.
An 8x10 mesh has four times the cells and twice the substeps. A rough 8x work
factor gives ~9.2 minutes for refinement+response at unchanged HVP counts.
Multiplying HVP counts by 86/26 gives ~30.4 minutes. These are planning models,
not measured bounds: tiny-grid framework overhead, conditioning, rejected
trials and replay can change the result substantially. GN/preparation cost is
additional. Multiplying the old whole-process RSS by eight would also be an
unreliable memory prediction because much of that process is fixed overhead.

Proposed first gate: at most 120 seconds, 2 GiB RSS, build/validate the scaled
fixture and measure a small fixed number of gradient/HVP/branch evaluations.
No solve in this cost gate. Use measured timings/RSS to report a revised range.
Proposed subsequent budget: at most 30 minutes and 2 GiB RSS per declared start,
maximum 60 minutes for two starts. Terminate and preserve partial records at a
limit; do not relax gates or turn timeout into numerical failure/success.

Required records: input/geometry/schedule/source identities, physical CFL
bound and actual CFL, GN reason, J and max gradient before/after correction,
Newton/PCG/HVP counts and actual residuals, branch signatures, whole sensitivity
when eligible, phase times, sampled peak RSS and final termination reason.

The previous user policy requires approval before larger experiments. None of
this scaled FV numerical work has been started. Approval covers these explicit
resource caps; no unbounded grid sweep is proposed.

Sources: `fv_minmod_inverse_probe.py::make_spatial_case`,
`fv_minmod_matrix_free_probe.py::make_research_functions`,
`variational.py::_robust_objective_from_residual`,
`transport.py::bounded_fv_coefficients`, `local_response.py` PCG cap.
