# PR161 follow-up: observation meaning and transform branches

Baseline: merged PR161, main 2de675c (same content as c6e606d).
This follow-up keeps the existing FV equation, donor-cell discretization,
normal-equation modes, stationarity tolerance, and fixed-support research scope.

## Mathematical contract

The likelihood treats censored samples as events below the detection limit,
not measured continuous values. The parameterized background now receives
raw detected values, the fixed detection limit for censored samples, and
min_dbz for invalid samples. The three observation masks remain unchanged.
Consequently the raw censored and invalid numbers have zero derivative through
B(y, theta); detected samples and theta retain their differentiable paths.
Callbacks that capture fixed masks may still distinguish all three classes.
The caller must construct its baseline background from these same canonical
inputs; a raw-censored-value baseline is rejected rather than silently changed.

For the initial transform let o=(B-min_dbz)/echo_transform_scale_dbz and
 e=transform_epsilon. Its clamp is smooth on o<e and o>e, but not at o=e.
Both B-dependent response modes (first_observation and parameterized) now reject
supported cells within a roundoff-scaled distance of that join:

    |o-e| <= 64 eps_machine ((|B|+|min_dbz|)/scale + e).

The lower open branch has zero initial-field derivative with respect to B;
it does not preserve B as the zero-control initial field. The upper branch
retains the original smooth softplus transform. Frozen B requires no such
B-derivative gate. No clamp or forward equation was changed.

This is a local branch check, not a certificate for finite perturbations or
arbitrary callback code. The builder must itself be locally twice differentiable.
A future finite-impact experiment must check the whole declared perturbation
path and mask transitions separately; the response API still claims neither.

## Next numerical checkpoints

- [ ] Compare donor-cell and the existing low-diffusion candidate at the same
  48 km, 180 minute, 32/64/128 grid conditions, holding truth and boundaries fixed.
  Report echo L2, centroid, variance/width, maximum, threshold area, JVP error,
  positivity, mass budget, wall time and peak memory. Keep independent continuous
  reference derivatives separate from discrete AD/FD consistency.
- [ ] Before enabling exact responses for a limiter, verify active branches,
  JVP/VJP and mixed second derivatives, then polished reanalysis differences.
  Do not silently lift the donor-cell-only gate.
- [ ] Predeclare absolute/relative impact tolerances and a numerical signal floor
  before evaluating both signs and several directions. Small actual changes are
  not successes solely because their signs match.
- [ ] Complete typed mean/precision/support, general inputs and whole-chain D7
  separately. No new full CI or unchanged 240-grid experiment is needed here.

For constant translation without boundary effects, SSPRK2 donor-cell has shift
weights (1-C+C^2/2, C-C^2, C^2/2), mean C and variance C per step. After time T
its added variance is |u| T dx. Reducing dt at fixed grid does not remove this
spatial diffusion. At u=0.5 m/s, T=10800 s and dx=375 m this is 2.025e6 m^2.
This algebra explains why tighter PCG tolerances or more substeps are not the
next spatial-accuracy fix. Existing 180-minute measurements are retained;
no improved long-horizon accuracy is claimed in this change.

## Focused verification

- Response/transform tests: 10 passed, 26 deselected, 102.63 s on the designated
  Python 3.12 environment. Includes the four existing parameterized mean
  dense-oracle/reanalysis/gate tests, three boundary/nextafter rejection cases,
  two open-branch directional-difference cases and first-observation join rejection.
  The earlier five-test development run overlaps these tests and is not added.
- basedpyright 1.39.9 on the changed product module: 0 errors, 0 warnings.
- Full CPU CI and the unchanged 240-grid experiment were not rerun.
- Default first-observation zero-score exact response: 1 passed, 35 deselected,
  13.02 s; the added B-dependent guard preserves this normal interior path.
- RED final reread found no remaining blocker in the canonicalized input helper
  or shared initial-transform guard. GREEN owns the censored callback regression.
- Parameterized partial-observation targets: 2 passed, 6 deselected, 57.81 s
  (bounded wrapper 58.58 s, sampled peak RSS 340230144 bytes).
- Strengthened censored target: 1 passed, 7 deselected, 66.99 s. It explicitly
  reruns solve/refinement and forecast after replacing a 0 dBZ censored sample
  with 4 dBZ (limit 5 dBZ). Control, forecast, observation response and theta
  response are bitwise unchanged; censored y response is zero and detected/theta
  responses remain nonzero. Sampled peak RSS is recorded in the JSON wrapper.
- Total: 13 distinct focused cases across these runs. The strengthened target
  is already included in the two parameterized partial cases, not added again.
- GREEN implementation review and final RED reread completed. Graphify cached
  code-only update: 6502 nodes, 64776 edges; semantic LLM tokens 0.
