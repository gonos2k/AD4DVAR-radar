# RED preflight — fresh-point two-direction radius trial

## Scope and status

Static review of the fresh-point runner, receipt-only endpoint loader, solver,
radius policy, and stored neighboring-candidate arithmetic. The trial plan is
frozen at SHA-256 `a0fef5718772032b4567d24ca75fbaf39cff8b58e30f7e4b249dbabb4a5c7d26`;
its source bundle is `b332268a74f0c5ed9f9e05b3d890288d2e953215b1422253eec2426cb183cab3`.
I verified the plan sidecar, all 216 source pins, all 287 archive pins, all 216
source-bundle members, and the current trial policy against `load_plan()`. The
receipt-only endpoint loader returned d19e without running FV. No production
trial or guarded launch was run by RED.

## Evidence and contracts reviewed

- The admitted point is the closed `d19e882d31b634ef0be755e6419b57ecf3691734efc736d5f40ea18c3ac4145e`
  endpoint from the prior direction comparison, whose starting point was
  `b6470069c9b8992bf19fa85ab848f5ef424a1e2e8e33a4909bcb0d036ef88660`.
  `fv_point_3h_netflux_comparison_endpoint.py:19-32,64-109,117-140,162-195`
  validates the pinned plan/source/receipt bundles, one committed step, control
  lineage, fixed non-control inputs, runtime, and final full-G closure. The new
  runner loads this endpoint and freshly observes P2 at d19e (`fv_point_3h_netflux_subspace_trial.py:119-159`).
- The new point recomputes the native pair, checks the previous and minimum-mixture
  P2, checks native face release before Q0, and refuses unsupported pairs
  (`fv_point_3h_netflux_subspace_trial.py:137-159`). It rebuilds both direction
  models at this point and performs four fresh side HVPs; no b647 HVP is reused
  (`:176-231`). The composed direction is checked by the existing chart-tangent
  validator (`:199-200` and `:73-83`).
- The trial policy keeps one 600-second internal / 660-second outer / 2-GiB
  sampled-RSS budget, four HVPs, 24 GN row VJPs and one dense solve, one commit,
  and at most six candidates per arm (12 total). Initial control-space radius is
  `1.8757177916958825e-5`; rejected/nonfinite or poor-model trials halve it,
  otherwise it holds, and it never expands (`fv_point_3h_netflux_subspace_trial.py:33-51`,
  `fv_point_3h_subspace_policy.py:7-18`).
- The chart safeguard checks actual retracted displacement, contracts alpha by
  at most 32 bisections, and checks the backend deadline before and after the
  chart work (`fv_point_3h_netflux_subspace_trial.py:99-117`). Its alpha is passed
  consistently to candidate evaluation and the residual-vector prediction
  (`:254-267`). The declared cap is 33 chart calls per candidate, or 396 across
  all 12 candidate slots. These calls count against the shared wall budget; they
  do not consume the FV candidate cap.
- Candidate eligibility still uses the existing actual J/R gates, followed by an
  independent repeated P2/full-G closure. Both arms must be independently
  eligible before the repeated-R-then-J selector can commit one candidate
  (`fv_point_3h_netflux_subspace_trial.py:260-296`). Model fidelity only changes
  the next radius; it does not relax or replace J/R acceptance.
- The subspace model normalizes each D column and its paired V column by the same
  control norm, solves the two-variable positive-orthant trust-radius problem,
  falls back to the best axis on near same-ray dependence, and refuses near
  antiparallel cancellation (`fv_point_3h_subspace_model.py:31-53,75-190`). The
  radius-active multiplier receives implicit Newton correction; focused gradient
  checks now cover identity and skew D/V cases. A deterministic 40-case dense
  angle oracle checks the returned merit against the feasible two-dimensional
  cone (`tests/test_fv_point_3h_subspace_model.py:42-97`).
- Stored b647 evidence shows sizable finite vector-model errors in echo
  components and a selector change between neighboring rejected/accepted GN
  candidates, while flow and growth errors are smaller (`candidate_model_error_blocks.json`).
  That report is stored-vector arithmetic only and has no base selector trace;
  it cannot isolate branch changes as the cause. The d19e experiment must report
  actual candidate J/R and fresh repeat evidence separately from local-model
  fidelity, and must not infer a causal selector effect from these historic traces.

## RED findings

No source-level acceptance-gate bypass found in the reviewed path. The chart
contraction now avoids unconditional pre-FV rejection of every boundary step,
while keeping the physical displacement inside the current radius. The scalar
KKT branch and its differentiability remain piecewise in the active set; the
gradcheck evidence applies only away from active-set transitions. Near-dependent
positive directions deliberately reduce the feasible search to one axis at a
time, so that outcome must be labeled reduced-domain and must not be described as
the full two-direction model optimum.

## RED launch clearance

The frozen plan policy matches the reviewed runner: 600-second internal and
660-second outer deadline, 2-GiB sampled RSS, one global budget, four fresh HVPs,
24 row VJPs and one dense solve, six candidate slots per arm (12 total), two
independent repeats at most, one commit, a 33-chart-call cap per candidate, and
initial radius `1.8757177916958825e-5` with the declared shrink/hold/no-expansion
rule. The source archive has exactly 216 pinned members with no content mismatch;
the 287 archive pins all match. `trial.load_plan()` accepted the frozen policy and
pins, and the receipt-only endpoint loader verified the admitted d19e control.

The independent observation plan is SHA-256
`4d467799150c88af4ebdef79cda7034aedbbb2e9bbb99f8321c8e10f4809b667`; the
preflight record identifies the earlier plan as superseded with zero calls. This
score remains observation-only and is excluded from candidate selection.

No RED blocker remains for one guarded launch under this exact frozen plan and
source bundle. After launch, independently recheck source-before/source-after,
fixed input/runtime identity, candidate receipt hashes, actual J/R gates, and
independent repeat closure. Report the bounded finite result without claiming
convergence, a minimum, response improvement, or forecast skill.

## Verification evidence available now

- Focused solver tests: 11 passed, including two active-radius gradient checks and
  the deterministic 40-case dense-angle oracle.
- Pyright on the solver: 0 errors, 0 warnings, 0 informations.
- `py_compile` on solver and focused tests passed.
- The final parent preflight log reports 44 passed (including the latest 11 solver
  tests) and Pyright 0 errors/warnings across its final module set.
- Root Graphify incremental refresh completed at 11,629 nodes and 323,065 edges.
  Updated artifacts are `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`,
  `graphify-out/manifest.json`, and `graphify-out/.graphify_detect.json`.
- No production run or guarded launch was performed by RED.
