# PR177–195 generalization review resolution checklist

Source: user review of PR #177–#195, received 2026-09-25. This ledger
separates input support, forward execution, stationary response, independent
reanalysis, and physical skill. Prior evidence remains in
`PR175_176_REVIEW_RESOLUTION.md` and `POST_MERGE_REVIEW_CHECKLIST.md`.

| ID | Review item | Closure criterion | Current state |
|---|---|---|---|
| R1 | Preserve completed scope | Verify PR #177–#195 main integration, regular 3-hour forward result, fixed-control point likelihood and derivative tests, and process-isolated local responses without adding them as one success rate | Done at their reported scopes; no new FV run implied |
| R2 | A full-valid correlated point profile needs a stationary response | In one declared 4×5 synthetic point problem, verify a qualified stationary point, true adjoint residual and full parameter gradient, then signed nonlinear reanalysis at two local sizes | **Closed for the constructed fixed centered-prior profile only.** One 26-control/13-parameter root, full VJP and four signed endpoints passed; see `FV_POINT_CENTERED_PRIOR_RESPONSE_RESULTS.md`. This is a different statistical problem from R2-O |
| R2-O | Original zero-centered-prior correlated point input still refuses | Obtain a qualified root and signed response on that original input, or retain an explicit unsupported-status contract without claiming root absence | Stage A and seven bounded numerical attempts are recorded below; the latest merit-only attempt refused all 16 iteration-7 candidates. **Open** |
| R3-M | One missing point needs a stationary response | Under complete model state/boundaries, bind status 1 and a selected correlation submatrix to a qualified point root, full VJP, inactive-slot check and signed endpoints | **Closed for the constructed fixed-centered-prior profile with one middle-time point missing.** Full 13-vector and two signed pairs passed; see `FV_POINT_MISSING_CENTERED_RESPONSE_RESULTS.md` |
| R3-Q | QC-excluded point profile needs a stationary response | Keep external QC status distinct from missing and clear sky; qualify a fixed-mask root, full VJP and signed endpoints | **Closed for the constructed status-2 external-QC profile.** Full VJP, zero inactive slot and two signed pairs passed; numerics match R3-M under the same active mask, while identity differs. See `FV_POINT_QC_CENTERED_RESPONSE_RESULTS.md` |
| R3-E | One wholly empty observation time needs a stationary response | Preserve time integration and external background while qualifying root, VJP and signed endpoints | **Closed for the constructed exactly-one-empty-first-time profile.** Four inactive first-time slots, nonzero theta path, full VJP and two signed pairs passed; see `FV_POINT_EMPTY_TIME_CENTERED_RESPONSE_RESULTS.md` |
| R4 | Two-hole collocated partial problem refuses final Newton correction | Keep the three original refusals; choose and predeclare a branch-aware nominal search policy separately from final-point local eligibility, then validate any new root and signed response | Open; no claim that a stationary point is absent |
| R5-F | Point observations and 3-hour leads need a combined forward path | Preserve the one-lead reference and use a regular 18-lead terminal score with matching point observation, boundary/time and resource contracts | **Closed for one fixed same-operator 4×5 forward case.** Point observations and 18-lead terminal FV field pass; strict branch refuses, so no long response. See `FV_POINT_3H_FORWARD_RESULTS.md` |
| R5-R | Point-observation 3-hour stationary response | Obtain a qualified long-horizon branch and stationary analysis, then full adjoint/VJP and signed endpoints without changing the time/score contract | Open; R5-F proves forward execution only |
| R6 | General observation products remain unsupported | Bind product-defined footprint/coordinate, censoring/QC provenance and covariance semantics separately; do not infer physical measurements from prepared synthetic masks | Open; needs a declared product contract |
| R7 | Service-level concurrency and result publication | Preserve process-level forward-AD isolation; verify cancellation, resource accounting, in-flight callback completion and publication ordering in a bounded service path | Open; current process result is two fixed local responses |
| R8 | Independent physical performance, finite influence and learning | Use independent verification events and a predeclared finite-amplitude range; keep learned error/prior normalization and data splitting separate from local synthetic derivative checks | Open; current truth is same-operator synthetic |

Do not relabel skipped CI jobs or focused tests as a full CPU/package
regression. Preserve the user's working-tree `AGENTS.md` change and all
historical numerical reports; new attempts must have distinct records.

## R2-O original-input sequential evidence

- [x] Source- and resource-bound correlated full-valid point preflight:
  26 controls / 13 parameters / 54 minmod stages; warm gradient maximum
  15.15045. See `FV_POINT_NOMINAL_ATTEMPT1_RESULTS.md`.
- [x] One direct warm exact-Newton feasibility run: first PCG refused
  non-SPD curvature after seven HVP calls; no candidate step.
- [x] One fixed-response-margin L-BFGS exploration: 13 accepted steps,
  objective 0.07993508→0.00644675, then all 16 next candidates refused
  by the combined stage/slope/face gate. See
  `FV_POINT_BASIN_ATTEMPT1_RESULTS.md`.
- [x] One separately planned mathematical-strictness search: 100 accepted
  steps, objective→0.001092566, 22 pointwise branch changes, gradient
  maximum 0.0108053; `step_budget`, not a stationarity handoff. See
  `FV_POINT_BASIN_ATTEMPT2_RESULTS.md`.
- [x] Locked last search endpoint: exact seed Hessian locally SPD, four PCG
  solves converged, but fixed-branch Newton iteration 4 refused all 16
  candidate steps before objective/gradient evaluation. See
  `FV_POINT_TERMINAL_NEWTON_ATTEMPT1_RESULTS.md`.
- [x] Separately declared core-strict, same-signature root-only diagnostic:
  five PCG solves converged and four corrections were accepted, but all 16
  iteration-5 candidates changed the full signature. Their diagnostic
  objectives and maximum gradients decreased without becoming accepted
  Newton iterates; no root or response was published. See
  `FV_POINT_CORE_STRICT_ROOT_ATTEMPT1_RESULTS.md`.
- [x] One predeclared sector-adaptive root diagnostic: two full-signature
  switches and four same-signature corrections were accepted, but all 16
  iteration-7 candidates failed the three actual-decrease conditions.
  No root or response was published. See
  `FV_POINT_SECTOR_ROOT_ATTEMPT1_RESULTS.md`.
- [x] One separately declared gradient-merit-only sector diagnostic from
  the preceding sixth accepted control: one signature switch and five
  same-signature corrections were accepted, then all 16 iteration-7
  candidates raised measured gradient merit. No root or response was
  published. See `FV_POINT_MERIT_ROOT_ATTEMPT1_RESULTS.md`.
- [ ] Find a qualifying stationary point for the original zero-centered
  input under a separately declared
  branch-aware or mathematically strict root-search policy. Keep final
  gradient `<1e-10`, exact SPD and robust branch/margin gates separate.
- [ ] Only then compute its fresh full 13-component adjoint/VJP response and
  two adjacent signed nonlinear reanalysis pairs; keep physical skill and
  finite-impact validation separate.

## R2 constructed centered-prior profile

- [x] Declare a fixed nonzero dynamics prior mean and generate independent,
  detached synthetic point observations and verification at its exact
  stationary control. Preserve the original point FV dynamics, symmetric
  correlation whitening and full support.
- [x] Verify fresh zero gradient, 26-column SPD Hessian, 54-stage strict
  branch and `>1e-4` slope/face margins; compute the full 13-component
  matrix-free adjoint/VJP and its true residual.
- [x] Reanalyze `p±h d` at `h=0.001` and `0.0005` dBZ on the four middle-time
  points, retain the final branch and stationarity, and pass both central
  comparisons with decreasing error. The parent independently recomputed
  derivatives, endpoints and signed scores. This closes R2 only for this
  constructed profile; see `FV_POINT_CENTERED_PRIOR_RESPONSE_RESULTS.md`.

## R3 incomplete point-observation profiles

- [x] One middle-time status-1 missing point at parameter index 5, with
  canonical inactive fill and a 3×3 principal correlation submatrix,
  kept the complete model state/boundaries and a qualified centered-prior
  stationary point. Its direct/indirect/total missing-slot gradients and
  nominal objective/score/control-gradient dependence are zero.
- [x] Three active middle-time points' common-bias direction passed full
  13-component VJP, true adjoint/tangent residuals, fixed branch and
  `h=0.001,0.0005` signed reanalysis. Parent-side derivative and
  endpoint recomputation passed (`FV_POINT_MISSING_CENTERED_RESPONSE_RESULTS.md`).
- [x] Repeat the qualified response contract with fixed external status-2
  QC exclusion at the same middle-time point. Its identity differs from
  status-1 missing; because the active mask and values are the same,
  numerical response/endpoints match. The QC decision itself was fixed,
  not differentiated (`FV_POINT_QC_CENTERED_RESPONSE_RESULTS.md`).
- [x] Qualify exactly one entirely empty **first** observation time in
  the constructed centered-prior case: four status-1 inactive parameters,
  no first-time whitener, full 54-stage model timeline, nonzero external
  theta dependency, full VJP and two signed pairs
  (`FV_POINT_EMPTY_TIME_CENTERED_RESPONSE_RESULTS.md`).
- [ ] More than one empty time and the original zero-centered R2-O input
  remain outside this result; do not infer finite removal impact.

## R5 point observations and long forward horizon

- [x] Extend only the point-problem lead-dependent boundary length,
  terminal forecast call, forecast time and total stage layout; preserve
  the original one-lead input identity.
- [x] Execute a bounded 0/10/20-minute four-point observation case with
  18 future ten-minute leads. The terminal same-operator synthetic field
  and point analysis values match at FP64 tolerance; 360 analysis and
  3,600 replayed terminal-call SSPRK stages were observed
  (`FV_POINT_3H_FORWARD_RESULTS.md`).
- [ ] Establish a strict long-horizon branch and eligible stationary
  response separately. This run's strict branch refused before an
  admitted stage, and no adjoint or nonlinear reanalysis was run.
