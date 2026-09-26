# PR177–195 generalization review resolution checklist

Source: user review of PR #177–#195, received 2026-09-25. This ledger
separates input support, forward execution, stationary response, independent
reanalysis, and physical skill. Prior evidence remains in
`PR175_176_REVIEW_RESOLUTION.md` and `POST_MERGE_REVIEW_CHECKLIST.md`.

| ID | Review item | Closure criterion | Current state |
|---|---|---|---|
| R1 | Preserve completed scope | Verify PR #177–#195 main integration, regular 3-hour forward result, fixed-control point likelihood and derivative tests, and process-isolated local responses without adding them as one success rate | Done at their reported scopes; no new FV run implied |
| R2 | A full-valid correlated point profile needs a stationary response | In one declared 4×5 synthetic point problem, verify a qualified stationary point, true adjoint residual and full parameter gradient, then signed nonlinear reanalysis at two local sizes | **Closed for the constructed fixed centered-prior profile only.** One 26-control/13-parameter root, full VJP and four signed endpoints passed; see `FV_POINT_CENTERED_PRIOR_RESPONSE_RESULTS.md`. This is a different statistical problem from R2-O |
| R2-O-D | Original zero-centered-prior correlated point input needs an explicit support decision | Preserve bounded attempts and input identity; separate child exit, numerical refusal and response non-issuance without claiming root absence | **Closed as current-policy refusal for this exact input.** The last two source-bound sector attempts each refused all 16 seventh-step candidates with maximum gradients above `1e-10`. See `FV_POINT_ORIGINAL_POLICY_RESULTS.md` |
| R2-O-R | Original correlated point input's qualified response | Find a fresh stationary root under a separately justified nominal method, then exact adjoint/VJP and signed reanalysis | Open research target; the constructed centered-prior success is a different statistical problem |
| R3-M | One missing point needs a stationary response | Under complete model state/boundaries, bind status 1 and a selected correlation submatrix to a qualified point root, full VJP, inactive-slot check and signed endpoints | **Closed for the constructed fixed-centered-prior profile with one middle-time point missing.** Full 13-vector and two signed pairs passed; see `FV_POINT_MISSING_CENTERED_RESPONSE_RESULTS.md` |
| R3-Q | QC-excluded point profile needs a stationary response | Keep external QC status distinct from missing and clear sky; qualify a fixed-mask root, full VJP and signed endpoints | **Closed for the constructed status-2 external-QC profile.** Full VJP, zero inactive slot and two signed pairs passed; numerics match R3-M under the same active mask, while identity differs. See `FV_POINT_QC_CENTERED_RESPONSE_RESULTS.md` |
| R3-E | One wholly empty observation time needs a stationary response | Preserve time integration and external background while qualifying root, VJP and signed endpoints | **Closed for the constructed exactly-one-empty-first-time profile.** Four inactive first-time slots, nonzero theta path, full VJP and two signed pairs passed; see `FV_POINT_EMPTY_TIME_CENTERED_RESPONSE_RESULTS.md` |
| R4-D | Two-hole collocated partial refusal needs a support decision | Preserve three original attempts, separate process exit/resource/numerical status, and choose current refusal or a newly justified sector-aware search without relaxing final response gates | **Closed as explicit current-policy refusal for this exact input.** Archived attempt-3 first-failure counts 3 signature / 13 face-margin are source/output-hash bound; no sensitivity issued. See `FV_PARTIAL_POLICY_DECISION_RESULTS.md` |
| R4-R | Qualified response for the original two-hole collocated case | Only a separately declared branch-aware numerical method and fresh final stationary branch/adjoint/signed endpoints could establish it | **Open.** The first current-source cross-sector search refused after eight steps at gradient maximum `0.00682447`; a read-only audit localized its shrinking face margin to (q_y[2,0]). A separately selected alternate-sector seed passed fresh branch and exact-Hessian SPD gates but has gradient maximum `0.00684812` and has not been refined. No sensitivity issued; see `FV_PARTIAL_SECTOR_ROOT_ATTEMPT1_RESULTS.md`, `FV_PARTIAL_FACE_GEOMETRY_RESULTS.md`, `FV_PARTIAL_ALTERNATE_SEED_GATE_RESULTS.md`. Root absence is not proved |
| R5-F | Point observations and 3-hour leads need a combined forward path | Preserve the one-lead reference and use a regular 18-lead terminal score with matching point observation, boundary/time and resource contracts | **Closed for one fixed same-operator 4×5 forward case.** Point observations and 18-lead terminal FV field pass; strict branch refuses, so no long response. See `FV_POINT_3H_FORWARD_RESULTS.md` |
| R5-R-D | Fixed 3-hour point input needs a response-support decision | Keep its forward success separate from strict branch eligibility and response publication | **Closed for the exact PR #204 input/control as current-gate refusal.** The strict branch rejected before an admitted stage; no sensitivity was issued. See `FV_POINT_3H_RESPONSE_POLICY_RESULTS.md` |
| R5-R-R | Qualified point-observation 3-hour stationary response | Find a strict long-horizon branch and stationary analysis, then full adjoint/VJP and signed endpoints without changing the time/score contract | Open research target; R5-F proves forward execution only |
| R6 | General observation products remain unsupported | Bind product-defined footprint/coordinate, censoring/QC provenance and covariance semantics separately; do not infer physical measurements from prepared synthetic masks | Open; needs a declared product contract |
| R7-L | Fixed-case response job lifecycle | Keep forward AD in separate workers; verify two-job cancellation, resource accounting, worker/diagnostic completion and atomic result publication | **Closed for two fixed fv4x5 research jobs.** One qualified response published after child exit; the other was cancelled and never published. See `FV_RESPONSE_JOB_LIFECYCLE_RESULTS.md` |
| R7-P | Production concurrency and recovery | Add general request/input contracts, durable queue/recovery, full concurrent GN/refinement and supported in-process policy only if verified | Open; the fixed two-job run is not an operational service |
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
- [x] Source/input/resource-bound read-only classification of the last
  two sector attempts: this exact input is unsupported by the
  **current declared nominal-search policies** and issues no response.
  The constructed centered-prior input has a different identity;
  root absence is not proved (`FV_POINT_ORIGINAL_POLICY_RESULTS.md`).
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
- [x] Classify this exact long-horizon input's response support
  separately: forward execution passed, but the strict minmod branch
  refused before admitting a stage, so no local response is issued
  (`FV_POINT_3H_RESPONSE_POLICY_RESULTS.md`).
- [ ] Establish a strict long-horizon branch and eligible stationary
  response separately. This run's strict branch refused before an
  admitted stage, and no adjoint or nonlinear reanalysis was run.

## R4 original two-hole collocated partial input

- [x] Preserve the three earlier bounded attempts; the third raw report
  and six output hashes show child exit 1 without resource termination,
  16 branch-policy refusals before any finite Armijo trial, and no
  nominal root or response. The 3 signature and 13 face-margin counts
  are **first-failing** checks and may overlap.
- [x] Declare this exact input unsupported by the current fixed
  GN-signature/final-margin nominal correction policy and return no
  local sensitivity. Record process exit, resource completion,
  numerical eligibility and response issuance separately
  (`FV_PARTIAL_POLICY_DECISION_RESULTS.md`).
- [x] Run one separately planned current-source branch-aware root-only
  search with the same fixed input tensors. Product GN, exact seed SPD
  audit, strict 54-stage positive-margin trial checks, two measured-merit
  signature switches and six same-signature Armijo steps completed under
  the sampled 600-second/1-GiB guard. Eight Newton iterations exhausted
  at maximum gradient `0.0068244687`, with final accepted face margin
  `5.5370e-7`; status is a completed numerical refusal and no response
  (`FV_PARTIAL_SECTOR_ROOT_ATTEMPT1_RESULTS.md`). This is a new source
  experiment, not a replay of the historical source/problem digest.
- [ ] A cross-sector nominal search and valid response for this input
  still require a further separately justified numerical method and
  final-point proof; neither the archived nor new bounded refusal
  proves that the inverse problem has no root.
- [x] Classify a known strict-branch refusal at the GN seed as
  `seed_branch_refused`, distinct from curvature/refinement refusal and
  callback execution error. The runner requires the exact failed seed
  trace and forbids curvature/root/response evidence in that status;
  expected/unknown/malformed cases and forged parent reports are
  covered by small regressions. This is a status-contract repair, not
  a new guarded FV GN/root experiment; focused preflight tests still
  exercise FV forward/branch checks. Attempt 1's GN seed passed and
  its refusal is unchanged
  (`FV_PARTIAL_SEED_BRANCH_STATUS_RESULTS.md`).
- [x] Diagnose the PR #209 accepted-path face-margin collapse without
  rerunning GN/Newton. The production face-flux map matches all 53
  archived 54-stage margin ratios exactly; one unique (q_y[2,0])
  face minimizes 52 points, and its accepted absolute flux falls by
  about 548 times after iteration 2 while the maximum face flux stays
  nearly constant. This localizes an approached upwind-switch surface,
  not a root or causal proof (`FV_PARTIAL_FACE_GEOMETRY_RESULTS.md`).
- [x] Freeze one post-hoc alternate-sector seed from the 29 archived
  eligible changed-signature candidates: iteration 5/backtrack 3,
  previously merit-refused. Fresh fixed-input J/gradient/54-stage
  branch matches the archive, both margins exceed `1e-4`, and its
  own exact 26-column Hessian passes the local SPD gate. No product
  GN/Newton/adjoint/reanalysis ran; the gradient maximum remains
  `0.00684812`, so this is **seed eligibility only**
  (`FV_PARTIAL_ALTERNATE_SEED_GATE_RESULTS.md`).

## R7 process-isolated response lifecycle

- [x] Add default-inactive process cancellation to the sampled wall/RSS
  guard; verify child-group termination and reap without changing
  existing callers.
- [x] Run two fixed archived 4×5 responses in distinct processes and
  directories. Wait for a complete `running` child record before
  cancelling one; require the other to finish all branch, PCG/HVP,
  source/input and resource gates before atomic publication
  (`FV_RESPONSE_JOB_LIFECYCLE_RESULTS.md`).
- [ ] Production request types, durable restart/recovery, concurrent
  GN/refinement, generic FV inputs and whole-system memory limits remain
  separate from this research lifecycle evidence.
