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
| R2-O-R | Original correlated point input's classical full26 smooth-root response | Qualify the original full smooth stationary API without borrowing a constructed prior | **Unsupported at the selected qx zero event; classical target remains open.** The unchanged statistical problem now has the separately scoped R2-O-A conditional response. Historical smooth/one-face refusals remain preserved. |
| R2-O-A | Conditional selected-qx-face response for the original zero-prior point input | Preserve all26 prior terms and13p; qualify25D tangent stationarity/curvature plus own one-sided normal signs, adjoint/VJP and actual signed reanalysis | **Closed at one approximate numerical branch, one direction and two sizes.** Positiveqy release2 has tangent gradient1.97e-12; all four actual-p endpoints pass fresh25D curvature/normal/branch gates. Relative FD errors4.91e-8/1.22e-8; no full26 classical, exact-root, uniform-ball, physical or3hour claim. See `R2_POINT_SINGLE_QX_RESPONSE_RESULTS_20261003.md` |
| R3-M | One missing point needs a stationary response | Under complete model state/boundaries, bind status 1 and a selected correlation submatrix to a qualified point root, full VJP, inactive-slot check and signed endpoints | **Closed for the constructed fixed-centered-prior profile with one middle-time point missing.** Full 13-vector and two signed pairs passed; see `FV_POINT_MISSING_CENTERED_RESPONSE_RESULTS.md` |
| R3-Q | QC-excluded point profile needs a stationary response | Keep external QC status distinct from missing and clear sky; qualify a fixed-mask root, full VJP and signed endpoints | **Closed for the constructed status-2 external-QC profile.** Full VJP, zero inactive slot and two signed pairs passed; numerics match R3-M under the same active mask, while identity differs. See `FV_POINT_QC_CENTERED_RESPONSE_RESULTS.md` |
| R3-E | One wholly empty observation time needs a stationary response | Preserve time integration and external background while qualifying root, VJP and signed endpoints | **Closed for the constructed exactly-one-empty-first-time profile.** Four inactive first-time slots, nonzero theta path, full VJP and two signed pairs passed; see `FV_POINT_EMPTY_TIME_CENTERED_RESPONSE_RESULTS.md` |
| R4-D | Two-hole collocated partial refusal needs a support decision | Preserve three original attempts, separate process exit/resource/numerical status, and choose current refusal or a newly justified sector-aware search without relaxing final response gates | **Closed as explicit current-policy refusal for this exact input.** Archived attempt-3 first-failure counts 3 signature / 13 face-margin are source/output-hash bound; no sensitivity issued. See `FV_PARTIAL_POLICY_DECISION_RESULTS.md` |
| R4-R | Original smooth 26-control response for the two-hole collocated case | Fresh full smooth stationary branch/curvature, adjoint and signed reanalysis under a justified method | **Unsupported at the selected zero-flux event; classical smooth-root target remains open.** The new R4-A contract computes a conditional active-face response for the unchanged statistical problem. It does not certify the old full-gradient gate or reopen the smooth26 API. Historical attempts remain preserved. |
| R4-A | Conditional selected-active-face response for the original two-hole input | Keep original full26 prior/data/score; qualify tangent stationarity, positive tangent curvature and one-sided normal signs, then adjoint/full VJP and signed reanalysis | **Closed at one approximate numerical branch, one direction and two sizes.** Four signed endpoints pass fresh tangent/curvature/normal/branch gates and score differences; no exact-root, uniform-neighborhood, global-minimum or physical claim. See `R4_ACTIVE_FACE_RESPONSE_RESULTS_20261003.md` |
| R5-F | Point observations and 3-hour leads need a combined forward path | Preserve the one-lead reference and use a regular 18-lead terminal score with matching point observation, boundary/time and resource contracts | **Closed for one fixed same-operator 4×5 forward case.** Point observations and 18-lead terminal FV field pass; strict branch refuses, so no long response. See `FV_POINT_3H_FORWARD_RESULTS.md` |
| R5-R-D | Fixed 3-hour point input needs a response-support decision | Keep its forward success separate from strict branch eligibility and response publication | **Closed for the exact PR #204 input/control as current-gate refusal.** The strict branch rejected before an admitted stage; no sensitivity was issued. See `FV_POINT_3H_RESPONSE_POLICY_RESULTS.md` |
| R5-R-R | Qualified point-observation 3-hour stationary response | Find a strict long-horizon branch and stationary analysis, then full adjoint/VJP and signed endpoints without changing the time/score contract | **Open.** Four fresh-curvature coupled original-J epochs lowered J to `0.083897`, but final full gradient maximum=`1.058817` remains above `1e-10`; epoch limit, no stationary candidate or response. No old Hessian is inherited at final `cd6b` point. See `FV_POINT_3H_COUPLED_ORIGINAL_J_CONTINUATION_RESULTS_20261003.md`. |
| R6 | General observation products remain unsupported | Bind product-defined footprint/coordinate, censoring/QC provenance and covariance semantics separately; do not infer physical measurements from prepared synthetic masks | **Open.** The [minimum external-input contract](FV_REAL_PRODUCT_AND_INDEPENDENT_VALIDATION_INPUTS.md) requires a selected product/version/decoder and trust-root-authorized native volume; calibrated dBZ with explicit echo/clear/censored meaning; signed grid/time identity; cell-level QC/source evidence; and preregistered error/covariance semantics. Typed repository contracts do not validate those product-specific facts. |
| R7-L | Fixed-case response job lifecycle | Keep forward AD in separate workers; verify two-job cancellation, resource accounting, worker/diagnostic completion and atomic result publication | **Closed for two fixed fv4x5 research jobs.** One qualified response published after child exit; the other was cancelled and never published. See `FV_RESPONSE_JOB_LIFECYCLE_RESULTS.md` |
| R7-P | Production concurrency and recovery | Add general request/input contracts, durable queue/recovery, full concurrent GN/refinement and supported in-process policy only if verified | **Open.** Same-host duplicate launch is excluded; a durable attempt manifest binds one fixed response to its raw/resource/publication records, and read-only triage classifies them. One actual fixed FV v2 run passed. This is not authenticated recovery or a general service; see `FV_RESPONSE_JOB_ATTEMPT_BINDING_RESULTS.md` |
| R8 | Independent physical performance, finite influence and learning | Use independent verification events and a predeclared finite-amplitude range; keep learned error/prior normalization and data splitting separate from local synthetic derivative checks | **Open.** The [minimum external-input contract](FV_REAL_PRODUCT_AND_INDEPENDENT_VALIDATION_INPUTS.md) requires trust-root-authorized future targets and native source closure, independent track/event and weather/range classification provenance, paired same-input parent/candidate outputs, fixed domain/metrics, preregistered finite-amplitude endpoints, and event-level train/validation/test separation. Current same-operator synthetic checks do not establish physical skill; study owners must preflight cohort size and choose amplitudes before seeing outcomes. |
| V1 | Full CPU and package regression remained unexecuted | Execute full CPU/package once; preserve failures and distinguish local affected repairs from whole-suite success | **Execution gap closed; full regression remains failed.** Run `36836402831` at `a9e321d` executed the disjoint 2,311-ID Linux inventory in all six terminal shards, with 87 pytest failed/error cases. Wheel/CLI and UI passed. Local research repairs are recorded separately; per user 2026-10-01 direction this is not deployment and unnecessary full CI/package repeats are avoided. No full-pass claim |

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
- [x] Locate the first strict-predicate refusal for that same input and
  control in one guarded run. Global callback stage 0 is the first
  analysis-replay SSPRK stage; `q_y[3,1]` has absolute flux
  `5.5511e-17` versus strict tolerance `4.5475e-15`, with no other
  violation in that stage. This is a branch-cause diagnostic, not a
  stationary-root or sensitivity result
  (`FV_POINT_3H_FIRST_BRANCH_RESULTS.md`).
- [x] Prove the face is a coefficient cancellation, then run one
  predeclared alternate flow control (`control[21]`, fraction
  `-0.60→-0.59`) while keeping observations, background parameters,
  target, boundary and time contract fixed. Its 49 static faces and
  all 3,600 pointwise and full-oracle stages pass under the unchanged
  strict rule. This qualifies a branch at that **control only**; it is
  not a stationary point, response or finite-interval validation
  (`FV_POINT_3H_SHIFTED_BRANCH_RESULTS.md`).
- [x] Measure the branch-supported seed's **analysis objective**
  gradient and one exact-HVP Newton RHS under a guarded budget. The
  complete 3,600-stage branch remains strict; `||g||_inf=2.25472968`
  fails stationarity, and PCG refuses its SPD premise after 15 HVPs.
  No Newton step, full Hessian spectrum, adjoint or response follows
  from this preflight (`FV_POINT_3H_SEED_LINEAR_RESULTS.md`).
- [x] Audit the same nonstationary seed with 26 exact gradient-JVP
  Hessian columns and one fresh minimum-eigenvector HVP. Symmetry and
  eigenpair residual pass; one eigenvalue is `-0.7460777`, outside
  the predeclared `3.4333e-5` uncertainty band. This explains the
  PCG curvature refusal **at this seed only**, without proving root
  absence or issuing a response (`FV_POINT_3H_SEED_HESSIAN_RESULTS.md`).
- [x] Run one predeclared `-Hg/||Hg||` joint J/gradient-merit
  exploration from that indefinite seed. Of five tried fixed step
  sizes, `alpha=0.00125` yielded a changed but strict 3,600-stage
  endpoint with J and merit decreases; gradient maximum fell to
  `1.26053496`. This is one candidate, not a smooth path, root or
  response (`FV_POINT_3H_MERIT_STEP_RESULTS.md`).
- [x] Continue under one 8-epoch/64-trial resource contract, recomputing
  each current branch/margin, J/g/Phi, Hg and H^Tg after a switch.
  Two changed-branch endpoints reduced J and Phi; the third epoch
  refused all eight fixed trial sizes. Final `||g||_inf=0.87277468`,
  no root or response (`FV_POINT_3H_MERIT_CONTINUATION_RESULTS.md`).
- [x] Refactor the continuation's fixed seed/alpha/budget settings into
  immutable default and tail policies, preserve the old default command,
  and run one pinned tail strictly below the previously refused minimum
  alpha. Both first-alpha endpoints were strict and lowered J/Phi, but
  final `||g||_inf=0.87197938` is not stationary or response-qualified
  (`FV_POINT_3H_MERIT_TAIL_RESULTS.md`).
- [x] At the latest tail endpoint, rebuild the full Hessian and the
  20+6 Schur system, including the nonzero field-gradient correction.
  One original-HVP Newton step with a block preconditioner decreased J;
  fresh curvature at its changed-branch endpoint then proved indefinite.
  Earlier SPD evidence is not inherited
  (`FV_POINT_3H_ACCEPTED_ENDPOINT_AUDIT_RESULTS_20261003.md`).
- [x] At fixed original dynamics and parameters, perform bounded field
  correction and a four-update continuation using the original full
  objective. The last accepted point has J `0.133702`, field gradient
  maximum `0.354372`, and dynamics/full maximum `3.916502`. Execution
  completed; the numerical result refused at the iteration limit and
  issued no corrected field or sensitivity
  (`FV_POINT_3H_FIELD_CONTINUATION_RESULTS_20261003.md`).
- [x] Preserve the separately bounded five-update precision continuation and
  its internal-budget refusal during solve 6. Terminal full-gradient
  diagnostics remain nonstationary; all five completed linear residuals
  pass and no field or response is issued
  (`FV_POINT_3H_FIELD_PRECISION_CONTINUATION_RESULTS_20261003.md`).
- [ ] Establish an eligible full long-horizon stationary point whose
  **final** strict branch passes, then compute adjoint/full VJP and
  signed nonlinear reanalysis. Field correction has been attempted;
  full stationarity, adjoint and reanalysis remain unqualified.

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
- [x] Run one direct root-only refinement from exactly that fixed
  exploratory seed, with no GN replay or alternate fallback. One
  signature switch returned to the former GN branch; seven later
  accepted steps stayed there. Eight 29-iteration PCGs converged,
  but 75 candidates (8 accepted/67 policy-refused) exhausted the
  eight-step budget at maximum gradient `0.0067083783` and last
  accepted face margin `2.3866e-9`. This is a completed numerical
  refusal with no final root/response, not root-absence evidence
  (`FV_PARTIAL_ALTERNATE_ROOT_ATTEMPT1_RESULTS.md`).
- [x] Diagnose only the first two archived accepted `q_y[2,0]` sign
  crossings with a guarded, source/input-bound finite-offset probe.
  At two offsets on each side the complete 54-stage signatures matched
  within each side; across sides only this face sign differed, and
  `g·H d` changed from negative to positive. All eight face margins
  were below `1e-4`, so this is neither an eligible response nor proof
  that the face caused the historical refusal. R4-R remains open
  (`FV_PARTIAL_FACE_EVENT_ATTEMPT1_RESULTS.md`).
- [x] Recompute eight finite-offset gradients and flow-only smooth
  face normals, then Euclidean tangent projections and same-event
  sampled gradient-segment minima. Both remain about `0.0096`, mainly
  initial-field controls, with small near/far changes. This motivates
  a further field/tangent correction study; it does not certify
  constrained/Clarke stationarity or close R4-R
  (`FV_PARTIAL_FACE_TANGENT_ATTEMPT1_RESULTS.md`).
- [x] Correct only the twenty initial-field controls at the frozen
  alternate seed's six dynamic controls. Three Newton corrections
  passed the original field-gradient gate at `7.03e-12`; unchanged
  face margin/full signature and a fresh parent final audit passed.
  Full gradient maximum `0.0061284` remains in dynamic controls, so
  this is a block-stationary handoff and R4-R stays open
  (`FV_PARTIAL_FIELD_CORRECTION_ATTEMPT1_RESULTS.md`).

## R7 process-isolated response lifecycle

- [x] Add default-inactive process cancellation to the sampled wall/RSS
  guard; verify child-group termination and reap without changing
  existing callers.
- [x] Run two fixed archived 4×5 responses in distinct processes and
  directories. Wait for a complete `running` child record before
  cancelling one; require the other to finish all branch, PCG/HVP,
  source/input and resource gates before atomic publication
  (`FV_RESPONSE_JOB_LIFECYCLE_RESULTS.md`).
- [x] Prevent two same-host coordinators from launching the **same**
  fixed job directory concurrently. A persistent sibling POSIX lock
  covers the empty-directory check through worker completion and
  publication; cross-process and symlink-parent-alias regressions
  prove one owner. Busy raises a distinct exception before a second
  worker starts. This is launch exclusion, not crash recovery
  (`FV_RESPONSE_JOB_SINGLEHOST_LOCK_RESULTS.md`).
- [x] Add read-only triage under that lock for fixed `fv4x5` crash
  artifacts: no observed artifacts, internally consistent but
  unauthenticated publication candidate, or needs reconciliation.
  Require a completed lifecycle, absolute worker output path,
  current source/archive/input and full worker/resource gates.
  Missing lifecycle or publication never triggers automatic publish,
  retry, PID signal or a recovered-success claim
  (`FV_RESPONSE_JOB_RECONCILIATION_TRIAGE_RESULTS.md`).
- [x] Bind one fixed job attempt before spawn to a durable UUID-bearing
  manifest with canonical job path, exact command, code/archive/input and
  budget identity. Bind the reaped child, stable raw/resource bytes,
  publication and terminal lifecycle to that attempt; require any v2
  marker to use v2-only read-only reconciliation. Fake-guard mutations,
  publication races and fsync failures are regression-tested. One actual
  `fv4x5` job completed and the inspector returned an unauthenticated
  `publication_candidate`, with no retry or recovered-success claim
  (`FV_RESPONSE_JOB_ATTEMPT_BINDING_RESULTS.md`).
- [ ] Production request types, durable restart/recovery, concurrent
  GN/refinement, generic FV inputs and whole-system memory limits remain
  separate from this research lifecycle evidence.

## R4 objective evaluation diagnosis (2026-10-03)

- [x] Same rejected +2eta0 trial: native and promoted-FP64 component deltas are positive (~1.27e-14), but independent 50/80-digit and definite 80-digit interval arithmetic give a negative captured-input equation delta (~-1.33e-19). Same-point objective-difference diagnosis closed; see `R4_OBJECTIVE_PRECISION_RESULTS_20261003.md`.
- [x] Audit-only Armijo comparison corrected; runtime mismatch now refuses before FV construction. Executed earlier source and its post-runtime audit are kept distinct.
- [ ] A precision-aware nominal search/acceptance policy and original full stationary root are still required for R4-R. No historical trial was accepted and no sensitivity was issued.

## R4 precision-aware correction (2026-10-03)

- [x] Implement and execute a separately declared interval actual-J comparison with the original objective, same128epsJ allowance and original gradient Armijo. One new correction passes tangent gradient8.34e-12 and fresh linear relative residual1.36e-11; all source/input/runtime checks pass. See `R4_INTERVAL_CORRECTION_RESULTS_20261003.md`. This closes the false scalar-J veto for this slice, without rewriting old refusals.
- [ ] Original full gradient remains0.00619, so R4-R is open and no adjoint/response is issued. Further normal-direction or justified nonsmooth-contract work is required.

## R4 selected active-face calculation (2026-10-03)

- [x] Preserve the original two-hole observations and all26 latent-prior terms in a structural zero-flux tangent chart; enforce the original pivot domain in every public calculation.
- [x] Obtain numerical tangent stationarity (7.95e-12), fresh restricted positive curvature and strict other-branch evidence.
- [x] Enclose correctly oriented one-sided normal derivatives at the approximate baseline and four changed-p endpoints; preserve independent evaluator programming failures separately.
- [x] Compute full61-component tangent-adjoint VJP and validate one middle-time direction by actual signed reanalyses at0.00025/0.000125dBZ; relative differences8.45e-9/2.11e-9. GREEN/RED terminal audits GO. See `R4_ACTIVE_FACE_RESPONSE_RESULTS_20261003.md`.
- [ ] Keep the original smooth26 API unsupported at this event. Formal exact/uniform-neighborhood certification and independent physical validation are not established by this conditional branch evidence.

## R2 original point-input active-face investigation (2026-10-03)

- [x] Reevaluate two exact archived controls under current code with original input/runtime; J/gradient/signatures match, and only qx[3,4] changes in all54 stages.
- [x] Implement a structural one-face25D point binding retaining the original26 prior terms, coefficient domains, observation likelihood and forecast/score.
- [x] Run one bounded4-Newton correction and preserve source/input/resource-bound refusal: four20-iteration PCGs pass; gradient remains0.003145, with15 signature and7 margin refusals among26 trials.
- [x] Identify approaching qy[3,0] by stored-coefficient curl arithmetic, without replaying rejected limiter choices or claiming a root/normal limit.
- [ ] Investigate a justified multi-event or changed-branch policy under a NEW plan. Retain final stationarity, coefficient domains, curvature and branch/normal eligibility; do not force success by lowering margins.
- [ ] Only after a qualified original-input point compute its parameter response and signed nonlinear validation.
