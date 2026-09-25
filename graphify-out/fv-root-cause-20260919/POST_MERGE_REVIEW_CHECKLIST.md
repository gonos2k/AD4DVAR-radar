# PR160 follow-up: theoretical FV scope

Baseline: PR head 0eb2a9d, merge 5d4666c, 2026-09-19.
Reuse stored numerical evidence; do not repeat the 240-grid runs unchanged.
Earlier Phase 1 completion and checkbox counts are not completion percentages
for this expanded FV scope. Real-data skill remains Phase 2.

- [x] Record final CPU run 35424239984 and bind its tested tree to main.
  Cancelled at the user's request; UI, packaging and typechecking passed, but
  the full CPU suite did not complete. Head and merge share tree
  dc06c91484ae24fae659bb253cffed1dd2678c8f. This closes status accounting,
  not full regression verification; use focused checks during follow-up.
- [ ] Connect typed background mean and precision at nonzero parameters.
  Check total observation/parameter derivatives against polished reanalysis;
  hold support fixed for differentiation and explicitly classify support changes.
  The pending external parameterized-mean API alone does not close this item.
- [ ] Connect partial observations from preparation through analysis, forecast
  and response, using one mask/support contract. Preserve grid/time and public
  output restrictions until their corresponding integration checks pass.
- [ ] Define the finite-impact domain using both signs and multiple directions.
  Report actual and linear changes, signal size, and absolute/relative error;
  declare tolerances before evaluating cases. Central derivative agreement is
  not finite-impact accuracy. Existing errors relative to actual changes are
  about 57.94%, 40.78%, 221.37% at +0.001, +0.0005, -0.0005 dBZ.
- [ ] Check field/JVP convergence over the intended forecast duration and record
  full D7 preparation, solve, refinement, forecast, adjoint, reanalysis, learning,
  serialization and retry costs. The 30-minute prescribed-flow probe and one
  490-second adjoint do not establish this whole-chain evidence.

Retain donor-cell exact-response limits unless a separately verified limiter
path is introduced. Keep exact-boundary numerical tests separate from synthetic
forecasts that use only boundary information available at issuance. Do not tune
on held-out outcomes or relax stationarity gates to obtain a pass.

## PR161 observation/branch follow-up

- [x] Normalize parameterized builder inputs by detected/censored/invalid status;
  censored numeric placeholders must not become continuous observations via B.
- [x] Reject the initial-transform clamp join in both B-dependent response modes;
  retain the equation and both smooth branches. This is not finite-path certification.
- [ ] Complete the same-duration low-diffusion comparison before changing the
  exact-response transport backend. See BACKGROUND_CONTRACT_REVIEW.md for metrics
  and branch/derivative checkpoints; increasing substeps alone does not remove
  donor-cell spatial diffusion.

## PR162 regression completion and comparison

- [x] Use the second input's freshly prepared state throughout the censored
  analysis/refinement/forecast/response regression, with unchanged numeric outputs.
- [x] Preserve nonzero-control B and mixed B/control derivatives in repository
  tests using closed-form and central gradient-difference oracles.
- [x] Measure existing donorcell and minmod on identical 180-minute conditions,
  with fixed physical threshold, moments/peak/area, field error and selected-AD
  JVP diagnostics. Results and scope: REGRESSION_AND_TRANSPORT_COMPARISON.md.
- [ ] Certify limiter branches and discrete derivatives before an inverse
  minmod response/reanalysis or any donorcell-only gate change. The comparison
  improves field-error evidence, not general FV/FSOI or typed-prior completion.

## PR163 engineering gates and branch counterexamples

- [x] Direct minmod forward/JVP option propagation and metadata regression.
- [x] Compare scheme/domain/time/threshold/case-grid coverage/CFL/producer identity
  before rendering; reject a donorcell report under a minmod filename.
- [x] Preserve actual-step tie counterexample, smooth directional check, and
  fixed-zero-flow identity case in repository tests.
- [ ] General RK-stage relevance/branch analysis across initial/flow/growth
  controls, mixed/replay verification and small inverse minmod response.
  See MINMOD_BRANCH_AND_PUBLISHER_REVIEW.md; no donorcell gate was removed.

## PR164 joint derivative and inverse follow-up

- [x] Exercise joint initial/flow/growth directions, two-sided state and gradient
  differences, mixed blocks and replay across the eight-substep tape boundary.
- [x] Solve a small **minmod** robust objective for all 12 controls; inspect all
  54 analysis/forecast RK stages and compare observation/parameter adjoints with
  actual plus/minus reanalysis. See MINMOD_JOINT_INVERSE_REVIEW.md.
- [x] Preserve the public donorcell-only response restriction and original demo
  arrays; display the small inverse as separate evidence.
- [ ] General control-dependent relevance classifier and finite-path branch proof.
- [ ] General minmod response API, typed mean/precision/support learning, valid
  finite-impact range and whole-chain D7 remain open.

## PR165 spatial inverse follow-up

- [x] Validate archived 3x3 stationarity, adjoint residual, positive curvature,
  branch margin and arithmetic error consistency before rendering. Bind the
  canonical report (including source hashes) to the measured PR165 artifact.
- [x] Preserve per-cell limiter choices and per-face signs; regression must
  distinguish spatial layouts collapsed by the former `.all()` signature.
- [ ] Run the 4x5/26-control inverse with endpoint controls, scores, gradients,
  signatures and phase costs, then compare the default product GN tolerances.
  Local response and optimizer convergence are separate completion conditions.

## PR166 local stationary path

- [x] Correct diagnostic elapsed time from a captured monotonic start; isolated
  AST check returns3.25s with injected timestamps10 and13.25. Historical reports
  are preserved; no old experiment is rerun for this logging fix.
- [x] Reuse the 26-control root, compute exact parameter-to-control tangent,
  and test raw sin(k), k=0..59, theta fixed, with h=.001*2^-j.
- [x] Obtain two successive matched-branch plus/minus stationary pairs at the
  unchanged1e-10 gradient gate and compare central reanalysis with adjoint.
- [x] Show measured local range separately from original .001 finite impact,
  general minmod response, product GN convergence and learned forecast skill.

The PR166 local observation task is closed at h=.000125 and .0000625; see
MINMOD_LOCAL_PATH_REVIEW.md. Parameter reanalysis, general minmod response,
original .001 finite impact, typed learning and whole D7 remain open.

## PR167 cache and independent directions

- [x] Check cached direct/rhs against freshly evaluated score derivatives, exact
  parameter directions, and numerical contract/payload identity before reuse.
- [x] Reuse the saved theta tangent for two consecutive plus/minus pairs at the
  unchanged stationarity, branch and relative derivative gates.
- [x] Predeclared second observation direction: uniform +1 at the middle of
  three observation times, zero at the first/last times and theta. Use the same
  h=.001*2^-j schedule, without changing direction or normalizing after results.
- [x] Publish separate measured direction results; preserve the archived sine
  result, original animation and all general-response/learning restrictions.

Reuse the nominal Hessian and accepted controls. No repeat of the old sine,
180-minute transport or 240-grid experiments is required for this follow-up.

Closed in MINMOD_CACHE_AND_DIRECTIONS_REVIEW.md: both new directions passed
at h=.001 and .0005. This closes these local response tasks only.

## PR168 reusable research response and matrix-free bridge

- [x] Isolate conditional local-response algebra with mandatory caller branch
  checks, fresh direct/mixed derivatives, exact HVP PCG and true transpose residual.
- [x] Keep minmod adaptation in the bounded full-support research example and
  reject support, reconstruction or nominal branch outside that contract.
- [x] Compare all 26 HVP basis columns with the archived Hessian (relative
  Frobenius error <=1e-10), matrix-free/dense adjoints (<=1e-8), and all three
  archived direction responses (relative difference <=1e-6). The independent
  transpose residual must be <=1e-10. Criteria are declared before execution.
- [x] Publish measured matrix-free diagnostics separately; no nonlinear
  reanalysis, new direction, public donorcell-gate change or learning claim.

Closed as a conditional research bridge in MINMOD_MATRIX_FREE_REVIEW.md.
General minmod API eligibility and all previously listed learning/D7 limits remain open.

## PR169 follow-up — portability and full parameter response

- [x] Resolve archived source paths relative to declared archival root; test relocated root, same basenames and mismatch rejection. Preserve historical reports.
- [x] Compute full parameter VJP once; project onto existing three directions (relative difference <=1e-6); retain direct/indirect components.
- [x] Feed actual archived GN controls through the current objective/branch gate; reject before PCG. Qualified archived stationary point must pass unchanged gate.
- [x] Separate preparation, response and comparison costs; new bounded measurement only, no optimizer/reanalysis reruns. Final GREEN/RED review and KG evidence.

## PR170 follow-up — explicit GN refinement to response

- [x] Conditional workflow: stationary input skips refinement; unqualified input explicitly refines or rejects; failures never return sensitivity.
- [x] Actual archived product GN control -> existing bounded oracle with fixed branch -> whole parameter response. Keep objective, score, 1e-10 gate and reference verification unchanged.
- [x] One report records both controls, J/gradient/branch before and after, refinement use, actual adjoint residual, full sensitivity and separated costs. No fresh GN or perturbed reanalysis run.
- [x] Small success/failure regressions, actual bounded execution, GREEN/RED review, KG and original demo update. General solver/FSOI/learning/D7 remain open.

## PR171 follow-up — refiner input isolation and current GN

- [x] Clone fixed p for external refinement; reject mutation and invalid candidate layout; regression covers successful/failed mutation with original p and control preserved.
- [x] Freeze sources before one bounded current GN -> explicit refinement -> full response execution. Keep independent fixed verification and the same numerical gates.
- [x] Record actual returned GN control, stage costs, before/after branch/gradient, full response and source stability; preserve archived evidence.
- [x] Focused tests, GREEN/RED review, KG/checklist and original demo. General new-case convergence/FSOI/learning/D7 remain open.

## After PR172 — matrix-free stationarity refinement (new milestone)

PR172 input isolation and current GN integration are closed. This milestone
changes the numerical refiner, not the objective, observation contract or gates.

- [x] Exact gradient-JVP Newton-PCG root refinement, fixed max-gradient <1e-10;
  independent linear residual <=1e-10, negative-curvature/nonconvergence refusal,
  branch-checked Armijo decrease of half the squared gradient norm.
- [x] Small analytic nonlinear tests with predetermined starts, including 64
  controls (above the dense oracle cap), and explicit refusal cases. No global
  SPD, minimum, convergence or finite-path certification from PCG diagnostics.
- [x] One bounded 26-control saved-GN -> new refiner -> full response execution;
  reuse existing dense results. Control distance <=1e-8, full-gradient and three
  directional relative differences <=1e-6; unchanged response residual <=1e-10.
  Record inner/outer iterations, HVPs, actual residuals, accepted scales, branch
  records and separate costs. No fresh GN or perturbed reanalysis rerun required.
- [x] GREEN/RED review, affected tests/typecheck, KG and original demo evidence.
- [ ] Larger FV grids and different observation cases: estimate costs and define
  convergence/refusal cases before execution; not covered by the small bridge.

## PR173 review follow-up — recoverable candidate domain failures

- [x] Reproduce original exp(x)-x behavior at x=-2,-7,-8 directly from b38fdb7.
- [x] Distinguish explicitly detected nonfinite candidate evaluations from
  initial-point failures and arbitrary callback exceptions; backtrack only the
  former, preserving budgets, derivative equations and gates.
- [x] Regress overflow recovery, nonfinite derivative/norm candidates, exhausted
  search, malformed outputs and exception propagation; retain 64-control coupled
  analytic evidence separately from actual FV execution.
- [x] GREEN/RED final review, affected tests/typecheck, Graphify/KG and PR update.
- [x] Costed 8x10/86-control FV contract and execution plan; obtain the previously
  required larger-experiment budget approval before numerical scaling runs.


## Approved 86-control discrete execution (after PR173)

- [x] Explicit physical-coordinate geometry, CFL, growth/time and boundary
  schedules; sum objective preserved; dimension/stage counts derived.
- [x] 120-second cost gate and two predeclared starts, each with 1800-second /
  2-GiB sampled resource guard; all outcomes and termination records preserved.
- [x] Current GN -> fixed local branch -> matrix-free correction -> whole
  parameter response. Both starts eligible; not independent FD certification.
- [x] Same problem/source identities, whole-VJP projections, between-start
  comparison, separate times and measured peak RSS; GN reference cost correctly
  distinguished from supplied-seed costs and internal candidate policy.
- [x] 128 affected tests, clean four-script typecheck, GREEN/RED final review,
  KG and original demo evidence; measured sources preserved before typing edits.
- [ ] Separate milestones: same physical/statistical inverse-grid convergence,
  other state/observation/support cases, independent scaled response validation,
  full-spectrum/observational identifiability, typed learning and general D7.


## PR174 follow-up — separate statuses and common problem

- [x] Process exit/source/phase gates separated from saved numerical eligibility;
  abnormal/limited/early runs never inflate completed-eligible counts.
- [x] Preserve raw FV reports and original summary; write new status summary
  with unperformed response validation kept separate from algebraic checks.
- [x] Shared thin definition over existing observation/frozen/transport objects;
  explicit collocated dBZ support, layouts and regular times; old profile guards.
- [x] Immutable-reference J/E/gradient/HVP/signature parity at both saved points;
  no new nonlinear solve, no physical or statistical model change.
- [x] 158 affected tests/typecheck, source fixture tracking, KG and display.
- [x] Independently budgeted middle-time-bias 86-control reanalysis check
  (FV86_REANALYSIS_RESULTS.md): seed A, two local sizes pass.
- [ ] Arbitrary observation/mask/time extensions remain separate milestones.


## Approved 86-control local reanalysis and review checklist

- [x] Create PR174_REVIEW_RESOLUTION.md with per-item scope, evidence and state.
- [x] Freshly bind cached nominal derivatives/adjoint and form matrix-free tangent.
- [x] Execute predeclared signed parameter perturbations; retain predictor refusals.
- [x] Two consecutive local sizes pass normality/branch/central-slope criteria.
- [x] Keep scope to one nominal point/direction; leave general and finite-path flags false.
- [x] Preserve source/input/resource records; R3/R6 closed only in that scope.
- [ ] Other review axes remain as listed in PR174_REVIEW_RESOLUTION.md.

## PR175/176 follow-up — integration and regular forecast leads

- [x] Integrate PR176's unchanged tree into `main` via PR177 (`e829a17`);
  distinguish the prior development-branch merge from this integration.
- [x] Preserve existing one-lead numerical behavior while adding an explicit
  two-lead fixed-support research profile at 180/240 seconds. Check full
  future-boundary schedule, growth subdivision, all 72 branch stages and
  local score JVP; see `PR175_176_REVIEW_RESOLUTION.md` (G2).
- [x] Bounded 4x5 partial-observation profile with fully known first frame and
  boundaries: two genuine later-frame missing cells, fixed-mask J/E and mixed
  derivative invariance, correlated-whitener masking, and explicit refusals
  (G3, `FV_PARTIAL_OBSERVATION_RESULTS.md`).
- [ ] Qualify a stationary point and independently check the partial-profile
  whole response with signed reanalysis (G3b); the one-lead problem, direction,
  gates and resource cap are in `FV_PARTIAL_REANALYSIS_PLAN.md`. The approved
  first guarded attempt stopped at nominal Newton correction before adjoint or
  endpoints; see `FV_PARTIAL_REANALYSIS_RUN1_RESULTS.md`. It remains open.
- [x] Diagnose the final Newton candidate refusal categories in a separately
  approved nominal-only run, capped at 240 s / sampled 1 GiB: all 16 final
  backtracks were branch-policy refusals, with zero nonfinite/finite-Armijo
  refusals. See `FV_PARTIAL_REFINEMENT_DIAGNOSTIC_RUN2_RESULTS.md`. G3b response
  validation remains open.
- [ ] Identify which branch gate refused the candidates, then justify any
  nominal-search policy change separately from final-point response eligibility.
  Run 3 quantified 3 first-failing signature checks and 13 first-failing
  face-flux-margin checks (possible overlap); the first-failure gate identity
  is closed for this nominal correction, while any solver
  policy change remains open and requires mathematical justification. See
  `FV_PARTIAL_BRANCH_GATE_DETAIL_RUN3_RESULTS.md`.
- [x] Bounded maintain/decay/growth fixed-branch classification and zero-face/
  flat forward-versus-response refusal (G4). See `FV_GROWTH_BRANCH_SCOPE.md`.
- [x] At zero-flow control, prove every face of the supported 4×5/8×10
  five-mode profiles is control-sensitive; distinguish a restricted one-mode
  structural-zero counterexample, retaining the strict full-flow refusal
  (G4b, `FV_ZERO_FACE_CONTROL_SCOPE.md`). A future restricted-control response
  profile needs its own contract and verification.
- [x] Bounded off-grid interior point-dBZ operator with independent
  observation-space diagonal errors, exogenous full state-grid background and
  fixed-mask/geometry identity (G5, `FV_POINT_OBSERVATION_RESEARCH.md`).
- [x] Fixed same-time point-observation correlation with symmetric principal
  whitening, exact diagonal limit, analytic two-point gradient, permutation
  and singularity checks (G5b, `FV_POINT_CORRELATION_RESEARCH.md`).
- [x] Genuinely missing fixed point-observation rows under diagonal or
  declared same-time correlation: select valid principal covariance before
  whitening, retain inactive parameter slots with zero derivatives, preserve
  the exogenous grid background, and reject censored/QC meanings. The bounded
  fixed-control FV probe and regressions are in `FV_POINT_MISSING_RESEARCH.md`.
- [ ] Regridded/footprint radar products, censored/QC point rows,
  cross-time covariance and a newly qualified stationary whole response
  remain separate G5c work. An entirely missing observation frame is also
  outside the current bounded point profile.
- [x] Bounded regular 10-minute/18-lead forward schedule with 90 substeps per
  interval, 3600 SSPRK stages, same-operator zero/nonzero-flow truth, explicit
  verification times and observed refusal of strict response eligibility
  (G6a, `FV_LONG_HORIZON_RESEARCH.md`).
- [ ] Irregular observation times, evolving future flow/growth/boundaries and
  long-horizon response remain G6 work.
- [x] G7 analytic coordinate/objective rescaling invariance and the existing
  86/241 toy probe's identity/branch/direction/local-step recording contract
  (`FV_RESPONSE_UNITS_SCOPE.md`).
- [ ] G7 new FV profiles still need a qualified stationary response and
  signed endpoints; cross-profile units/conditioning and a versioned general
  tensor identity are separate unfinished contracts.
- [x] Retrospective G8 four-stratum/six-case accounting keeps execution,
  forecast, stationarity, branch, local-response eligibility, response
  computation, independent validation and
  resource states separate; no cross-profile percentage or success-only
  denominator (`FV_SUPPORT_MATRIX_RESULTS.md`).
- [ ] Prospective radar-condition case matrix and rates (G8), broader
  concurrent integration (G9), and independent physical/finite-impact
  validation (G10).
- [x] G9 minmod-stage observer context replaces global branch/face-margin
  function patching; nested, concurrent and ordinary zero-face stage paths
  are verified (`FV_STAGE_OBSERVER_SCOPE.md`).
- [x] G9 PCG instrumentation in the FV86 execution/signed reanalysis and
  partial-observation producers now uses a per-call context instead of
  module-alias patches; direct, nested, threaded and imported-alias tests
  verify routing (`FV_PCG_OBSERVER_SCOPE.md`).
- [x] A bounded in-process two-thread 4×5/8×10 response run refused during
  simultaneous forward AD; preserve its raw failure and the independent tiny
  two-thread reproduction (`FV_CONCURRENT_RESPONSE_RUN1_RESULTS.md`).
- [x] The same fixed local responses completed in separate guarded Python
  processes with distinct monitored PIDs, overlapping actual HVP calls,
  unchanged input/source/archive identities, strict branch and residual gates,
  and archived-value parity (`FV_PROCESS_ISOLATED_RESPONSE_RESULTS.md`).
- [x] G9 asyncio task-local diagnostic ownership: sibling tasks retain their
  own stage/PCG records, and delayed inherited task/`to_thread` calls that
  start after scope exit cannot append to that collector. In-flight callbacks
  still require completion before report publication. See
  `FV_ASYNC_OBSERVER_SCOPE.md`.
- [ ] Full concurrent GN/refinement, in-process simultaneous forward-AD
  response, production async-service publication/lifecycle integration and
  general input coverage remain separate unverified G9 scope. The process result is limited to
  two archived synthetic cases and is not an independent reanalysis or
  physical-accuracy check.
