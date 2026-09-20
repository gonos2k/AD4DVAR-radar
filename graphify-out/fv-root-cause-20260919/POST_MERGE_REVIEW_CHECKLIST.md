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
