# Stationarity and learning: bounded numerical closure

## Equations and scope

The existing donorcell FV analysis/forecast and robust objective J are unchanged.
For fixed masks, observation statistics, geometry and boundary schedules, solve
F(c,y)=grad_c J(c,y)=0. With nonsingular exact H=J_cc, the local total derivative is
E_y - J_cy^T H^{-T} E_c. IRLS Gauss–Newton replaces H and is explicitly approximate.
For a declared calibration y(theta), chain the total derivative with dy/dtheta;
this is a retrospective supervised DA update, not causal observation attribution.

## Checkpoints

- [x] Reuse cold/warm endpoint failures and factored cost-difference experiment.
- [x] Identify evaluation precision below the predicted objective reduction; do not loosen production cost acceptance.
- [x] Remove arbitrary absolute cost slack from the <=32-control verification oracle.
- [x] Verify gradient-root refinement, positive local exact curvature and each refinement step's donorcell branch box.
- [x] Verify exact adjoint and finite reanalysis Taylor agreement on the actual FV objective.
- [x] Verify learned parameter, fixed held-out reanalysis improvement and save/reload equivalence.
- [x] GREEN/RED review, record failures and scope, expose verified evidence in the demo.

The gradient-root oracle uses the merit function ||F||²/2, whose directional
derivative for an exact Newton step is -||F||². A residual decrease therefore has
a mathematical meaning even when separately evaluated costs cannot resolve a
change. It does not assert global minimization. A positive Hessian at a sampled
point is local numerical evidence, not an interval proof over an unknown path.
Production FV stationarity eligibility and the 240x240 case must not inherit a
small dense-oracle result. No new registry or certification type is proposed.

Executed bounded evidence and remaining generalization limits: LEARNING_RESULTS.md.
The selected learning target is log observation-error scale; its mixed derivative
J_c_theta is evaluated directly, rather than chaining an observation calibration.

## Follow-up checkpoints (2026-09-19)

- [x] Explicit 240×240 local first-order verification on its own saved analysis.
- [x] Exact robust-Hessian research response option; keep approximate default named.
- [x] Actual three-parameter neural error-model update and held-out replay.
- [x] Total fixed-NN observation response including features and B=y[0].
- [x] Original HTML, GREEN/RED integrity review, code graph and source manifest.
- [ ] Independently resolve the tiny NN-specific observation contribution.
- [ ] General FV boundary/support/nonsmooth contracts and large-grid exact adjoint.
- [ ] Background neural-prior integration and persistent learning.

See LOCAL_VERIFICATION_RESULTS.md for the executed evidence and why these local
results do not establish general eligibility or Phase 2 real-data performance.
