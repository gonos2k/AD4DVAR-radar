# PR164 follow-up: joint derivatives and a matched small inverse

Baseline: merged `main c5d3f28c`. Phase1 synthetic/local evidence only.

## Mathematical scope

The equation and positive shared-face FV operator are unchanged. The new oracle
uses the actual `robust_objective` and `forecast_fv_analysis`, both with minmod.
All nine initial cells, two constant-flow streamfunction coefficients and growth
are optimized. Its dense Newton solve is a small verification oracle, not a
claim about the production GN optimizer or recovery from arbitrary initial guesses.
The declared nonzero starting control differs from the generating truth.

The background is `B(y,theta)=y[0]+theta*pattern`, with nonzero theta=.02.
It enters the objective and verification forecast through the same contract.
With `p=(y,theta)`, the response is

`H.T lambda = E_c; dE/dp = E_p - (D_p grad_c J).T lambda`.

The fully detected data, known boundaries, support, precision and verification
target are fixed. Initial echo is at least30, so the background is about14.77dBZ
or higher, far from the -10dBZ transform floor under the tested .001 perturbations.
This is mean-background differentiation, not typed uncertainty/support learning.
The verification target is synthetic and frozen conditional on the nominal
solved control (`nominal forecast + .1*pattern`). It tests a nonzero local score
gradient; it is not independent forecast validation or a held-out learning test.

Each analysis+forecast evaluation inspects54 actual Euler-stage inputs (three
intervals, nine substeps, two RK stages). Every face flow is strictly positive
with a roundoff-scaled margin; controlled growth is positive. Interior slopes
are positive and strictly unequal. Perimeter slopes are zero by the operator's
definition for all controls. Thus this fixture has a locally smooth composite
map without treating a merely nominal zero flux as irrelevant.

Newton candidate endpoints and reanalyzed endpoints retain the RK choices.
This proves neither a finite segment certificate nor a general relevance
classifier. `finite_path_certified` and `general_minmod_response_eligible` remain
false. The public donorcell-only response gate is unchanged.

## Executed evidence

`minmod_joint_inverse_final.json` records environment, controls and producer
SHA256 values. `minmod_joint_inverse_final.run.json` records bounded execution.

| Quantity | Result |
|---|---:|
| Maximum stationary gradient | 6.5656855e-12 |
| Dense Hessian minimum eigenvalue | 4.5108884 |
| True adjoint relative residual | 1.2201763e-15 |
| Minimum normalized interior slope margin | 0.0725392 |
| Wrapper time / sampled peak RSS | 304.54s / 340426752bytes |

| Direction | h | Adjoint | Central reanalysis | Absolute difference |
|---|---:|---:|---:|---:|
| Observation | .001 | -.00401000304652 | -.00400997311218 | 2.9934e-8 |
| Observation | .0005 | -.00401000304652 | -.00400999556319 | 7.4833e-9 |
| Background parameter | .001 | 1.05756490889e-6 | 1.05756490184e-6 | 7.05e-15 |
| Background parameter | .0005 | 1.05756490889e-6 | 1.05756501234e-6 | 1.03e-13 |

Observation errors reduce by approximately4; parameter differences are already
near the cancellation/root-solve floor, so no artificial fourfold rule is imposed.
These are directional local derivative results, not finite-impact accuracy or
held-out learning improvements. The positive Hessian is used only after the
strict local operator branches have been established.

The initial run stopped at its180s limit (339787776bytes), before completion.
It is retained as `minmod_joint_inverse.run.json`, not counted as success or a
numerical failure. The completed run includes endpoint checks added by RED review.
No unchanged180-minute or240-grid experiment was repeated.

## Regression and display

- `test_fv_minmod_joint_derivatives.py`: signed, nonzero face flows on4x5;
  initial/flow/growth and their three pairwise directions; both-sign state and
  gradient differences at two step sizes; scaled roundoff floor; nonzero mixed
  Hessian blocks; replay gradient/HVP equality across9 substeps. Zero-flow
  forward identity has no joint-Hessian safety claim.
- `test_fv_minmod_inverse_probe.py`:54-stage tracing, controlled-zero flux/growth
  and active limiter-tie rejection.
- 32 focused cases passed in the bounded combined run. Two additional publisher
  scope cases passed in its18-case rerun: **34 distinct cases**, not50. The full
  inverse is a separately recorded experiment, not a cheap automatic test.
- Original HTML frame-data SHA256 values are unchanged. The new evidence panel
  separates the3x3 inverse from the original animation and preserved240-grid data.
  Aside desktop DOM and screenshot (`minmod_joint_inverse.png`) confirmed all
  four rows and the scope warning. No new mobile check was performed.
- Graphify reused the structural cache:6558 nodes,64870 edges,238 communities;
  no semantic extraction. GREEN/RED reviewed the local-scope claims and evidence.

## Remaining work

General minmod branch relevance under arbitrary joint controls, exact response
API integration, finite-impact validity, typed prior precision/support learning
and full-chain D7 remain open. Full CPU CI was not started. This closes a small
matched-inverse verification step; it does not assign an overall completion rate.
