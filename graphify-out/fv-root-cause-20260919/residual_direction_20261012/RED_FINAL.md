# RED final — residual-direction helper

## Result

No helper-level mathematical blocker found in the tested supported domain.
`build_residual_direction` constructs the minimum-mixture tangent seed with the
Euclidean face projector, validates the supplied chart/pivot override, applies
the caller-owned directional action, projects the raw residual derivative, and
reports the actual products without assuming descent.

## Evidence

- `.venv/bin/python -m pytest -q tests/test_fv_point_3h_residual_direction.py`:
  4 passed.
- A 26-variable curved `tanh` face compares `DG[d]` to direct autodiff of the
  full projected-gradient map and compares its path derivative against an
  independent chart-path JVP.
- Synthetic checks cover Euclidean normal-scale invariance, unsupported jump,
  unresolved jump, chart mismatch, and finite adverse curvature whose side and
  merit gates remain false.
- Incremental graph refresh completed through the repository Graphify CLI;
  it re-extracted 651 code files and updated the structural graph. HTML export
  was skipped because the graph has 11,672 nodes, above its 5,000-node limit.

## Conditional limitation

This is pure synthetic helper evidence. It does not validate the production
callback's fresh derivative actions or the main solver's two-arm HVP and
acceptance path. The helper reports `g*·P DG[d]`; if the main algorithm also
needs `G·DG[s]`, the main caller must evaluate and gate that separate product.

## Integrated model formula and seed review

The production `minimum_tangent_model` uses the minimum-mixture derivative

`theta' = -(j·Hmix + g*·(H+ - H-)) / ||j||²`.

For the supported smooth common face, `j=k n`; differentiating `n·g*=0`
along the face yields `n·g*' + n'·g*=0`. The formula above includes the
`g*·(H+ - H-)` term needed for the changing-normal contribution. An analytical
26-control tanh-face probe formed the Hessian actions explicitly (no HVP calls):
the model derivative matched the independent derivative of
`G(x)=P(x)∇Phi(x)` with `1.5e-16` norm error, and the changing-normal closure
had `1.7e-17` absolute error.

With `seed_direction` set to the baseline direction, the helper passes that
exact direction to its callback and normalizes the projected action to the
same norm as the seed. The curved-face identity `G·DG[s] = -c||P DG[d]||²`,
where `c=||d||/||P DG[d]||`, held at float64 precision in the explicit-action
probe. The fixed production budget remains two side HVPs for `d` and two for
`s`; the helper itself invokes only the caller-supplied action for `d`. No FV
run or endpoint comparison is included in this RED formula review.

## Frozen attempt 1

Independent SHA and member checks passed for the frozen plan
`cd9db66869652c8807264dcdb9172f94e4c65118fc3982465e516e706502fa64`, source
bundle `da936a87c96bcb5970afbed650ba00760949763652f77febfebdd9b44d6ac2b8`,
234 source files, and 302 archived inputs. The recorded run completed in 54.0
seconds under its 660-second outer/600-second internal policy, with four fresh
HVPs, zero GN rows/solves, and one commit. Source, fixed input, and runtime
identities closed.

From the stored models and pinned base endpoint, `s` exactly equaled the
normalized Euclidean projection of `r_d` to `||d||` (zero vector difference in
the replayed float64 arithmetic). The stored second action satisfied
`G·DG[s] = -c||P r_d||²` within `1.1e-16`; its fresh model gate also passed.
Both actual arms passed their independent P2/full-G eligibility checks, but the
two-direction QP selected `[baseline, 0]`. Both arms therefore evaluated the
same candidate (`J=0.06098887837`, `R=0.00243053463`); the residual direction
added no selected candidate gain in this attempt. The one bounded commit moved
from base `J=0.06099005812`, `R=0.00244126957` to that candidate. No score,
forecast, root, minimum, or response claim was made. This is one conditional
local comparison on the pinned point, not evidence for a broader physical or
forecast improvement.
