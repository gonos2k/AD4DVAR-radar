# PR267 tangent-resume RED preflight — 2026-10-09

## Scope and disposition

Read-only design/math review for a proposed continuation from the accepted PR267 endpoint. No FV, production gradient/HVP, guard, solver, test suite, or shared Graphify refresh was run. The archived PR267 raw result and its J objective remain untouched. The proposal can reuse the current tangent-correction mathematics under the existing bounded policy; the runner cannot safely resume by changing its module-level base hash or pointing the existing plan at the new endpoint.

The supported next state is the final accepted PR267 point: control SHA-256 `ed106d7bc277906141e9d4a52aea529c575c9ac17df334fcbfb8d9c5013d44f7`, `theta=0.332026125873074`, native `J=0.0612512070514904`, and the paired 360-stage traces whose saved signature is `67524657…`. This is the input state for a new run, not a rewrite of the PR267 start point or its metrics.

## Saved-array effectiveness check

For each archived step I used the stored residual `F`, residual direction `DF`, accepted `alpha`, and accepted `F²`; no FV or derivative was evaluated. Along the local linear model, the best nonnegative scalar step has

`alpha* = -(F·DF) / ||DF||²`,

and the maximum model reduction divided by `||F||²` is

`cos²(F, DF) = (F·DF)² / (||F||² ||DF||²)`.

The accepted alphas equal `alpha*` to saved FP64 precision in all three steps. Values are:

| Step | cos² | Model-predicted F² decrease | Actual F² decrease | Actual / predicted | Actual relative F² decrease |
|---|---:|---:|---:|---:|---:|
| 1 | 0.13993 | 8.83593e-4 | 8.66492e-4 | 0.98065 | 13.72% |
| 2 | 0.02605 | 1.41930e-4 | 1.50574e-4 | 1.06090 | 2.76% |
| 3 | 0.01112 | 5.89300e-5 | 5.89324e-5 | 1.00004 | 1.11% |

The close actual/model reduction ratios support this as a local model-effectiveness diagnostic. The falling `cos²` and relative decreases show diminishing progress over the archived path. `cos²` describes alignment of the auxiliary residual and its directional derivative; it is not an objective-J score, convergence certificate, forecast metric, or new stopping threshold. Preserve the existing acceptance gates and report the measure descriptively for a continuation. If the review later classifies progress as slow and requests curvature comparison, plan that as a separate bounded diagnostic at the then-current accepted point; the old six HVPs are at previous points/directions and cannot establish current curvature.

At the archived final point, saved arrays give tangent-gradient norm `0.0723738824`, normal mixture component `0.0006383105`, and `F²=0.00523838630`. With the saved side gradients and static face normal, minimizing only the normal mixture component yields `theta≈0.3297935629` and a predicted `F²` reduction of `4.0744e-7` (`0.00778%`). This is an arithmetic-only estimate, not a fresh native endpoint receipt. It confirms that theta cleanup alone cannot replace further control-tangent progress at this point.

## Theta-only scope counterexample

The present helper is a tangent-control correction helper. `candidate_alphas` requires strict side-objective descent and a nonzero control direction (`fv_point_3h_tangent_continuation.py:193-211`), so it intentionally has no theta-only acceptance path. A minimal analytic example shows the boundary: let `g-=(1,0)`, `g+=(-1,0)`, `n=(1,0)`, chart `Z=(0,1)^T`, `H-=H+=0`, and `theta=0.75`. Then the mixed gradient is `(-0.5,0)`, its tangent projection and control direction are zero, `delta_theta=-0.25`, and the merit directional derivative is `F·DF=-0.25`. The theta-only endpoint at `theta=0.5` makes the residual zero, but `candidate_alphas` returns no candidates because the control direction has zero norm (and both side slopes along that zero direction are zero). This is a documented limit of this helper, not a defect in the three accepted positive-control-move steps. Keep any final theta-only cleanup as a separate future stage with its own fresh pair/face/J/input checks; do not relax the current tangent-step gates to admit a zero control move.

## Resume adapter constraints

The existing runner is deliberately tied to its original PR266 base. `BASE_CONTROL_SHA`, `CURRENT_ARCHIVE/RUN/RESOURCE`, `PLAN`, and `REQUAL_PLAN_SHA` are fixed at module scope (`:33-53`). `_load_plan` requires the exact original plan and original base hash/theta (`:379-435`); `_load_current_base` loads one PR266 requalification record and requires the original `6b29dacd…` control and theta (`:440-506`). Child receipts also stamp that old base (`:530-604`, `:945-960`). Updating the global `BASE_CONTROL_SHA` to `ed106d7b…` would therefore leave loader, input-identity, source-pin, and child-receipt expectations inconsistent. Reusing the old plan should continue to refuse because the live audited source no longer matches its frozen source pins.

Use a new resume-specific plan/loader or pass a validated immutable base-state object through the run boundary. Do not monkeypatch the global constants or repin the executed PR267 plan. The new plan should pin the current reviewed code and test files, the PR267 plan/raw/gzip/run/resource receipts, and the new output paths. Before any HVP, the loader should verify the final raw child, gzip-to-raw digest, parent child digest, run/resource equality and successful resource closure; require the last accepted control/theta/J to equal the saved final state; and reconstruct fixed inputs/runtime against PR267's `input_after` while allowing only the control identity to advance from the PR266 base. Recompute current native J, face value, both side gradients/traces, and strict pair at `ed106d7b…`; compare them with saved final-repeat evidence within existing units/dtype tolerances before proceeding. Fresh side HVPs must be computed at this new base and current direction, never reused from prior steps.

Keep the frozen objective J, selected face `Q=0`, current branch-pair/side-objective/face/Armijo/fresh-repeat gates, and current hard budget: at most 3 accepted corrections, 6 fresh HVP calls, 16 candidates per correction, radius `0.05`, 240-second internal / 300-second outer wall limits, and 1 GiB sampled resource limit. The existing raw J values are history; the first new J Armijo comparison must use the freshly reconstructed native J at `ed106d7b…`. Do not change J to optimize `F²`, change its Armijo constants, or turn low `cos²` into an unreviewed stop/acceptance rule.

If a later slow-progress review authorizes curvature comparison, measure curvature at the then-current endpoint with new products. For constrained curvature on `Q=0`, use the restricted Lagrangian Hessian (including the face-curvature term from the normal multiplier), not an unconstrained ambient Hessian or a stale one-direction HVP. A positive/negative local curvature observation at this nonstationary point is not a minimum/root certificate. Curvature budget and acceptance semantics belong in that later plan; they are not added to the next 3-step/6-HVP continuation here.

No solver replacement, new efficacy threshold, theta-only implementation, or production launch is proposed in this preflight. A final source-and-plan audit is required before any later guarded launch.
