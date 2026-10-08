# Face-transition diagnostic RED preflight

Date: 2026-10-08
Decision: **GO for the single bounded diagnostic launch**, subject to the already frozen plan. This is a preflight only; no FV, objective, gradient, HVP, PCG, score, adjoint, reanalysis, or guarded launch was executed by this review.

## Pin and contract checks

- The diagnostic plan is `FACE_TRANSITION_DIAGNOSTIC_PLAN_20261008.json`, SHA-256 `6b79d07ce609ebe35879dcb3f1cec95649a786155c392e7ea041063b4e19283e`. I independently recomputed this file hash and read its scope: 113 source pins, 54 archive pins, external `Qy[4,3]`, resume control `2cdccade…`, η values −2e−6, −1e−6, 0, +1e−6, +2e−6, one guarded launch, 240/300-second limits, 1-GiB sampled RSS, and zero HVP/PCG/optimizer calls.
- The root reported that real plan loading and `_validate_model_source` passed, that the 22 focused tests passed in 2.15 s, and type checking had zero errors. Those are producer-reported checks, not runs performed for this RED review.
- The code binds the result to the completed PR #259 resume endpoint and checks current source/input/runtime closure. The four nonzero probes reconstruct the smooth face chart and verify the analytic face value against production flux. The zero point calls only `_cost_only`; no gradient, Φ, AD, or branch check is requested there.
- The detached minmod observer is installed only around `branch_check` under `torch.no_grad()`. It records analysis donor traces only (360 stages); branch metadata still counts and partitions the full analysis-plus-future signature. The plan intentionally does not claim future donor traces. Boundary edge reconstruction applies the growth scaling only at the matching start stage for positive growth and chooses the upwind side by face orientation/sign.
- The full 26-control objective/gradient path remains in use, so the pivot prior is retained. The chart is local to this diagnostic and does not create a persistent face constraint.

## Mathematical scope and interpretation gates

The current jump helper compares both finite pairs (±1e−6 and ±2e−6) against a common zero-chart face normal, records ambient gradient jumps and their normal/tangent projections, and computes each pair's sampled gradient-segment minimum-norm point. It also records the algebraic Φ jump identity from full gradients. The identity is finite-sample algebra, not an exact one-sided derivative or proof of Φ continuity/discontinuity at zero. The segment minimizers are not Clarke subdifferential certificates, and they do not imply an active-face minimum or root.

The branch comparison can identify which stored endpoint signatures differ and whether changes occur in analysis or future stages. It cannot prove a continuous path is on one branch or attribute a J/Φ change to this single face. If a nonzero sample is branch-refused or its observer/margin record is incomplete, the child status must remain `branch_refusal` or incomplete; process completion alone does not qualify a complete two-sided comparison. The execution/resource status, sample status, branch eligibility, and trace completeness must remain distinct in the final interpretation.

## Bounded launch and remaining limitation

The frozen policy is appropriately narrow: one serial guarded launch, no optimizer, no Hessian/HVP, no PCG, no score or response. The code durably writes the cost-only zero record and each returned nonzero sample; deadline checks surround objective, gradient, branch tracing, summary and final identity closure. A budget refusal must retain all previously written samples and must not be described as a completed five-point comparison.

No required prelaunch defect remains in the reviewed draft. The only review limit is evidence scope: preflight does not verify actual FV execution, endpoint stability, forecasting skill, or causal attribution. Proceeding with the one planned bounded diagnostic is justified; any resulting report should remain within those limits.
