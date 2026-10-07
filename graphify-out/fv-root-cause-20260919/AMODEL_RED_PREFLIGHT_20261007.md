# Model-guided full-space continuation — RED preflight

Date: 2026-10-07
Disposition: **GO for the single bounded launch under the frozen plan.** This preflight is read-only; it is not evidence that any new full-space step has been accepted. No FV, HVP, PCG, forecast, adjoint, reanalysis, or test was run by this reviewer.

## Frozen plan and scope

Plan SHA-256: `a75b5e474e68fb74475c3fd7fb2f31e85c891edcb06c40e83b4287b16a7b3136`. I independently hashed its **111 source pins and 43 archive pins**; all match the current files. The parent reports its real plan/base loader preflight passed. The plan starts from the committed PR #258 endpoint `058895848fdeb348e8d6bb1f22ac13e3f2dadfa48bff46522ffbf94fedf72ffa`, retains the fixed original problem/parameters/prior, and declares at most 3 full-space iterations, at most 3 fresh current-point HVPs, at most 16 dyadic candidates per iteration, radius 0.05, no PCG, one guarded launch, 240 s internal / 300 s external / sampled RSS at most 1 GiB.

## Math and candidate policy

At each committed current point, the loop uses a fresh full gradient (g), chooses (d=-g), and computes a new live (Hd) by JVP of the full gradient at that point. It checks the resolved slopes (g^Td=-\|g\|^2<0) and (g^THd<0), then sets

\[
\alpha_0=\min\left(1,\frac{0.05}{\|d\|},\frac{-g^THd}{\|Hd\|^2}\right).
\]

The last term is a conservative limit for decrease in the linearized-gradient merit model. It is not a guarantee of actual Φ decrease. Every candidate is a straight full-space control (c+\alpha d); its 26-control displacement equals α∥d∥ and is bounded before FV evaluation. The original full objective and all prior terms remain active. Candidate acceptance requires actual J Armijo, actual full-gradient Φ Armijo plus a roundoff-resolved actual Φ decrease, and that candidate's own complete strict branch/margins. A target-face crossing is possible and is recorded through the actual flux and endpoint branch; no path-smoothness claim follows.

The model uses a new HVP at each current point. It does not reuse the tangent-step HVP, `alpha/2`, or any cached old-point Hessian. `full_Hessian_computed`, root, score, response and reanalysis claims remain false. The root condition only produces `root_pending_audit`; it does not certify a stationary point.

## Multi-step refusal and commit accounting

Each iteration's endpoint is independently remeasured, then source/input/runtime/deadline closure gates its commit. A later refusal or timeout keeps all prior committed iterations and the last committed `current_control/current_state`; only the active uncommitted trial is marked refused. Started and completed HVP counts are separate, and the durable started callback runs after deadline/cap checks immediately before the JVP. Direction refusals use the same closure check as other outcomes. The final receipt performs another closure/deadline check. The parent verifies iteration counts and that the final accepted control matches the last commit.

These controls distinguish normal process completion from numerical direction/grid/budget refusal and from integrity failure. The line search has no minimum-J-decrease floor near the root; the actual J Armijo and actual Φ conditions remain the acceptance gates.

## Remaining coverage note

The parent reports the combined focused run as **18 passed, 18 known warnings, 2.69 s**, with zero error-level type diagnostics and an isolated AST refresh. I did not rerun those checks. I found no blocking issue in the reviewed paths. One nonblocking test gap remains: there is no direct regression that injects a source/input/runtime closure failure after a prior commit on the outer `StepRefusal` path and proves the child reports `integrity_refusal` while retaining the earlier committed state. The code does compute closure before classifying that refusal, and parent receipt checks also fail closed; the missing case is regression coverage, not a current result invalidator.

The planned attempt can establish at most three actual full-space dual-merit corrections from the committed tangent endpoint. It cannot establish full Hessian SPD, a full root, forecast skill, or adjoint/reanalysis eligibility without the separate required audits.
