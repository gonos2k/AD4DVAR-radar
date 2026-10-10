# RED final audit: fixed GN local candidate commit — 2026-10-10

## Disposition

The saved run closes one fixed-alpha candidate as one optimizer commit, then builds fresh coupled-GN readiness at the committed control. I found no actionable implementation or receipt inconsistency. The readiness result does not include a second candidate, convergence, or a stationary/minimum claim.

## Candidate and independent closure

The frozen plan is `11e4eff337084ba9a9602b477854349969af8306926911fa49c4ab1e8e869b6e`. The archive records a lossless round trip from the 7,694,618-byte raw child to the 105,706-byte gzip; the raw SHA is `0f2892badb2eeebb9ae8d11ff3f3898a7166d68e13e0279e3a3c9625c8c1ff00` and gzip SHA is `4e59a33c63b16f0b0c8d62c1bccdd9f46b554427bcf8ec524bb36c802d579a94`.

At base control `2ec34eee…`, the run freshly checked J, both side gradients, minimum-mixture theta, strict branch traces, face, input, and runtime before reusing the two archived same-point HVP vectors. The fixed alpha was `5.574488376651601e-6`; its control hash `311f27d095be4976ca6131b83b6c4f3d46bc404d4f58921800b5600e38ec0652` matches the authorized saved sample. Both candidate and independent P2 repeat retain the sample’s side trace signatures. Candidate and repeat side gradients match exactly in the saved arrays.

Native J decreased from `0.06119249248090723` to `0.06119247898831076` (`0.00002205%`), and minimum mixed-gradient F² decreased from `0.00447848922842147` to `0.004478429816264897` (`0.00132661%`). Both original Armijo checks pass. The candidate theta is `0.483079578890406`; P2 confirms the fresh theta, J, merit, side objectives and gradients, face, branch pair, traces, source, fixed input, runtime, and deadline.

The endpoint branch signatures match the base, but the minimum normalized limiter gap is `9.075476315907308e-9`, about `81.6%` below the base gap. The saved local-window sample at twice this alpha changed a selector. These endpoint checks do not certify a smooth path between base and candidate; no such claim is needed for the pointwise acceptance.

## Postcommit readiness and resources

At committed control `311f27d095…`, the saved readiness evidence contains all 24 distinct side/row VJPs for rows 0–11 on each side, one 12-dimensional Cholesky-positive dense solve, and two fresh selected-face HVPs. Row records bind to the committed control/theta. The solve direction matches the published readiness direction exactly; both HVP start and completion labels bind to that same control, direction hash, theta, operator, and `robust_gn_coupled` model. The solve residual `2.5142361782388313e-15` is below its `2.1541996557120295e-12` budget.

Source-before/after receipts match; fixed-input, runtime, and deadline flags are true. The parent reports exit 0 in `56.536 s`, with sampled peak RSS `1,328,414,720` bytes under the 2-GiB sampled limit and no resource termination. `candidate_count` and `optimizer_steps_after_commit` are both zero. `root_claim`, `minimum_claim`, `score_claim`, and `response_claim` are false.

## Limits and follow-up

This is one small accepted local decrease followed by a ready direction/model at the new point. Any next candidate needs a separate continuation using that point’s fresh base identity and branch checks; this run does not authorize or assess a second step. It does not establish convergence, an optimum, response skill, or forecast value. The saved focused suite reports 18 passed with 18 existing warnings; basedpyright reports zero errors, warnings, and notes. Those checks are separate from the production run evidence above.
