# Final RED preflight: same-base tangent preconditioner comparison — 2026-10-09

## Disposition

No blocking math or transaction defect found in the current comparison source for the single planned guarded run. This is a source-and-plan preflight only; no seed reconstruction, FV observation, production gradient/HVP, guard, or test run was performed. The comparison plan is hash `be2886f2686cd04e71d3d244967da378bae9c22e39168c5f22bd85e7801169a4`, and its read-only base/plan loader passes with 131 source pins and 116 archive pins. The accepted base is exactly `33cb86ca…`, theta `0.3295008210283633`, J `0.06124426330405451`.

## Math and objective path

The profile gate in the adapter fixes the needed assumptions: 26 controls, 13 parameters, 3×4 all-detected point observations, no neural prior, identity control-prior residual, and zero field-smoothness weight. `problem.observation_residual` supplies the same quality-standardized and correlation-whitened rows used by the native pseudo-Huber objective. The adapter builds 12 scalar row pullbacks under each selected-face extension; 24 row VJP products are separate from four HVPs. It checks current side residual equality using the whitening/observation construction scale and projected row parity using ambient as well as tangent row norms before using the side blend. J is reconstructed from those rows plus the existing identity prior and zero smoothness term, then checked against both side objectives and the observed native J.

The robust weight `(delta/hypot(delta,r))**3` is the correct scalar `rho''` for the pseudo-Huber objective; it is distinct from the squared frozen IRLS weight. `diag(P I P)` is `projector.square() @ 1`, `diag(P Q_cc P)` is the matching contraction with diagonal tanh `Q_cc`, and the fixed ambient-prior floor `W_i=1/max(1,B_raw_i)` makes `0 < W_i <= 1`. The `-mu P Q_cc P` sign matches `L=J-mu Q`. This remains an approximate projected robust-GN search metric: it omits residual-Hessian terms and can vary with theta/extension. The source and design docs make no exact Hessian, condition number, minimum, or curvature-certification claim.

For positive W, `-P W P g_mix` lies in the tangent space and has negative mixture slope for a nonzero projected gradient. The common-face gradient-jump support and both one-sided slope gates remain in `tangent_model`. The baseline arm is the existing chart direction; the adapter verifies it equals `-P g_mix` to a scale-aware budget. The robust arm supplies one direction as both its HVP input and `direction_override`; the shared helper checks exact tensor equality before either HVP and validates tangent/chart reconstruction. Each arm then uses its own two fresh side HVPs, and history records base, theta, side, direction digest, and model name.

## Comparison and failure behavior

The shared bounded loop runs the baseline and robust-GN arm searches before selection. Each arm uses the existing first-passing-alpha search, capped at 16 candidates; it stops at its first passing alpha or grid exhaustion. The plan explicitly records that semantics. If “fully attempted” were instead intended to mean evaluating all 16 alphas after a pass, that would require a separately reviewed change; it is not the current plan.

The selector uses only actual accepted endpoint F², then native J within the frozen 128-epsilon tie, then baseline as deterministic final tie-break. Predicted F² and positive diagonal values do not add a new acceptance gate. Only after both arms finish does it run one fresh final closure on the selected proposal and commit at most that one endpoint. Row/HVP exceptions before selection leave the original `33cb86ca…` current/last-confirmed point untouched; an interrupted second arm cannot commit the completed first-arm proposal. The synthetic tests exercise both-arm selection and interruption.

The parent closure requires one comparison record with both named arms, no more than 16 trials per arm, exactly 24 completed row VJPs, exactly four completed HVPs split by direction model and side, matching direction digests, consistent final control/theta, and current source/input/runtime/deadline closure. It therefore cannot report one arm as a closed two-arm comparison if the other arm is incomplete.

## Remaining bounded risks

The new tests cover exact whitened residual reconstruction, mask grouping, pseudo-Huber tails, dense-vs-diagonal projected algebra, identity `W=I` compatibility, ambient-scaled row tolerance, direction/HVP mismatch refusal, the winner tie cascade, successful two-arm selection, incomplete-arm noncommit, and synthetic row/HVP budgets. The `W=I` regression establishes baseline compatibility only; there is no general `W=cI` endpoint-invariance claim or test. Such invariance does not hold in general because `delta_theta=-(nu·g_mix + nu·H_mix d)/(nu·jump)` contains a fixed gradient term, and alpha caps can change. The precondition design memo records this limit.

One low-severity coverage gap remains: the factory contains refusal paths for residual-pair and projected-row parity mismatch, while current tests exercise the parity budgets and matching rows but do not directly assert that a deliberately mismatched side row is rejected before HVP/candidate evaluation. The implementation visibly refuses on either mismatch; adding a small synthetic negative case would lock down that contract. This is not a reason to run the production comparison, and the gate fails closed.

The 600-second internal / 660-second outer / 1-GiB sampled limits are explicit, but prior resume runtime/RSS does not measure the added VJP row work or two candidate searches. Each row stores only a 26-vector, while a side's VJP retains its forward residual graph during 12 scalar pullbacks; actual peak resource use must be read from the single guarded run. No resource-fit prediction is made here.

The source/plan were still being finalized in this preflight. Root should rely on the frozen plan's source hashes and perform the final launch/result review after this handoff. No curvature comparison, theta-only cleanup, condition-number estimate, minimum, response, or forecast claim is part of this preflight.
