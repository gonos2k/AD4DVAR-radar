# Dual-merit normality-correction review — 2026-10-06

**Disposition: GO for the bounded prelaunch artifact.** The cached direction supports the two descent gates; the current source, frozen inputs, and focused checks are consistent. This is not approval of any numerical outcome: no dual-merit FV run or new HVP/PCG computation was performed.

## Cached direction and base point

The relevant immutable run record is `a35_live_hvp_newton_20261006_attempt1/step.json`, pinned by `NEWA35_LIVE_HVP_NEWTON_PLAN_20261006.json`. It identifies the base a35 control with SHA-256 `a35e2f6ef000cca4b593a3e5ed49f1c3e94b951450f2f235902a50e1dde6647c`, fixed parameter SHA-256 `8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed`, and the saved accepted endpoint with SHA-256 `38901e3a4c2ee97c8083b43e01e335f6328c3410982118b4ba891023468c471d`.

From the cached full base gradient, `||g||² = 2.225947234183745`, so `Phi = 0.5 ||g||² = 1.1129736170918725`. The 26-component saved direction has `gᵀs = -0.04171826501708994`; with `alpha = 0.17287763958236457`, `alpha ||s||₂ = 0.05`. The saved true linear-system residual is `8.214105073635838e-11` relative and its dot product with `g` is `7.9190891e-11`.

For `Phi(x)=0.5||∇J(x)||²`, its directional derivative is `D Phi[s] = gᵀ Hs`. With the solved system `Hs = -g + r`, this is `-||g||² + gᵀr = -2.225947234104554`, which is strictly negative by a wide margin. Thus this direction is locally descending for both `J` and `Phi` at the cached point, subject to the fixed-branch differentiability represented by the live operator and the finite residual.

## Why both acceptance conditions matter

The historical first candidate passed the original-J-only Armijo test: `J` fell from `0.07161498269916244` to `0.06681574666444642`. Its actual `Phi` rose from `1.1129736170918725` to `1.8573840876146208`. Under the proposed `cJ=cPhi=1e-4` dual Armijo policy, the first candidate must therefore be rejected: the Phi bound is about `1.11293515`, far below its actual value. That historic J-only acceptance remains a valid record under its original policy; the dual-merit policy is a new acceptance rule and must label it rejected under that rule.

The first-order Phi slope supports the proposed gate, but it does not guarantee that any of the 16 dyadic trial points will pass: finite steps can cross branch boundaries and the previous step already showed poor gradient linearization. A no-candidate outcome after the fixed search budget is an acceptable bounded result. It would not falsify the negative derivative or establish that no smaller mathematical step can work.

## Source review criteria

The inspected source and tests meet these mathematical and policy conditions:

- Uses the pinned a35 control, fixed parameters, observations, prior, objective, and cached `s`, `Hs`, and residual; it performs no new HVP/PCG solve.
- Forms `D Phi` from `gᵀHs` (equivalently `-||g||² + gᵀr`) and forms `D J=gᵀs`; finite residual error is checked consistently with the saved solve tolerance.
- Evaluates the actual full candidate gradient and actual `Phi=0.5||g_candidate||²` at every trial, plus actual `J`; a linearized or partial gradient must not be used as the acceptance value.
- Requires both actual Armijo inequalities with `cJ=cPhi=1e-4`, rejects nonfinite objectives/gradient components and strict-branch failures, and uses only the specified 16 dyadic trials from the pinned initial alpha/radius.
- Keeps one guarded launch and the existing 300 s outer / 240 s internal / 1 GiB sampled RSS limits; preserves source and fixed-input identity checks. The resource guard is sampled RSS evidence, not a hard allocation cap.
- Reports no global SPD, branch-path, stationarity, root, response, score, or physical-validity claim from this one search.

## Implementation review

The source uses cached `s`, `Hs`, and residual without new HVP/PCG calls. `_direction_audit` validates the solve residual and `Hs+g=r`; candidate acceptance uses fresh full gradients, actual `J`, actual `Phi`, both Armijo bounds, and complete strict endpoint margins. It starts at `min(1, 0.05/||s||)` and halves for the remaining 15 candidates.

Two initial RED concerns are closed in the final source. Candidate acceptance stays in a local `candidate_update` until the postprocessing, deadline, source snapshot, and fixed-input checks pass. Exceptions from objective, gradient, or branch callbacks preserve an active trial row with its alpha/control and failure phase before propagating. The tests include regressions for candidate noncommit and callback-error persistence.

The pinned raw direction control is the start point; it is hashed against the saved record and checked against current input identity. The factory seed's different control is not substituted. Parameter identity is tied through the plan and compared to the current fixed parameters. The child report must parse as a JSON object. These checks preserve the original J, prior, and observations and keep the historical 389 J-only endpoint as separate evidence.

The final plan SHA is `62e291b8b683ef1480f60ae302e562c7313c2a545a3cb6517fd53f3b294220c0`. I independently recomputed every source and archive digest: all 94 source pins and all 8 archive pins match current bytes. The required producer Newton plan, original step/parent/resource, and f82c curvature chain are included. The plan's `control_sha256`, `parameters_sha256`, direction-source SHA, and policy match the cached a35 record and source constants. The loader conditions are satisfied by the inspected code and verified key sets; no FV/HVP execution occurred.

The saved focused log reports 24 tests passed in 1.56 seconds, with 18 existing TorchScript deprecation warnings from the HVP Newton-step tests. The saved type log reports zero errors, warnings, or notes. Both logs postdate the final source edits; I did not rerun them. The new regression checks that malformed branch output preserves the active candidate receipt. The AST and isolated Graphify records match the final module/test hashes, and the shared structural graph/report stayed at the recorded prior hashes.

Final code review confirms the fixes raised in the initial review: an accepted candidate is held in `candidate_update` until model/physical diagnostics, the deadline check, source snapshot, and fixed-input closure pass; otherwise it is marked `candidate_not_committed`. The true a35 control comes from the pinned raw direction record, is hashed against that record, and is checked against reconstructed input identity; the different factory seed control is not substituted. Parameter SHA is bound through the plan and checked against the fixed factory parameters. Child JSON must be a dictionary. Callback exceptions persist their active alpha/control/status before propagating. These paths close the RED review's earlier result-record findings, with a mock regression for post-candidate refusal and a callback-error persistence regression.

The supported claim remains narrow: a source/input-pinned, bounded one-direction dual-merit search is ready for one guarded prelaunch. The first historical `389...` endpoint remains a J-only acceptance and is expected to fail the new Phi gate. Sixteen candidates are not guaranteed to succeed. The new mode establishes no stationary point, global SPD property, branch-path certification, response, score, or physical/weather validation. I reviewed saved test/type evidence but did not run tests or FV/HVP work.
