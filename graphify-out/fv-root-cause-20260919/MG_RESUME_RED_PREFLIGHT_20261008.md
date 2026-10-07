# Model-guided continuation resume — RED preflight

Date: 2026-10-08
Disposition: **GO for the single bounded resume launch under the frozen plan.** This is a read-only preflight; no FV, gradient, HVP, PCG, forecast, adjoint, reanalysis or tests were run by this reviewer.

## Plan and source integrity

The frozen resume plan is `MODEL_GUIDED_FULL_SPACE_PLAN_20261008.json`, SHA-256 `9fc5e046ca484737ebd040469990f10cc918fcd9dc245d0762e65f297151440c`. I independently hashed every declared file: all **111 source pins and 50 archive pins match**. The plan declares `model_guided_resume`, names the October 7 producer plan, and fixes the base control to the producer’s final accepted point `69c4a794730750c09c1bcddf542b66131f64f5bfec362e02ac269413a75d4992`.

The source-version bridge is narrow. The immutable source snapshot manifest `model_guided_source_20261007/manifest.json` pins exactly the prior model-guided module bytes (`67341f6d…`) and focused-test bytes (`852d8054…`) recorded in the producer’s raw `source_before` map. The resume plan permits current-version drift only for those two paths; all remaining original numerical/operator source pins and producer archive pins are retained. The recorded numerical AST audit reports identical imports, mathematical helpers and complete iteration loop between archived producer and resumed module. A fresh source-before/after snapshot covers the new invocation.

The October 7 producer plan/raw/parent/resource artifacts are pinned independently. The preflight correction record documents one rejected metadata-only plan validation caused by a loader expectation mismatch; it records zero guarded launches, zero HVPs and unchanged old numerical data. The corrected loader consumes `current_state.branch`, checks the last accepted trial/state and resumes only from `69c4a794…`.

## Endpoint, input and counter semantics

The producer ended with three accepted commits, three started/completed HVPs, no PCG, `max_iterations_completed`, and `candidate_committed=true`. Its `current_control`, current-state control hash, and last iteration’s accepted control/hash agree at 69c4. The parent child hash, numerical status and normal resource receipt are required to close before resume.

The producer’s `input_before.control_sha256` is `05889584…` and its `input_after.control_sha256` is `69c4a794…`; that change is the legitimate accumulated control update. The resume adapter compares `input_after` fixed fields (parameters, observations/archive identity and terminal truth) while binding the reconstructed current control to 69c4. It does not require the entire before/after identity object to be equal. It freshly evaluates J, full gradient, Phi and strict branch at 69c4 and compares them to the producer’s final current state before any new HVP.

The resumed invocation starts its own `iterations` list and HVP counters at zero. It can take at most three additional committed steps and three fresh current-point HVPs under a fresh 240 s / 300 s / sampled 1 GiB / one-launch budget. The producer’s earlier three commits and HVPs remain lineage fields and are not charged to or reused by the new call. Each new direction is recomputed as `d=-g`, and every new point gets a fresh HVP and actual J/Phi/branch checks. No archived HVP, alpha, or tangent direction is reused.

On timeout, direction refusal, grid refusal or integrity failure, the last already committed current control/state remains the resume result; only the active uncommitted candidate is refused. A source/input/runtime closure failure is an integrity refusal rather than a budget refusal. Reaching the (10^{-10}) gradient threshold only records `root_pending_audit` and does not certify a full root.

## Remaining scope

The resumed run uses the same full-space numerical kernel; its preserved branch transitions remain finite endpoint evidence, not a smooth-path certificate. Neither the original full root, current full-Hessian SPD, response/adjoint/reanalysis, nor independent future verification is implied. The prior 058/e29 controls remain historical; the resume starts only from the committed 69 endpoint.

The parent reports the final focused verification as **19 tests passed, 18 existing PyTorch warnings, 1.60 s**, zero error-level type diagnostics, and an isolated AST refresh. I did not rerun tests or the real loader. No model execution is supported by this preflight.
