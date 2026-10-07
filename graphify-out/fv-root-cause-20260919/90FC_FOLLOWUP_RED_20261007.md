# 90fc follow-up continuation — RED prelaunch review

Date: 2026-10-07
Review type: read-only prelaunch review. No FV run, live FV HVP/PCG, forecast, adjoint, or reanalysis was started.

## Disposition

**GO to launch the one frozen, bounded follow-up plan.** The adapter starts from the exact accepted `90fc4555…` control, validates its final committed trial and fixed-input/runtime/source receipts, and delegates numerical work to the existing continuation kernel. The frozen plan retains the stated 720 s internal / 780 s external / 1 GiB limits, 90 cumulative HVP cap, three-iteration cap, and 40 PCG iterations per solve.

The 780 s legacy parent receipt is accepted only as the exact pinned historical case: raw child, parent, and resource hashes; original producer and test hashes; and the classification-fix receipt are all bound by the follow-up plan and checked in `load_base`. The original parent remains recorded as `failed`; the adapter requires the new classifier to evaluate that same saved resource as completed.

## Start-point and partial-solve safety

`_accepted_endpoint` requires exactly two completed accepted iterations, and the final accepted trial must match the final control hash/vector, J, Phi, full gradient, branch, and margins in `current_state`. The trial record does not contain `gradient_inf`; the adapter recomputes it from the recorded gradient when checking that state, avoiding reliance on a missing field.

The adapter passes only the validated accepted control and the archived f82c matrix as a block inverse preconditioner to the continuation runner. Each new iteration obtains its own current-point live HVP solve. The partial third solve is checked only for the expected budget-stop receipt shape; its unfinished direction and partial HVP sequence are not loaded or reused.

The plan fixes `max_iterations=3`, `max_hvp_calls=90`, `pcg_max_iterations=40`, `pcg_relative_tolerance=1e-10`, `max_candidates=16`, the existing 0.05 radius and both `1e-4` Armijo constants, root threshold `1e-10`, and the resource limits above. No cap was extended for this follow-up. The adapter now requires every source path from the prior frozen plan, plus the new adapter and tests; a regression removes `src/advar/physics.py` and confirms that preflight rejects the incomplete source map.

## Frozen-plan audit

The final frozen plan is `graphify-out/fv-root-cause-20260919/90FC_DUAL_MERIT_FOLLOWUP_PLAN_20261007.json`, SHA-256 `1c3cc494667ca6f9a3a0f57d714f78479d18249a44c1b3a397223484bd7f8067`. I independently checked every listed file digest and path: **98 source entries and 19 archive entries, with no missing files, digest mismatches, or path escapes**. The plan includes all 96 source names from the previous frozen continuation plan. The eight paths present only in the old raw source inventory are pinned as archives in the new plan. The plan timestamp follows the recorded focused tests and type check.

## Saved verification evidence

- `90FC_FOLLOWUP_TESTS_20261007.log`: **51 passed, 18 existing TorchScript deprecation warnings, 2.75 s**. This refreshed run includes rejection of a plan that omits an inherited physics source. The warnings are from the existing HVP Newton-step test module.
- `90FC_FOLLOWUP_TYPES_20261007.log`: **0 errors, warnings, or notes**.
- The final plan passed the adapter's source/receipt loader according to the root-run preflight record; the plan path map and all 117 listed file digests were independently checked in this review.
- `90FC_FOLLOWUP_GRAPHIFY_20261007.json`: cached structure consulted; four changed Python files refreshed to **66 nodes / 233 edges**. I confirmed its four recorded SHA-256 values against the current files and confirmed the shared graph and report hashes are unchanged.
- `git diff --check`: clean at review time.
- `PR252_ROOT_PR253_INTEGRATION_20261007.json`: PR #253 merged at `30cabf339231ee9ce3c66db4a3ec5d7dea3da50b`; UI check succeeded while CPU and Wheel/CLI CI were skipped. This is integration metadata, not live FV verification.

I read these saved records but did not rerun the tests or type check. The frozen plan is ready for the single FV launch under the same caps specified above. This review makes no claim that the follow-up will converge, reach an eligible stationary point, pass curvature/adjoint/reanalysis audit, or improve forecast skill.
