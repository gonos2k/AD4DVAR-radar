# 90fc bounded dual-merit follow-up — prelaunch GREEN review

Date: 2026-10-07
Disposition: **GO for one guarded run using the frozen plan below.** This is a prelaunch review; no FV trajectory, new HVP/PCG solve, forecast, adjoint, or reanalysis was run.

## Exact start point and archived lineage

The adapter `examples/weather_scenarios/fv_point_3h_90fc_followup.py` uses the raw child receipt from `e0b_repeat_dual_20261006_attempt1/step.json`. It validates exactly two completed accepted iterations and selects the accepted trial whose control hash matches the final committed endpoint. The full control hash is `90fc45552d068ae4f1b83eb54ab095362f223dc10b884695ccdbe1e63c9825fc`; parameters hash is `8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed`. The preceding point is separately identified as `e0b04a…` in the archived plan, not used as the new start.

The historical raw child receipt says `phase=finished`, `execution_status=completed`, `numerical_status=budget_refusal`, two optimizer steps, and 62 completed HVP calls. Its partial third solve remains recorded with status `linear_solve`; the adapter uses neither that incomplete solve nor its direction. The old parent receipt remains `failed` as originally written. The pinned classification-fix record and unchanged resource receipt establish that the child exited 0 and the new 780-second classifier reads the same resource as `completed`. The implementation preserves this distinction and does not rewrite historical evidence.

The adapter also validates the prior plan and classification record, fixed parameters/runtime, source/input integrity, accepted endpoint fields against the last trial and `current_state`, and the complete archived f82c curvature/checkpoint receipts. The 26×26 f82c SPD matrix is reconstructed only as a block inverse preconditioner. Each new direction and true residual use the continuation’s live current-point HVP operator.

## Reused kernel and fixed execution profile

The follow-up adapter delegates to the existing `continuation.run` through its `base_loader`/`base_path` interface and launches its module through the shared guarded `main` callback. It does not introduce a second optimizer or a new fallback. The accepted endpoint is freshly checked for J/g/Phi and strict branch before iteration begins. Subsequent candidate processing retains the same original-J and Phi Armijo tests, strict branch requirements, diagnostic/integrity closure, cooperative deadline checks, and pending-only root status.

The frozen plan equals `continuation.policy_dict()` and keeps the previous limits unchanged: one guarded launch; 720 seconds internal, 780 seconds outer, 1 GiB sampled RSS; at most 3 iterations; 90 cumulative HVP calls including PCG products and true-residual Hs; at most 40 PCG iterations per solve; PCG relative tolerance `1e-10`; 16 line-search candidates; radius `0.05`; both Armijo constants `1e-4`; root infinity norm threshold `1e-10`. Three complete solves are not promised under those limits. Any resource, solve, branch, input, or integrity refusal preserves already committed points and stops without extending a cap.

The frozen plan path is `graphify-out/fv-root-cause-20260919/90FC_DUAL_MERIT_FOLLOWUP_PLAN_20261007.json`, SHA-256 `1c3cc494667ca6f9a3a0f57d714f78479d18249a44c1b3a397223484bd7f8067`. It contains 98 source pins and 19 archive pins. Every listed file exists and its current SHA-256 matches the plan. The plan inherits the full prior source set, adds the follow-up adapter and focused test, and carries the raw child/parent/resource, prior plan, classification fix, producer/test, and curvature/checkpoint receipt pins.

I called the adapter's metadata-only `load_base(plan, plan_sha)` as a preflight check. It passed, returned the exact 90fc endpoint and matching fixed parameters, classified the same saved resource as completed while preserving parent status `failed`, reconstructed a 26×26 preconditioner, and reported that the old partial solve is not reused. It did not run FV or HVP work.

## Verification evidence and pins

The recorded affected test log reports **51 passed, 18 warnings, 2.75 seconds**. All 18 are existing TorchScript deprecation warnings from `tests/test_fv_point_3h_hvp_newton_step.py`. The recorded type check reports **0 errors, 0 warnings, 0 notes**. The real archived endpoint regression confirms the final accepted trial selection and keeps the legacy parent’s `failed` value intact. No FV execution was part of these checks.

Current source/test hashes match the frozen plan:

- `examples/weather_scenarios/fv_point_3h_90fc_followup.py`: `81bd588107705733d01339dc7417642016893d321da93a0d37ca0c272d79dd12`
- `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py`: `8118ebce815969d07b49e129e5fccecf811321b42f0ef48273089171a4fc29ac`
- `tests/test_fv_point_3h_90fc_followup.py`: `d99e3181b78695700f6e87904bf8fe34d64c07bd925c78378afd1d7824ae9ec2`
- `tests/test_fv_point_3h_dual_merit_continuation.py`: `4051ea8ee8962ea139e013284ac398b708f61eee09b8d8c954b0b7cccd87dd9e`

The merged PR #253 integration record identifies main `30cabf339231ee9ce3c66db4a3ec5d7dea3da50b`, PR head `b56a9f17c2db2667993e7415dc3a67e15f7e1c9a`, and equal PR/main trees. CI reported UI success with CPU and Wheel/CLI skipped; the focused local suite and type log above provide the relevant local evidence.

## Review boundary

This is approval for one bounded follow-up launch from the exact 90fc committed point under the unchanged plan. It does not assert that the next direction will converge, that a qualifying stationary point exists, or that the point passes curvature, adjoint, nonlinear reanalysis, or independent forecast-skill validation. Report each newly committed step and any refusal separately; the prior two steps and partial third solve remain historical inputs only.

## Final post-Graphify pin recheck

After the final isolated structural refresh, I rechecked the plan and all four code/test pins. The plan remains `1c3cc494667ca6f9a3a0f57d714f78479d18249a44c1b3a397223484bd7f8067`; the adapter, shared continuation kernel, and both tests still match the hashes listed above. `90FC_FOLLOWUP_GRAPHIFY_20261007.json` reports 66 nodes and 233 edges across the four files, with shared `graph.json` and `GRAPH_REPORT.md` hashes unchanged before and after. The frozen plan has already passed the metadata-only loader check. No code, test, or plan changes followed that check.
