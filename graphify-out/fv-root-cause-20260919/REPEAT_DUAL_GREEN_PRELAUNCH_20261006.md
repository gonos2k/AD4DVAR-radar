# GREEN prelaunch review: bounded repeated dual-merit correction

## Disposition

**GO for the single frozen guarded launch only.** The reviewed source and test pins match the frozen plan, and the loop uses a fresh current-point Hessian-vector operator for each new direction. The archived f82c Hessian is scoped to construction of an SPD block preconditioner. The run has explicit dual Armijo and endpoint branch gates, preserves committed controls across later refusals, and keeps numerical stationarity separate from final eligibility.

Reviewed plan: `graphify-out/fv-root-cause-20260919/E0B_DUAL_MERIT_CONTINUATION_PLAN_20261006.json`, SHA-256 `ee75ca45b03495b055023a984457c4aab73eb76c646c1c17bf8760d2e53f80d9`. Its 96 source files and 12 archived inputs all exist and match their recorded SHA-256 values. The plan fixes the start control to `e0b04a4dc019cea2664af811e7cd956453dd6b34c9d2f7a35bff171fdc57132a` and parameters to the preceding accepted receipt. The loader checks the dual-merit parent/child/resource chain, the f82c curvature/checkpoint/parent/resource chain, and the cached objective sources; the runner repeats the source/plan pin comparison after preflight to close the capture gap.

The bounds match the plan: up to 3 iterations, 90 total started live HVPs, at most 40 PCG iterations in each solve, 16 dyadic candidates per iteration, 0.05 per-step L2 radius, true residual `rtol=1e-10`, 720 seconds internal, 780 seconds outer, and a sampled RSS ceiling of 1 GiB under one guarded launch. The shared HVP counter includes PCG operator actions, PCG residual-restart actions, and the independently recomputed post-solve `Hs`. The 90-HVP cap can stop the run before a third solve or before three accepted steps; neither is promised.

## Numerical and state-flow checks

At each committed control the loop recomputes original `J`, the full 26-component gradient, `Φ=||g||₂²/2`, and the complete 3,600-stage strict branch/margin gate. Its new operator is `H_k v = J_cc(c_k,p)v`, passed to the existing `matrix_free.pcg` with `-g`, the old f82c block inverse preconditioner, `rtol=1e-10`, and `max_iterations=40`. It then evaluates a fresh `H_k s` for the true residual and both directional slopes. The residual gate, finite checks, positive observed direction curvature check, and strict negativity checks on both `gᵀs` and `gᵀH_ks` run before candidate search.

For `r=H_ks+g`, `gᵀH_ks=-||g||₂²+gᵀr`; thus the `1e-10` relative residual supports local `Φ` descent at a nonzero gradient. `J` descent is checked separately. The actual candidate evaluations apply both existing `1e-4` Armijo tests and the candidate's complete strict endpoint branch/margin gate. Direction and `H_ks` remain fixed within one search grid and are rebuilt at the next accepted point. Stagnant or numerically zero decreases are refused before commit. Candidate branch partition, physical diagnostics, input/source/runtime closure, and deadline checks complete before the control and iteration receipt are committed; if a later iteration fails, earlier committed records remain and only the current provisional candidate is cleared.

The review found no automatic J-only fallback, curvature shift, prior/parameter change, stale-Hessian operator, or path-branch claim. The raw record fixes `full_root_claim=false`, `eligible_stationary_point=false`, accepted-point curvature as not computed, and response/forecast score as not performed. Reaching `||g||∞≤1e-10` is recorded as `root_pending_audit`. `finaleligible` remains outside this run and requires separate curvature, branch, original-Hessian adjoint/response, and reanalysis qualification. Strict endpoint checks do not certify a smooth path between endpoints; `Φ` reduction does not certify a local minimum or weather skill.

One bounded accounting limit is explicit: if PCG exits before returning a `PCGResult`, the failed solve's exact recurrence iteration count is recorded as `not_recorded`; its solve-start count and all started/completed HVP products remain recorded, and no solve can exceed the pinned 40-iteration cap. This does not turn an incomplete solve into a successful one or affect the hard HVP limit.

## Verification reviewed

The saved focused log reports **24 passed**, 18 existing PyTorch deprecation warnings, in 4.13 seconds. The saved type check reports **0 errors, 0 warnings, 0 notes**. The isolated two-file Graphify receipt contains 37 nodes and 155 edges; the shared `graphify-out/graph.json` and `GRAPH_REPORT.md` hashes are unchanged. The root's metadata-only `load_base` preflight passed for the frozen plan.

This is a source, plan, test-log, type-log, and receipt-chain review. It did not execute the FV trajectory, HVP, PCG, forecast, adjoint, or reanalysis. The launch result must be reported strictly from the child, parent, and resource receipts after the one guarded run; a completed execution or a root-pending numerical status does not establish final eligibility or forecast improvement.

## Locations checked

- `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py`: fresh current-point direction, HVP accounting, dual search call, provisional rollback, commit closure, outer guard, and final source/input/runtime checks.
- `examples/weather_scenarios/fv_point_3h_dual_merit_step.py`: reused within-iteration dual-merit search and strict endpoint gate.
- `src/advar/matrix_free.py`: PCG convergence, max-iteration limit, and true-residual/restart operator calls.
- `tests/test_fv_point_3h_dual_merit_continuation.py`: repeated-point relinearization, root-pending label, cumulative HVP refusal, curvature refusal, later-commit rollback, cooperative deadline, and malformed child receipt cases.
- `graphify-out/fv-root-cause-20260919/REPEAT_DUAL_TESTS_20261006.log`, `REPEAT_DUAL_TYPES_20261006.log`, `REPEAT_DUAL_GRAPHIFY_20261006.json`, and frozen plan listed above.
