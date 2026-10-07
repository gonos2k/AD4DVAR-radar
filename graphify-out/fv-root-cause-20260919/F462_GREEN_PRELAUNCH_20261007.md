# f462 inexact continuation — GREEN prelaunch review

Date: 2026-10-07
Disposition: **GO for the one declared bounded continuation from f462.** This is a read-only prelaunch review plus metadata-only loader validation. No FV, HVP, PCG, forecast, adjoint, or reanalysis was run here.

## Start point and provenance

The plan is `F462_INEXACT_CONTINUATION_PLAN_20261007.json`, SHA-256 `499e89862d86c8e4258e5199194674714c9001c59fca00e5436c935d98a3e33b`. It selects the comparison top receipt `d31_inexact_comparison_20261007_attempt1/step.json` and its adjacent `step.run.json` and `step.resource.json`; it does not point at the strict arm or fabricate arm-level sidecars. The plan's base and expected-base control hashes both identify selected inexact control `f462a4961634efa253e3757f7de44304386135eb3a229f3729a61eb911e6eed9`. The comparison's original starting point remains separately recorded as d31, `d31ecd393002a1e098c939838ce22e9a7526c1b7f0f426effbb3f3a408ccdeeb`.

I recomputed the frozen plan digest and every listed source/archive digest: **104 source pins and 28 archive pins are present and match**. The pinned comparison plan has 102 source files; all 102 match the corresponding entries in the comparison raw's 121-entry before/after source map. The comparison raw, parent, and resource sidecars are pinned by the new plan.

The top receipt says `phase=finished`, `execution_status=completed`, `numerical_status=paired_comparison_completed`, `comparison_complete=true`, and `selected_arm=planned_selected_arm=inexact`. Its accepted control is f462. The parent child hash matches the raw; parent iteration count is 1 and its 53 HVP total matches the top raw. The resource receipt records exit 0 in 695.530 s, sampled peak RSS 373,604,352 bytes, and no signal, monitor error, or resource termination. The nested inexact arm and top endpoint close through the accepted control, trial/iteration, current state, and input-after identity checks. The strict arm remains comparison context; it is not a fallback.

The source comparison's 53 HVP count is shared across strict and inexact. The nested selected arm records 22 HVP for its solve. The fresh continuation passes neither count as `shared_counts`; the core initializes a new invocation's counters at zero. The prior counts remain provenance, not budget already spent in this new launch.

## Loader and execution policy

The wrapper calls the plan-driven common loader and explicitly checks the inexact continuation policy. For a comparison source it requires the complete comparison and inexact selector, then closes the inexact endpoint against the top-level chosen state. A later normal inexact continuation can serve as a base through its own pinned plan. The plan must match the new runner and its `policy_dict("inexact", 3)`; the base case is selected by the plan rather than a fallback to a historical strict or d31 endpoint.

I ran only the metadata loader, `.venv/bin/python ... _validated_loader(plan, plan_sha)`. It passed and returned the f462 hash, `linear_mode=inexact`, three-iteration policy, and the cached f82c preconditioner. The loader reconstructed and checked pinned receipts and cached curvature; this did not evaluate the FV objective, gradient, or HVP.

The declared new-run limits match the existing bounded continuation contract:

| Setting | Frozen value |
|---|---:|
| Accepted corrections | at most 3 |
| Total HVP | 90, including PCG products and independent true-residual `Hs` |
| PCG iterations per solve | at most 40 |
| Cooperative internal deadline | 720 s |
| Guarded wall limit | 780 s |
| Sampled RSS limit | 1 GiB |
| Guarded launches | 1 |

The wrapper fixes `linear_mode="inexact"`, passes no prior direction or shared counter/deadline, and lets each solve compute eta from the current full gradient norm. The rule remains `clip(0.1 * ||g||∞, [1e-10, 1e-3])`; f462 begins in the `1e-3` cap region. Each solve must pass the fresh current-point true-residual check against its eta. The cached f82c matrix remains a preconditioner only.

The current iteration code retains the dual actual J/Phi Armijo search (`c1=1e-4` for each), descent/curvature gates, strict endpoint branch/margin and input/source/runtime closure. The root-pending audit remains in place, and final stationarity stays `||g||∞ <= 1e-10`. These are still the original-J correction criteria; the inexact forcing value does not change final root, curvature, adjoint, or reanalysis tolerances.

## Verification records and limits

The saved focused suite reports **70 passed with 18 existing deprecation warnings**; saved type checking reports 0 errors, warnings, and notes. The incremental Graphify record covers four files (40 nodes, 107 edges) with shared graph/report hashes unchanged. These are inspected records, not reruns in this prelaunch review.

The design review's red findings are addressed by using the parent comparison receipt and its actual resource sidecar, and by enforcing `comparison_complete` plus selected-arm closure. The remaining launch interpretation is narrow: permission to run this one bounded inexact continuation from f462 under the pinned plan. The comparison endpoint has `||g||∞≈0.416619`, far above `1e-10`; neither this preflight nor a normally completed capped run would establish a stationary point. The stricter eta region below `||g||∞=1e-2`, current full curvature, final adjoint/reanalysis, and independent forecast score remain open.

## Evidence inspected

- `examples/weather_scenarios/fv_point_3h_inexact_continuation.py`
- `examples/weather_scenarios/fv_point_3h_committed_base.py`
- `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py`
- `graphify-out/fv-root-cause-20260919/F462_INEXACT_CONTINUATION_PLAN_20261007.json`
- `graphify-out/fv-root-cause-20260919/F462_TESTS_20261007.log`, `F462_TYPES_20261007.log`, and `F462_GRAPHIFY_20261007.json`
- `graphify-out/fv-root-cause-20260919/d31_inexact_comparison_20261007_attempt1/{step.json,step.run.json,step.resource.json}`
