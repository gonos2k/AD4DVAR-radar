# d31 strict/inexact comparison — final RED prelaunch review

Date: 2026-10-07
Scope: source and frozen-plan review only. No FV/HVP/PCG run and no test rerun were performed.

## Disposition

**No blocking routing, selection, plan-pin, or runtime/data-identity gap remains in the reviewed snapshot.** The earlier pre-arm budget classification issue is closed: a run with no verified arm can preserve a budget stop using validated base metadata, marks `base_metadata_only=true`, and leaves `selected_arm=null`. It does not manufacture an arm result. A normal-arm budget refusal leaves `comparison_complete=false`.

The frozen comparison plan SHA is `d69ce3000f1605bed67ea67a4c6751bbbdf2ab142fa47bd24418c368d9220d41`. I independently checked its 102 source pins and 24 archive pins against the current files; all matched. The plan binds the d31 endpoint hash `d31ecd393002a1e098c939838ce22e9a7526c1b7f0f426effbb3f3a408ccdeeb`, its producing 90FC plan, the current continuation/loader/comparison code, tests, and f82c preconditioner evidence.

## Routing and selection

- `plan.base_step` names the pinned completed PR254 receipt. The generic loader checks raw/parent/resource linkage, normally completed execution, closed accepted iterations, equality between final accepted trial and `current_state`, finite FP64 control/full gradient, fixed parameter hashes, unchanged source/runtime records, and the producing-plan relationship. It rejects provisional and partial-solve state.
- The producing 90FC run is the base. Its older failed legacy ancestor remains separately hash-pinned by the producing plan; the generic loader does not accept arbitrary failed parents.
- Strict then inexact each reload the same base and create a fresh control from d31. The direction and partial PCG state are never passed between arms. Both receive the same shared counter object and absolute deadline; the total HVP cap is 90.
- The top-level result now distinguishes `planned_selected_arm="inexact"` from `selected_arm`, which stays null unless the inexact arm has one committed step and both source and fixed-input closure. The strict endpoint is never used as the inexact start or selected endpoint.
- The outer budget/no-arm path now validates source pins before and after and confirms base receipt metadata plus live runtime. It sets `base_metadata_only=true` when no arm receipt was verified. This accurately means the base metadata closed; it does not claim that either arm ran.

## Numerical contract and experiment meaning

The strict arm remains at relative residual tolerance `1e-10`. The inexact arm uses

\[
\eta_k=\min(10^{-3},\max(10^{-10},0.1\|g_k\|_\infty)),\qquad
\|H_k s_k+g_k\|_2\leq\eta_k\|g_k\|_2.
\]

The comparison policy is explicit that this is a strict-first shared-remaining-budget experiment: it offers no equal per-arm reserve, makes no comparison claim if either arm is unfinished, and supports no global speedup claim from one pair. If strict uses the budget before inexact completes, the recorded result is an incomplete budget stop, not evidence against inexact Newton.

The inspected continuation keeps fresh current-point HVPs and an independently recomputed true residual, the rounded descent checks for `g·s` and `g·Hs`, refusal for unresolved observed direction curvature, and the existing actual J/Phi Armijo, full endpoint branch/margin, closure, deadline, and HVP gates. The full-gradient root threshold remains `1e-10`. The forcing rule is labeled heuristic, not Eisenstat–Walker; it does not lower final-root, curvature, adjoint, reanalysis, or forecast criteria.

## Remaining limits

This review establishes launch-plan/source consistency, not runtime success or numerical benefit. The test/typecheck evidence reported by the parent was not rerun here. Strict-first resource allocation may prevent a two-arm comparison, and one completed pair would still be a single local cost-versus-progress observation at d31. A changed branch signature at a candidate remains endpoint evidence only; it does not establish smoothness along the step. No convergence, final curvature, response, adjoint/reanalysis, or forecast claim follows from this prelaunch review.

## RED decision

The reviewed code and frozen plan are ready for the declared bounded experiment from a routing and integrity perspective. Preserve both arm receipts and the shared-budget status. Report numerical outcomes only after the guarded run; do not describe a one-arm result as a paired comparison or infer general speedup from this single start point.
