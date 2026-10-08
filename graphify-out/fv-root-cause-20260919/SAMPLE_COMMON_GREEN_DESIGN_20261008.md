# PR261 follow-up — sampled common-direction design review

**GREEN design verdict: mathematically admissible as one bounded, predeclared original-J cost exploration.** It is not a Newton/root step, a Clarke subgradient method, a forced face crossing, or evidence of a limiting generalized gradient. This review reads the stored PR260/261 evidence and existing bounded search helper only; no FV, objective, gradient, or HVP was run here.

## Pinned state and direction

The reviewed face plan is SHA-256 `6b79d07c…`; its raw diagnostic is SHA-256 `ffa9473e…`. The base control is `2cdccade81a8cf218dd7cad97adf71f6675f8445481dd8f6394ec337ea8e8007`. Its cached PR260 resume record is `model_guided_resume_20261008_attempt1/step.json` (SHA-256 `66aa57c3…`); its current gradient exactly equals the last accepted gradient. The face raw sample and resume record name the same base. The stored base values are `J=0.0613476383231`, `Phi=0.0126732465030`, `||g||₂=0.159205819636`, and `||g||∞=0.088510892528`. The base branch receipt reports a strict 3,600-stage point; `Qy[4,3]≈−2.37504e−6` is nonzero. This supports a classical Hessian-vector product on the currently selected smooth branch after a fresh strict/base check; it does not extend Hessian validity across a branch transition.

From the stored inner face-probe gradients `g−=g(eta=−1e−6)` and `g+=g(eta=+1e−6)`, independently recomputed:

```text
Delta = g+ - g-
theta = clip(-g-·Delta / ||Delta||², 0, 1) = 0.288468024322864
v = g- + theta*Delta
d = -v
||v||₂ = ||d||₂ = 0.136797778407174
g_base·d = -0.0186907662532645
```

Thus `d` is a descent direction for the actual base gradient, although it is a **sampled two-gradient segment direction**, not `−g_base`: its angle from `−g_base` is about 30.8848°. With the common eta-zero normal `n0`, `n0·d=1.03227812882e−4` and `asin(|n0·d|/(||n0||||d||))=0.0480482°`, below the proposed 0.305° bound. This establishes near-tangency at the chosen base geometry; it does not guarantee that a trial crosses `Qy=0` or stays on the same face branch.

The first-order face crossing scale is approximately `−Qy/(n0·d)=0.02301`. The radius-only cap `R/||d||` for `R=0.05` is `0.36550`, so that cap alone would permit a crossing; a positive-curvature cap or dyadic backtracking may shorten it. The policy should not modify the direction or face coordinate to force a crossing.

## One-HVP cap and actual acceptance

At the verified base, form one fresh `Hd` for this exact `d`. Record `dᵀHd` and `g_baseᵀHd`; on the current smooth branch the latter is the directional derivative of `Phi=||g||²/2`. Use it as a reported diagnostic only. For finite, resolved positive `dᵀHd`, define `alpha_curv=−(g_base·d)/(dᵀHd)`; otherwise omit that cap and use the radius cap. Then

```text
alpha_init = min(1, R/||d||, alpha_curv)   if dᵀHd > 0
alpha_init = min(1, R/||d||)               otherwise
```

Refuse if the HVP or derived scalars are nonfinite, or if the checked base slope is no longer strictly negative. Do not reuse a Hessian product for another point/direction. A positive quadratic curvature cap is only a proposal scale: finite steps can cross nonsmooth branch boundaries and the local quadratic model is not an acceptance certificate.

The existing `bounded_original_j_search` evaluates a supplied `step` with `alpha_start=min(1,R/||step||)` and candidates `c+alpha_helper*step`, testing original-J Armijo with `g_base·step`. Pass `step=alpha_init*d`; then mathematically `alpha_start=1` because `alpha_init||d||≤R` (allowing a tiny roundoff reduction). The actual coefficient of `d` is `alpha_actual=alpha_init*alpha_helper`, where `alpha_helper=alpha_start*2^(−backtrack)`. Record both and their product; Armijo is correct because its internal slope is `g_base·step=alpha_init(g_base·d)`.

This is the predeclared **J-only cost exploration**. Require finite candidate objective/gradient/Phi for report integrity, original-J Armijo, and that candidate's own strict endpoint branch/margins plus final source/input/runtime closure. Phi is reported but is not a decrease gate. The current helper computes Phi and does not require it to decrease. Keep one guarded launch, one HVP maximum, no PCG, at most one committed candidate, 16 dyadic trials, 240-second internal/300-second outer limits, and the 1-GiB sampled RSS guard. A changed endpoint branch signature is reportable; no path certificate follows from it.

## Interpretation limits

The finite probes use different full control points even though 25 retained coordinates match. Their convex-hull direction is useful because it is near the selected face tangent and descends at the base, but it is not a one-sided limiting gradient or a Clarke generalized gradient. The fresh `Hd` supplies local curvature at the base only. The cost search must decide from actual full `J`, Armijo, and endpoint branch eligibility. Any accepted point is one bounded cost-improving step, not a root, stationarity result, face-constrained solution, or meteorological inference.

## Draft adapter review

The new `fv_point_3h_sample_common_search.py` draft has the intended execution order: load the pinned PR260 face samples and current `2cdccade…` model state; perform one current base branch check; recompute and compare base `J/g/Phi`; reject an unresolved non-descent `g_base·d` before starting HVP; persist HVP-started/completed counters; then use one `torch.func.jvp` for this exact direction. It passes `alpha_init*d` to the existing original-J helper, records `alpha_init*helper_alpha`, and separately measures the actual control displacement norm. The helper applies J Armijo plus the candidate's own strict endpoint branch. Phi is checked for finite consistency and reported but has no decrease gate. A tentative helper acceptance is explicitly marked `candidate_not_committed` if final fresh checks or source/input/runtime closure fail; only a candidate that passes the final J/g/Phi consistency and strict-branch recheck can be committed.

Before a guarded launch, fix two small numerical/reporting gaps in `initial_scale`: a nonfinite `dᵀHd` or nonfinite curvature-rounding budget currently falls through as “curvature unresolved” and silently uses the radius cap. Refuse nonfinite derived curvature values; finite nonpositive or roundoff-unresolved curvature may use the radius cap. Also record `g_baseᵀHd`, the current-branch directional derivative of Phi, as a finite diagnostic field only. Do not use it as an acceptance gate. For reviewability, retain the common-normal angle and `n₀ᵀd` alongside the sampled direction audit, making the verified 0.04805° versus 0.305° bound visible in the run receipt.
