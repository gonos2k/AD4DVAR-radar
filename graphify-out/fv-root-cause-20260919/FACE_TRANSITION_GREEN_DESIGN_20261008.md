# PR #260 face-transition diagnostic — GREEN design 검토

날짜: 2026-10-08
범위: accepted control `2cdccade…`와 고정 4×5 profile에서 외부 `Qy[4,3]` 주변 finite-face 진단을 설계했다. static chart/domain 및 소스 연산 구조만 확인했으며 objective, gradient, FV trajectory, HVP, PCG 또는 prediction은 평가하지 않았다. Transport source 변경은 제안하지 않는다.

## 고정점·face chart

현재 시작점은 resume 최종 accepted 26-vector `2cdccade81a8cf218dd7cad97adf71f6675f8445481dd8f6394ec337ea8e8007`이다. 저장 상태는 `J=0.0613476383231`, `Phi=0.0126732465030`, `||g||₂=0.159205819636`, `||g||∞=0.088510892528`이며 최대 성분은 flow latent `c[24]`다. `Qy[4,3]`는 zero-based index로 현재 `−2.3750409848e−6`; internal `Qy[3,2]`는 `+4.3161315908e−5`다. These are the correct PR259 endpoints; do not substitute prior 69c/e29 values from older runs.

The fixed streamfunction at vertices `(i,j)`, `i=0..4`, `j=0..5`, is

```text
psi = a0*i + a1*j + a2*i*j + a3*(j²−i²)/2 + a4*i*j²
a_k = L_k tanh(c[20+k]),  L=(0.11, 0.08, 0.07, 0.04, 0.03)
Qy[i,j] = −(psi[i,j+1] − psi[i,j])
```

At external-face cell indices `Qy[4,3]`, the `c[20]` basis `i` has zero x difference. Thus, holding all other controls fixed,

```text
Qstar(c) = −0.08 tanh(c[21]) −0.28 tanh(c[22])
           −0.14 tanh(c[23]) −0.84 tanh(c[24])
```

`c[25]` growth and the 20 field controls are not in this static face map. For a predeclared pivot `p∈{21,22,23,24}` with nonzero weight `w_p`, preserve the other 25 raw controls and set

```text
c[p] = atanh((eta − sum_{k != p} w_k tanh(c[k])) / w_p)
```

using these weights for this face. At the stored `2cdccade` point, static evaluation for requested `eta∈{−2e−6,−1e−6,0,+1e−6,+2e−6}` gives `max |atanh input|<0.061`; all four pivots remain well inside `(-1,1)`. A single pivot must be selected before sampling and used on both sides so chart changes do not change what “fixed retained controls” means. If comparing alternate pivots, treat each as a separate chart diagnostic. Do not clip the inverse argument or overwrite a face-flux array. Recompute the production face from the reconstructed full control and reject a requested-side mismatch beyond an FP64 scale-aware allowance.

The zero-based `Qy[4,3]` face is on the outer positive-y edge (vertex row 4; the saved grid's rows increase in physical +y), over zero-based x-cell column 3, in the saved 4×5 cell grid. Its static flux is negative at the resume point. The previous internal `Qy[3,2]` face is a different face and remains positive here.

## Profile, state, stage, and boundary fidelity

`fv_minmod_inverse_probe.make_spatial_case` defines the actual fixed profile: 4×5 cell echoes, 5×6 vertex streamfunction basis `(y,x,xy,(x²−y²)/2,x²y)`, limits above, `spacing_yx=(10,10)`, minmod reconstruction, 26 controls = 20 field + 5 flow + 1 growth, and 13 parameters. The PR259 10-minute long-horizon profile uses 90 substeps per interval (seed 9×10), two analysis intervals and 18 forecast intervals. The `FVPointResearchProblem` branch oracle therefore covers **360 analysis Euler stages + 3240 future Euler stages = 3600**.

`transport.observe_minmod_stages` at `src/advar/transport.py:47–73` forwards detached diagnostic copies. `_euler_minmod` calls the observer with its actual `q,qx,qy` immediately before `_muscl_slopes(q)` (`transport.py:204–260`); here `q` is the real 4×5 internal state for that Euler stage, not a ghost-padded reconstruction. Face arrays are `qx[4,6]`, `qy[5,5]`. Reconstruct choices by calling the existing `_muscl_slopes(q)` or reproducing its same zero-perimeter/inner minmod formula, and record `qx/qy` signs with stage index. Boundary edge traces are separate inputs used to construct `left_source/right_source/bottom_source/top_source`; they are not ghost cells inside `q`.

Use stage index `s` to partition analysis `s<360` and future `s≥360`. With 90 substeps per interval and 2 SSPRK Euler stages per substep, `substep=(s//2)%90`, Euler stage parity `s%2`; for analysis, interval index is `(s//2)//90`, and for forecast it is `2+(s−360)//180`. Boundary data for a substep comes from the corresponding frozen analysis schedule or future schedule (whose `boundary_start_interval` is 2), choosing its stage-0 or stage-1 edge tuple by parity. A trace record should preserve phase, interval, substep, Euler stage, real `q`, flux signs, reconstructed choices/margins, and boundary-source labels.

Growth is positive at this fixed control (`c[25]=0.1854143`); the profile uses `max_log_growth_per_step=log(1.35)` and 90 substeps, so the substep log growth is `ell=log(1.35)*tanh(c[25])/90≈6.12e−4`. For positive `ell`, Euler stage 0 uses the production helper `_scale_by_growth` on interior echo and start boundary traces; for `|ell|<0.125`, that helper uses `addcmul/expm1`, not plain `exp(ell)*value`. Euler stage 1 uses the actual stage-1 state and raw end boundary traces. For nonpositive growth, stage-0 state/start boundaries are raw before scaling stage 1. Do not synthesize edge-scaled states with a different floating-point operation.

Before claiming exactly 3600 observer stages, verify **every actual echo boundary edge at every substep/stage is scalable** under the positive-growth check `edge <= finfo(dtype).max / exp(ell)`. If any edge is not scalable, `finite_volume_step` makes an extra fallback Euler call for large edges and the observer receives additional events; refuse or explicitly tag those fallback events rather than silently shifting stage indices. The fixed-case source constructs boundary traces from the known initial echo schedule, but this review did not materialize those schedule tensors. The seed-state amplitude is not a certificate for the final frozen analysis/future boundaries; runtime must check every actual edge. No observer belongs inside AD: capture the branch/state trace in the existing no-grad primal branch pass, then evaluate full-objective AD gradients separately only at the four nonzero face coordinates.

## Sample set and claims

Use identical retained controls at `eta=−2e−6,−1e−6,+1e−6,+2e−6`; the `eta=0` sample evaluates primal full original `J` only. At each nonzero point use the original full-26 objective, all original priors, fixed observations/parameters/boundaries/time, and a fresh full gradient. Save strict pointwise branch status, analysis/future signatures or hashes, all per-stage choices/face signs, margin minima, control hashes, requested and production `Qstar`, and actual `J/Phi/g`. Do not evaluate AD, `Phi`, or strict branch eligibility at exactly zero flow.

For a single face event, suppose the two smooth branch extensions `J⁺(t,eta)` and `J⁻(t,eta)` are continuous on the common surface and differentiable tangentially. Then the boundary restrictions agree, so differentiating with respect to retained `t` gives

```text
Z0ᵀ(g⁺_0 − g⁻_0) = 0
```

and the limiting gradient jump is normal: `Pi_T(g⁺_0−g⁻_0)=0`, where `T=ker(nᵀ)` and `n=∇c Qstar`. In chart coordinates, `q_t=Z_etaᵀg`; the Euclidean tangent metric is `G=ZᵀZ`, so `||Pi_T g||²=q_tᵀG⁻¹q_t`. This is a necessary limit relation for one isolated continuous surface, not a conclusion from samples.

The merit jump identity is exact for any two gradients:

```text
Phi⁺ − Phi⁻ = 0.5 (g⁺−g⁻)ᵀ(g⁺+g⁻)
```

Even if the limiting gradient jump is purely normal, `Phi` need not be continuous; the normal component contributes. With multiple face/limiter events, `g⁺−g⁻` may also have tangential components. Branch traces show discrete selectors and event locations, but they do not contain reverse-mode adjoint weights and cannot by themselves determine the full-J gradient jump.

Finite cross-side probe pairs may report a **diagnostic candidate only**: project both gradients using one common surface normal `n0` (not each side's different normal), select/report the finite pair with smallest `||Pi_T,0(g⁺−g⁻)||`, and record `g⁻ᵀd`/`g⁺ᵀd` for a predeclared common tangent candidate `d`. Those values are finite-sample directional slopes, not asymptotic derivative bounds or proof that the minimizing pair approximates the one-sided limits. Without branch-preserving continuation and explicit Lipschitz/error bounds, do not call them a bound on the limiting gradient jump. Report the direct finite-sample `Delta Phi` alongside the exact identity; `eta=0` remains primal-only.

## Frozen draft source-review gates

The source draft still has a prelaunch namespace defect: plan loading now uses `model.tangent._pinned_path`, but the validation and final-closure paths still call bare `tangent._check_fixed_input` / `tangent._pinned_path` without a module-level `tangent` binding. Those references raise `NameError` during loader or closure unless changed to `model.tangent` (or an explicit alias is imported). Focused tests should exercise the real loader and closure.

The observer receives the actual internal `q, qx, qy` arrays and reconstructs the selected face donors correctly. The current detailed per-stage donor trace is limited to the 360 analysis stages; future-stage selected face fluxes feed only the aggregate summary, while the full branch signatures remain partitioned over analysis/future. Either record future per-stage target-face donor signs/details too, or state clearly that donor attribution is analysis-only and use the signatures for future branch-change evidence.

The in-progress `_gradient_jump` update now computes both inner and outer pairs from full ambient gradients, projects each against the common eta-zero normal, records each exact `DeltaPhi = 0.5 Delta_g·(g⁺+g⁻)` identity, and computes a clamped segment-minimizer `theta` for each **finite pair**. This is useful finite-pair evidence, not a limiting-gradient result. Keep the direction named as a finite-pair full-space convex-hull diagnostic; if a chart-covector average direction is retained, name it separately as a chart-metric tangent candidate. Do not label any finite probe as a limit bound. Do not compute gradient or strict branch eligibility at `eta=0`.

## Verdict and limits

The fixed profile, array dimensions, real-state observer source, stage counts, growth transformation, and arbitrary-pivot chart are compatible with a bounded diagnostic without changing transport code. The required scientific gate is to retain all branch/state evidence and preserve the distinction between a single-face hypothesis and simultaneous events. If a requested pivot is out of chart domain, actual production `Qstar` has the wrong sign, boundary schedule triggers fallback stages, or any side point fails its own strict oracle, preserve the partial receipt and stop inference. Even complete finite samples do not prove one-sided limits, tangent-gradient continuity, `Phi` continuity, a normal minimizer, single-face causality, or meteorological wind/precipitation structure.
