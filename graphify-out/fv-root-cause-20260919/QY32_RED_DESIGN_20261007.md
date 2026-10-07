# Qy[3,2] two-sided boundary diagnostic — RED design

Date: 2026-10-07
Scope: bounded design review from the committed PR #256 endpoint. This note proposes one diagnostic; it does not report new model results. No FV, HVP, PCG, optimizer, forecast, adjoint, reanalysis, test suite, or shared Graphify job is authorized by this design.

## RED disposition

The proposed fixed-tangent chart is mathematically appropriate for distinguishing a smooth continuation through this face from a kink in the full objective, provided it is implemented as a coordinate reconstruction of the complete 26-vector and all probes are evaluated with the original problem. The exact-face point must be cost-only. At η=0, the production upwind/minmod branch can be nondifferentiable; an AD gradient there would depend on framework branch conventions and cannot stand in for either one-sided gradient.

This experiment can determine whether the current small-step behavior is consistent with a (J)-decreasing face crossing rejected by the raw-Φ gate. It cannot certify an active-face minimum, a root, positive tangent curvature, a causal role for this one face, or physical wind/rain behavior.

## What the existing evidence establishes

The PR #256 archive reports the same internal (Q_y[3,2]) approaching zero at three accepted controls:

| Control | (Q_y[3,2]) |
|---|---:|
| f462 start | (-6.777421930689864\times10^{-5}) |
| d139407b… | (-8.147753831233562\times10^{-6}) |
| e29c348d… | (-5.910612738066479\times10^{-7}) |

The two nearest larger line-search candidates crossed to positive flux, passed their own strict branch and (J) checks, and were rejected by Φ increase. Those archived candidate records do not give full fixed-tangent one-sided limits. The endpoint remains far from stationarity, so the observed event is a reason to diagnose the search, not evidence that the face is active at a solution.

The existing boundary arithmetic is in `F462_BOUNDARY_PROGRESS_20261007.json`; the run/provenance audit is `F462_RED_FINAL_20261007.md`; the saved-array arithmetic is `F462_RESULT_20261007.json` and `F462_SCIENTIFIC_CHECKS_20261007.py`.

## Fixed-tangent chart

Let the full original control be (c\in\mathbb R^{26}), with flow latents at indices 20–24 and growth at index 25. Set

\[
 t=(c_0,\ldots,c_{23},c_{25})\in\mathbb R^{25},\qquad \eta=Q_y[3,2].
\]

Hold every component of (t) at its value in the validated e29c348d… endpoint. For the pinned 4-by-5 basis and coefficient limits, reconstruct only the pivot latent by

\[
 c_{24}(t,\eta)=\operatorname{atanh}\!\left(
 \frac{-\eta-0.08\tanh c_{21}-0.21\tanh c_{22}-0.10\tanh c_{23}}{0.45}
 \right),
\]

and define Γ(t,η) by replacing component 24 with this value. The other 25 raw controls, including all field controls, the remaining flow control, and growth control, remain unchanged. The full 26-term original prior and original (J), observations, parameters, boundaries, time schedule, and FV/minmod settings remain in force. Do not zero a stored face-flux array or drop the pivot prior term.

### Chart domain and numerical risks

Write the atanh argument as (z(t,\eta)). Every probe requires finite (z) with (-1<z<1), finite (c_{24}), and finite production coefficients. Report (1-|z|) and (dc_{24}/d\eta=-1/[0.45(1-z^2)]); a small domain margin signals an ill-conditioned chart even when atanh is finite. A future implementation should refuse out-of-domain and poorly resolved probes, not clamp (z), since clamping would change the requested η.

The displayed linear face equation is exact in real arithmetic for the pinned basis, limits, coefficient map `limits * tanh(control)`, and face orientation. Production `einsum` and finite-difference reductions can differ by FP64 summation order. At each nonzero probe, recompute the actual production (Q_y[3,2]) from Γ and record `actual_flux - requested_eta`; do not label the point as lying on the requested side unless the actual sign agrees and its magnitude exceeds a scale-aware roundoff allowance. The existing `FVFaceFluxCoordinateChart` explicitly warns that its coordinate reduction is not an event or branch certificate (`fv_face_flux_coordinates.py:52-60, 209-255`).

## Minimal probe set and evaluation contract

Use the five requested levels, in ascending order:

\[
\eta\in\{-2\times10^{-6},-1\times10^{-6},0,+1\times10^{-6},+2\times10^{-6}\}.
\]

At ±1e-6 and ±2e-6, run one original-(J) endpoint evaluation and obtain the complete full-control gradient only if the actual production flux is strictly nonzero, all branch/limiter choices are defined, and the branch-margin checks pass. At exactly zero, run the objective and production flux evaluation only: **no AD, HVP, PCG, optimizer, or root/eligibility check at this point.** Do not use a centered finite difference across zero as a substitute for a derivative; compare each side using its own pair of nonzero probes.

For each nonzero sample retain:

- requested and production-recomputed η; reconstructed full (c); full-control hash;
- original (J), full (g=\nabla_cJ), Φ, raw-control (\|g\|_2), (\|g\|_\infty);
- branch signature, selected-face sign, minimum scaled face-flux margin, and all limiter signatures/margins exposed by the existing strict branch checker;
- chart domain margin (1-|z|), (dc_{24}/d\eta), and whether any other face or limiter changed between probes on the same side;
- derivatives in chart coordinates computed by chain rule, with definitions and units recorded.

For the chart derivatives, for each (i\ne24),

\[
\frac{\partial c_{24}}{\partial c_i}
=-\frac{\partial Q/\partial c_i}{\partial Q/\partial c_{24}},
\qquad
\frac{\partial c_{24}}{\partial\eta}=\frac{1}{\partial Q/\partial c_{24}},
\]

where the derivatives use the pinned analytic face map (and ∂Q/∂c24 is negative here). Then (J_\eta=g^T\partial_\eta\Gamma) and (J_t=(D_t\Gamma)^Tg). Also report the Euclidean orthogonal tangent projection

\[
 g_T=g-n\frac{n^Tg}{n^Tn},\qquad n=\nabla_cQ,
\]

and its norm. Do not call (\|J_t\|_2) the Euclidean tangent residual: (J_t) is a coordinate covector whose norm depends on the raw coordinate scaling. If reporting a QR or other metric norm, define and freeze that metric before inspecting results; include both the unscaled full (g) and the Euclidean projection so a rescaling cannot conceal residual size.

The within-side pairs ((-2,-1)\times10^{-6}) and ((+1,+2)\times10^{-6}) give one-sided secants for (J_\eta) and chart-gradient changes without crossing the face. They are diagnostics over this short interval, not asymptotic derivatives at zero. Compare (J(0)) with the two side trends only as a continuity check; account for FP64 objective resolution and do not infer differentiability from five values.

## Interpretation gates

1. **Chart invalid or unresolved:** if any (z\notin(-1,1)), actual flux has the wrong sign, the realized flux error is too large, or the pivot is not representable, stop and report chart refusal. No scientific inference follows.
2. **Other active branches change:** if limiter/other-face signatures change within either same-sign pair, the data do not isolate a single-face one-sided trend. Report a multi-event neighborhood and do not attribute the result to Qy[3,2] alone.
3. **(J) decreases across the face while Φ increases:** this supports the local merit-conflict hypothesis for the tested fixed tangent. It does not prove the Newton line search will cross, nor that the face is the sole cause.
4. **Normal and tangent gradients:** a nonzero (J_\eta) at/near the face rules out a constrained stationary point on this slice. A large Euclidean (\|g_T\|_2) shows the tested point is not close to tangent stationarity. Neither establishes an unconstrained stationary point or active-face minimum.
5. **Potential active-face follow-up:** only consider a separate constrained analysis if a later point has small tangent residual, suitable positive tangent curvature on the same smooth stratum, and a normal condition consistent with a local minimum from both sides. That work requires its own current-point operator, branch-eligible response definition, and complete prior. This five-probe diagnostic does not compute curvature or qualify response.

## Provenance, execution, and resource guard

The diagnostic must bind to the existing PR #256 experiment rather than merely accepting a control vector that happens to match:

- pin `F462_INEXACT_CONTINUATION_PLAN_20261007.json` and its SHA;
- validate the top-level raw `step.json`, `step.run.json`, and `step.resource.json`, their child digest/normal process completion/resource status, and the raw accepted-control hash `e29c348d51e7ec2de23f3f74d1522a34f58bf0bb72fd56e63d9b89cadea15e37`;
- close the PR #256 source/archive manifests against the plan and ensure the raw `source_before == source_after`; source drift in transport, objective, observations, prior, branch checker, or parameter construction must refuse the run;
- require frozen input and parameter hashes to match the accepted endpoint. Keep execution success, numerical/branch eligibility, and physical interpretation as separate result fields.

The cited PR #256 archive audit already reports 104 source and 28 archive pins matched, normal child/parent completion, numerical `budget_refusal` only after two accepted steps, 724.747 s guarded wall time, 377,421,824-byte sampled RSS, and expected e29 endpoint. Treat those as archived producer evidence; do not rerun or inflate its 780-second / 1-GiB resource budget to perform this diagnostic. Give the new diagnostic its own single guarded launch and a declared cost-only/gradient-evaluation budget before launch. There is no new HVP budget because the contract forbids HVPs.

## Existing source locations

- `src/advar/transport.py:702-760` — bounded flow coefficients and fixed coefficient limits.
- `src/advar/transport.py:365-380` — production oriented face fluxes (Q_y=-(\psi[:,1:]-\psi[:,:-1])).
- `examples/weather_scenarios/fv_point_3h_current_newton_step.py:100-116` — full-control to physical flow and production face flux reconstruction.
- `examples/weather_scenarios/fv_face_flux_coordinates.py:52-60, 209-255` — chart-domain, FP64, and non-certificate limitations.
- `graphify-out/fv-root-cause-20260919/F462_BOUNDARY_PROGRESS_20261007.json` — saved endpoint face-flux trend.
- `graphify-out/fv-root-cause-20260919/F462_RED_FINAL_20261007.md` — producer provenance and resource review.

## RED conclusion

Proceed only as the fixed-tangent, five-level diagnostic above. Its strongest possible result is evidence that the dual (J/\Phi) acceptance policy encounters a local merit conflict near this modeled face for this frozen endpoint and tangent. Keep the chart exact in the original latent controls, retain all prior terms, evaluate zero cost-only, and refuse interpretations that require an eligible root or a constrained minimum.
