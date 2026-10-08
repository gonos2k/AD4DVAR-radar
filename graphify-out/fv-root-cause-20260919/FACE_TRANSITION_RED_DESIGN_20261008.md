# External Qy face-transition diagnostic — RED design

Date: 2026-10-08
Scope: a fixed-retained-coordinate finite-sample diagnostic at the accepted resume control `2cdccade…`, focused on zero-based external top face `Qy[4,3]`. It evaluates no candidate direction and applies no optimizer update. No source file, FV trajectory, objective gradient, HVP, PCG, forecast score, adjoint or reanalysis was run or changed for this review.

## Fixed chart and objective contract

Use the accepted 26-control vector at `2cdccade81a8cf218dd7cad97adf71f6675f8445481dd8f6394ec337ea8e8007`. Hold its other 25 raw controls exactly fixed and restore pivot `c[24]` so the production face has each of the four predeclared nonzero values

\[
\eta\in\{-2\times10^{-6},-1\times10^{-6},+1\times10^{-6},+2\times10^{-6}\}.
\]

For the pinned basis and limits, the zero-based external top-face map is

\[
Q_\star(c)=Q_y[4,3]=-.08\tanh(c_{21})-.28\tanh(c_{22})-.14\tanh(c_{23})-.84\tanh(c_{24}).
\]

Derive these weights from the loaded streamfunction basis/coefficient limits and compare against the pinned expected values; do not trust a separately copied formula alone. For each probe, require finite atanh argument in the open domain, preserve the full 26-vector, and verify the actual production bounded-coefficient/combined-streamfunction/face-flux value against requested η with a predeclared FP64 reduction bound. Do not overwrite a face-flux array. Keep the original full J and every prior component, including the pivot prior. No face constraint persists beyond this diagnostic.

At exact η=0, evaluate the original primal J and actual production face value only. It may be a roundoff-sized nonzero value. **Do not compute a gradient, Φ, strict branch certificate, or AD derivative at zero.** At the four nonzero points compute full-control J, (g=\nabla_cJ), Φ (=\|g\|_2^2/2), and pointwise branch/margin evidence. The finite samples are not exact one-sided limits.

## Correct branch-stage and external-trace contract

For the fixed profile, the analyzed objective uses the three observation frames across two intervals: (2\times90\times2=360) Euler stages. The pointwise forecast branch check covers 3600 stages total; the remaining 3240 are future stages and do not enter this observation-only J. Report analysis and future partitions separately.

Use `transport.observe_minmod_stages` only around the existing `problem.branch_check` inside `torch.no_grad()`. The observer receives detached copies of each actual stage’s (q,q_x,q_y); it does not differentiate the FV path. Evaluate each finite point’s objective gradient in a separate ordinary `torch.func.grad` call, outside the observer and no-grad scope. At zero, run no branch observer. Validate that observer callback count, branch signature counts and fixed 3600-stage layout agree. If callbacks are missing/duplicated (for example an auxiliary large-boundary-trace minmod call), retain the partial trace and mark it unavailable; do not silently align or truncate it.

Do not bind all four nonzero probes to the resume-base `expected_branch` signature: the positive-η side must be allowed to have the target-face sign change. Each point must pass its own strict local branch/margin gate; then compare signatures and margins across probes. This is endpoint qualification only, not a smooth path certificate.

The observer does **not** receive boundary edge values. To compare an effective interior/external trace at this top face, reconstruct the exact stage edge from the existing boundary schedule and use the production sign/upwind convention:

- (Q_y[4,3]>0) is top outflow and uses the interior reconstructed top state (`top_face`);
- (Q_y[4,3]<0) is top inflow and uses the external top echo trace.

Use zero-based SSPRK Euler-stage parity. For each substep, stage 0/even is the start stage and stage 1/odd is the end stage. If the per-substep log-growth is positive, the start-stage boundary echo supplied to Euler is scaled by \(\exp(g_{step})\); the end-stage boundary echo is raw. If log-growth is zero or negative, both Euler edge traces are raw. The observer’s (q) is already the actual interior stage state used by transport; do not scale it a second time. Compute `g_step` from the interval growth divided by the configured 90 substeps, and map analysis and future schedules to the same stage index used by the branch trace. If this schedule/parity mapping cannot be bound to the production call, omit the effective-trace comparison and report only observed (q,q_x,q_y) plus strict branch metadata.

The sign switch selects different upstream data at this boundary. Endpoint traces can explain a local branch mechanism but cannot certify a continuous trace path, exclude a simultaneous limiter/upwind event, or show that this face caused a J/Φ change. Record all non-target face signs, analysis limiter choices and future limiter choices. Any extra analysis branch or non-target face change within a same-side pair blocks single-face attribution even if each endpoint individually passes the strict margin gate.

## Gradient jump, Φ identity and sampled hull diagnostics

Use a single common Euclidean control-space normal for the four pairwise comparisons, computed from the smooth face map at the zero-η chart control with the same retained 25 coordinates:

\[
n_i=w_i(1-\tanh^2(c_i)),\quad i=21,22,23,24;\qquad \hat n=n/\|n\|_2.
\]

This differentiates only the static smooth Q map; it does not evaluate J or J’s derivative at the switch. Verify ∥n∥ is finite and resolved before normalization, and report the metric as Euclidean in the standardized raw-control coordinates. For each matched offset pair ((g_-,g_+)) at ±1e-6 and ±2e-6, report

\[
\Delta g=g_+-g_-,\quad \bar g=(g_++g_-)/2,\quad
\Delta g_n=\hat n^T\Delta g,\quad
\Delta g_T=\Delta g-\hat n\Delta g_n.
\]

Check the exact identity

\[
\Delta\Phi=\tfrac12(\|g_+\|^2-\|g_-\|^2)=\bar g^T\Delta g
=\bar g_n\Delta g_n+\bar g_T^T\Delta g_T
\]

using the full stored gradients. The decomposition is a finite-sample algebraic attribution in this chosen coordinate metric, not a limiting normal jump.

For each offset pair separately, compute the minimum-norm point on the sampled gradient segment:

\[
\lambda=\operatorname{clip}\left(-\frac{g_-^T\Delta g}{\|\Delta g\|_2^2},0,1\right),\qquad
g_{seg}=g_-+\lambda\Delta g.
\]

If ∥Δg∥ is below a scale-aware FP64 bound, record a degenerate pair and do not divide. Report λ and ∥gseg∥, plus full/tangent/normal blocks. These are two-point segment diagnostics only: not a Clarke subdifferential, not proof that every limiting gradient has been sampled, and not a normal/tangent stationary-point test. Do not combine the ±1e-6 and ±2e-6 pairs into one hull.

## Provenance and bounded attempt

Pin the final resume plan/raw/parent/resource, final control SHA, fixed inputs, Qy face diagnostic plan/source and existing boundary archive. Require the completed resume receipt to close at `2cdccade…`; remeasure inputs by comparing fixed parameters, archived observations/truth, boundaries and schedule while allowing the expected control hash to equal the final resumed point. Require the diagnostic plan’s original 111 source pins to remain unchanged and add only its two new diagnostic source/test pins. Snapshot all inputs/sources/runtime before and after. Write into a fresh attempt directory; preserve resume raw/result/resource, previous model-guided producer files and gzip archive byte-for-byte.

Use one serial guarded launch, 240 s internal / 300 s external / sampled RSS at most 1 GiB. Budget zero HVP, zero PCG and zero optimizer steps. Order η=0 primal-only plus four nonzero J/g/Φ/branch samples once each; write partial point traces durably after each result. A timeout or branch refusal is not a complete five-point comparison. Keep process/resource completion, numerical sample status, branch eligibility and trace completeness separate.

## RED conclusion

This experiment can report finite-sample changes in full gradients, Φ, analysis/future branch signatures and one-sided effective boundary traces near external (Q_y[4,3]). Its strongest valid claim is evidence from four fixed-tangent nonzero samples plus a cost-only zero sample at the pinned resume endpoint. It cannot establish exact one-sided derivatives, Φ continuity/discontinuity at zero, a Clarke or conditional minimum, global absence of other branch events, or a meteorological barrier/forecast effect.

## Draft implementation audit (read-only)

The current draft is **not yet runnable** and should not be launched until these source-level blockers are fixed and the frozen source pins regenerated:

1. The module has two `_load_plan` definitions. The later definition replaces the earlier one and calls `tangent._pinned_path`, but `tangent` is never imported or bound in this module. `_run_child` also calls `tangent._check_fixed_input` in both model-receipt validation and final closure. The earlier shadowed definition refers to `model.tangent`; that does not make the unqualified name in the active definition valid.
2. `_run_child` calls `_verify_production_face(...)`, but no such function is defined or imported. The nearby `_verify_face_value(...)` has a different signature and does not substitute automatically.

The jump summary also falls short of the declared math contract. `_gradient_jump` currently computes only the inner (pm10^{-6}) pair's (Delta g), its normal/tangent split, a scalar identity value, and that pair's sampled segment minimizer. It should separately compute both matched pairs ((pm10^{-6})) and ((pm2\times10^{-6})), using one common zero-chart Euclidean unit normal, and emit the measured (Delta\Phi), (ar g^T\Delta g), normal and tangent contributions, plus a roundoff-scaled identity residual for each. Each pair's segment minimizer should report its full gradient and normal/tangent/tangent-space norms; do not merge the pairs. These are finite-sample diagnostics only, not Clarke or limiting-gradient certificates.

For bounded-attempt integrity, retain and durably write a branch-refused sample before stopping or classifying it incomplete. Check the internal deadline after the caught branch-refusal path and after final source/input/runtime closure, as well as before starting the close; otherwise the process can report `completed` after the 240-second internal limit. The outer guard remains a separate 300-second/RSS bound. Keep `execution_status`, sample/branch eligibility, trace completeness and numerical status separate. A budget refusal or incomplete donor trace must not be promoted to a complete five-point comparison.

The observer contract still needs the exact checks from this design: run only the detached stage observer inside `no_grad` and the existing `branch_check`; use separate full-control AD gradients only at the four nonzero points; never differentiate or branch-check the η=0 cost-only control. Preserve all 26 prior terms and the original objective. The external donor trace may explain which edge was selected at a sampled endpoint, but even a same-side branch signature match does not isolate this face as the cause of any cost or Φ change; the full signature and other-face/limiter deltas must be retained.

No FV objective, gradient, HVP, PCG, optimizer step, score, adjoint, or reanalysis was executed for this RED review.
