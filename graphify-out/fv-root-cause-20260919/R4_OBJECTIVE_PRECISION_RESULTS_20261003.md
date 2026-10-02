# R4 rejected-trial objective precision: bounded closure

The same physical FP64 control vectors give native Delta J = +1.2712053631958042e-14, while independent 50/80-digit equations give Delta J approximately -1.32771101336227017e-19. An 80-digit directed-rounding interval enclosure of the captured-input discrete equations is entirely negative, with width 1.8251048411743553e-77. The native actual-J veto is therefore an objective-evaluation discrepancy at this point, not evidence of actual ascent in those equations. This closes the same-trial objective-difference diagnosis only. The historically rejected trial remains rejected; no full-root/response claim follows.

| Quantity | Result |
|---|---:|
| Native FP64 scalar difference | +1.2712053631958042e-14 |
| Accurate reduction of already rounded FP64 residual/prior components | +1.2712991096848402e-14 |
| Independent 80-digit difference | -1.3277110133622701709e-19 |
| 50/80 point-precision drift | -8.094940533321509e-50 |
| Directed interval difference width | 1.8251048411743553e-77 |
| Frozen original objective allowance | 5.9018533159127995e-16 |
| Max independent vs native prediction difference | 9.0432849836e-15 dBZ |

The interval endpoints are about -1.3277110133622701709e-19; exact binary tuples in `partial_slice_interval_attempt1/interval.json` are authoritative. Every interval limiter selection, flow sign, growth sign, initial clamp and softplus decision was definite. All 36 analysis-stage choices match the measured native path for both points; both endpoints also retain the original strict 54-stage signature and positive margins. Inputs, parameters and all 25 original source paths remain bound to the archived run. The reference independently implements the same latent transform, frozen binary constants, positive-growth SSPRK2/minmod, stage boundaries, per-frame rank-one whitening, 58-valid/two-missing mask and zero-centered prior.

The promoted FP64 component difference still reproduces the native increase: simply subtracting loss scalars more carefully would not fix it. A few prediction ulps become important when the objective decrease is far below a final scalar ulp and robust/prior differences strongly cancel. The gradient line-integral diagnostic (-1.10814e-19) concerns the chart path and is not a bound on the rounded physical endpoint difference; it is not substituted for the independent reference.

The predeclared check that both finite point approximations lie inside the 80-digit enclosure is **not met**: the 80-digit value is inside; the 50-digit value is outside by about 8.1e-50. That is finite point-precision error relative to a much narrower interval, not an interval-enclosure failure. No bound was widened to absorb it. The interval bounds enclose the captured-input real discrete equations assuming the interval library's directed elementary arithmetic; they do not enclose the original FP64 rounded-operation sequence, certify segment-wide branch regularity, or validate the physical model.

## Execution and corrections

- Precision attempt 1: exit 1, 3.407 s, no completed output. The diagnostic compared raw gradient norm ratio to the refiner's normalized Armijo ratio. Its original code/preflight/plan/log/resource record are preserved.
- Precision attempt 2: exit 0, 3.132 s, sampled peak child RSS 325,009,408 bytes in 12 samples; no resource termination or monitor error. Audit metric was corrected to norm_ratio/sqrt(1+2*c1*alpha*slope). The measured probe/reference bytes are preserved in that attempt directory.
- Interval attempt 1: exit 0, 0.265 s, no resource termination or monitor error. One RSS sample is insufficient for a useful peak-memory claim. This child performs no native PyTorch replay or optimization.
- The executed precision probe omitted its planned prelaunch runtime comparison. Recorded Python 3.12.13/Torch 2.13.0/CPU FP64 match the original run in a separate **post-execution** audit; partial plan conformance is explicit. Current probe now refuses mismatched runtime before problem construction. Current code is distinct from preserved measured snapshots; no repeat was performed merely to change check timing.
- Eight focused tests pass, including the Armijo distinction, fail-closed runtime mismatch, analytic growth/whitening, interval tanh containment, uncertain-branch refusal and interval-output serialization/containment. Focused type checks and source Graphify evidence are recorded separately. No full CI, package or deployment validation was added.

## Remaining checklist

- [x] Reconstruct and identify the historically rejected trial without accepting it.
- [x] Compare native scalar, promoted component, high-precision and interval objective differences.
- [x] Establish a negative same-point discrete-equation difference with definite interval branches.
- [x] Correct the audit metric and add a preconstruction runtime gate with regressions.
- [x] Declare and validate the fixed-slice interval objective gate; see `R4_INTERVAL_CORRECTION_RESULTS_20261003.md`. No tolerance was loosened.
- [ ] Find/qualify the original 26-control stationary solution, or justify a separately named nonsmooth/active-face contract.
- [ ] Only then issue the corresponding adjoint/VJP and nonlinear response checks.

R4-R remains open. A zero normal face flux is an upwind-flow event, not missing data or clear sky. These calculations do not demonstrate observation-based identifiability, forecast skill, finite physical influence or learning performance. GREEN and RED audit records distinguish actual measured versions from current code.
