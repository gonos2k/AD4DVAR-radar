# R4 precision-aware correction: execution verified

A separately declared one-step run closes the false scalar-objective-veto limitation for this fixed +2eta0 tangent correction. It changes only the objective-comparison evaluator to directed interval arithmetic. The exact original objective/prior, binary inputs, source/runtime identity, strict 54-stage signature, normalized gradient Armijo, 128*eps64*J one-sided allowance, PCG/true residual and final tangent-gradient thresholds are retained.

| Measured result | Value |
|---|---:|
| Newton corrections / PCG iterations | 1 / 26 |
| Refiner HVP calls, including independent monitor product | 29 |
| Fresh tangent gradient max | 8.341830590001542e-12 |
| Independently recomputed linear relative residual | 1.3586958021141202e-11 |
| Original full 26-control gradient max | 0.0061897982634800605 |
| Native scalar Delta J | +1.2712053631958042e-14 |
| Interval mathematical Delta J | about -1.32771101336227017e-19 |
| Original unchanged allowance | 5.901853315912799e-16 |
| Original normalized Armijo ratio | 0.0012678976705653648 |
| Guarded elapsed / sampled peak child RSS | 28.9021 s / 344,293,376 bytes |

Exit code 0, no resource termination or monitor error; 110 RSS samples. A fresh gradient is checked against both refiner return and `<1e-10`. The saved Newton solve is independently re-audited using fresh gradient and HVP. Both the original25 measured-source set and all new producer/adapter/reference/precision-helper bytes are unchanged before/after. The reference uses the captured fixture and checks all36 analysis choices; the original branch callback checks the full54-stage signature/margins. The interval implementation does not separately certify the54-step forecast trace.

The newly accepted control SHA256 is `c1ed15743517e34728c8db2bdbb50d568294eb08750b9b6ac1186f3c04152aae`. It matches the coordinate vector reconstructed from the old rejected full step. This is a **new policy execution**, not retrospective acceptance or editing of historical evidence. The original signed-slice run stays `partial_slice_refusal`; precision attempt1's audit implementation error and attempt2's post-runtime verification remain explicit.

The objective gate is one-sided: interval upper<=tau adds no veto; interval lower>tau vetoes a definite increase; overlap refuses as precision uncertainty. Large genuine decreases are permitted. Returning `None` never overrides native gradient Armijo. Neither the objective nor its permitted increase is rescaled. The original native FP64 scalar calculation is unchanged and is not claimed to have gained accuracy; this research adapter compares the intended captured-input discrete equations at adequate precision instead.

## Closure and remaining scope

- [x] Diagnose the same-point objective discrepancy with independent point and interval arithmetic.
- [x] Implement the precision-aware gate, with increase/uncertainty/large-decrease regressions.
- [x] Execute one correction and independently verify tangent stationarity, true linear residual and identities.
- [ ] Original full-root/normal-direction stationarity remains unresolved: full gradient is about6e-3, partial eta slope is +0.00354318946.
- [ ] Full-root qualification, adjoint/VJP and signed nonlinear response checks remain required for R4-R.

Three older tangent points plus this new fourth point were obtained under two objective-comparison policies; do not aggregate them as one uniform success-rate experiment. Finite opposite-side slopes do not prove a cusp, active-face minimum, root absence or uniqueness. Forecast skill, flow identifiability, learning and finite physical influence are not measured. This is fixed4x5 serial research code and does not introduce a general interval solver, runtime dependency lock change, deployment or full-CI rerun.
