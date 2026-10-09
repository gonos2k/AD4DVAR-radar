# PR265 GREEN preflight

Date: 2026-10-09
Scope: read-only final preflight of the exact selected-upwind-face coupled probe. No FV/model case, gradient, HVP, optimizer step, forecast score, or reanalysis was run for this review; no source or test files were edited.

## Identity and bounded policy

The frozen plan is `NONSMOOTH_COUPLED_FACE_PLAN_20261009.json`, SHA-256 `83ea077a6d7ba615a84729a3588a92b156687dd02a8fc7320cf9b90ecf5ce299`. It pins 122 sources and 78 archive inputs, the accepted PR264 control `95a55578…`, and the selected external top face `Qy[4,3]`. The archived pre-extension transport source is `kink_source_20261009/transport.py`; its manifest preserves the original bytes. The plan policy permits one guarded launch, 600 s internal / 660 s outer time, sampled RSS below 1 GiB, 56 HVPs, one dense solve, at most eight candidate trials, and one accepted correction. It disables root, minimum, and response claims.

The current loader checks the plan digest and policy, source/archive pins, producer plan lineage, and PR264 raw/run/resource receipt hashes in [fv_point_3h_nonsmooth_coupled_probe.py](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py:386). `_load_base` now derives the producer’s base control from the pinned plan and closes the accepted endpoint and receipts ([same file](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py:415)); the earlier hard-coded, mismatched parent control is removed. The root reports that both loaders passed against this plan and receipt.

## Mathematical and implementation gates

The source helper changes one face’s positive/negative split while preserving `plus − minus = Q`; its context validates the fixed face and resets in `finally` ([transport.py](/Users/yhlee/ADVAR/src/advar/transport.py:59)). Trajectory replay freezes both the selected branch and native `None` branch so checkpoint recomputation cannot inherit a later context ([transport.py](/Users/yhlee/ADVAR/src/advar/transport.py:1017)).

The runner verifies the raw PR264 endpoint, checks its saved native gradient exactly, and then reconstructs the exact zero-face chart by changing only the selected pivot while retaining the other 25 raw controls ([probe](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py:605), [chart qualification](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py:630)). Production face value is checked against the weighted-coordinate map and a cancellation-scaled FP64 bound. The 360-stage analysis gate requires strict non-target face and minmod margins, identical other-branch signatures on both extensions, and a roundoff-sized selected-face flux ([trace gate](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py:151)). The ±1e−8 parity probes now verify the production face value and requested sign before comparing native and selected-extension J, gradient, HVP, and branch support ([parity gate](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py:672)).

The coupled system uses both exact-face side gradients and Hessians, a minimum-norm segment weight, and a chart-tangent direction. Acceptance requires both side slopes and the mixed slope to descend, J Armijo decrease, scaled residual merit decrease, exact-face production audit, and unchanged non-target branch support ([candidate gates](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py:795)). A proposal is committed only after fresh native J, both side gradients, extension/native value parity, mixed merit, traces, face value, source/archive hashes, fixed inputs, runtime, and deadline close ([final closure](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py:853)). The code labels base-point curvature as diagnostic only and makes no smooth-root, minimum, or response claim.

## Verification and readiness

The existing focused evidence records 38 tests passed in 2.58 s, 25 transport replay tests passed in 1.98 s, and zero type errors; both test runs had 18 existing TorchScript deprecation warnings. The root reports a fresh native-gradient receipt comparison, plan/source preflight, and type/source checks passed. No new FV result follows from those checks.

**Readiness:** the frozen source, plan, lineage, and bounded runner gates are ready for the single guarded numerical launch. This is launch readiness only; resource use, branch support, the candidate, and any scientific outcome remain unmeasured until that launch completes. A branch-support refusal is an acceptable diagnostic result and must remain distinct from an optimizer success or model-validity claim.
