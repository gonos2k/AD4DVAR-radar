# GREEN final review: repeated dual-merit correction from e0b04a

## Disposition

**The saved numerical evidence supports two committed correction steps under the frozen plan, followed by the planned internal-budget refusal during the third solve.** The child finished with two committed endpoints and passed source and fixed-input closure. The raw parent receipt nevertheless says `execution_status=failed`: its caller used an older 300-second resource classifier for this plan's 780-second outer guard. Treat this as a reporting/contract defect in the archived parent record. The later classifier fix correctly classifies the same resource receipt as completed, but it does not rewrite that record or change the producer that ran. No rerun was performed.

The attempt used plan SHA-256 `ee75ca45b03495b055023a984457c4aab73eb76c646c1c17bf8760d2e53f80d9`. Its archived producer, `e0b_repeat_dual_20261006_attempt1/producer.py`, hashes to the exact producer digest pinned by that plan. The raw result SHA matches the parent's child digest; `source_before == source_after` and `fixed_input_unchanged` is true. The original `step.run.json` and its failed status remain preserved.

## Numerical result

Starting from the accepted e0b04a point, both committed steps reduced actual original `J`, `Φ=||g||₂²/2`, and the maximum gradient component. The endpoint results are:

| Point | `J` | `Φ` | `||g||∞` |
|---|---:|---:|---:|
| Start e0b04a | 0.06835683126 | 0.87567774445 | 0.78815557616 |
| After step 1, `8c1285…` | 0.06457612277 | 0.50556623587 | 0.62478453429 |
| After step 2, `90fc45…` | 0.06301158390 | 0.40098104405 | 0.54803068272 |

From the saved start to the final committed endpoint, the reductions are **7.82% in `J`, 54.21% in `Φ`, and 30.47% in `||g||∞`**. These percentages are independently reproducible from the raw endpoint values. Neither step reached the configured `||g||∞≤1e-10` gate.

Step 1's accepted candidate passed both Armijo gates and its strict endpoint branch check after one larger trial failed only the `Φ` gate. Its PCG solve used 26 iterations, 28 HVPs including true-residual verification, and had true relative residual `1.28e-11`. Step 2 passed after two larger candidates failed the `Φ` gate; its solve used 27 PCG iterations, 29 HVPs, and had true relative residual `6.81e-11`. Thus the two accepted solves used 53 PCG iterations and 57 HVPs in total.

The third solve started and stopped at the 720-second cooperative deadline after the run reached 62 started and 62 completed HVPs. Its exact PCG iteration count is explicitly `not_recorded`; do not infer it from HVP calls. The top-level numerical termination is `budget_refusal`, with both committed endpoints retained and no provisional third candidate. The run used less than both outer resource limits: guard time was `724.979` seconds against 780 seconds, and sampled peak child RSS was `372,686,848` bytes against 1 GiB. The child exited 0; there was no monitor error, resource termination, or SIGTERM.

The independent arithmetic companion verifies the accepted control hashes, per-step true residuals and directional slopes, both actual Armijo thresholds, candidate/control hashes, strict endpoint flags, and the realized HVP limit using the archived source copies. This was a replay of saved arrays and scalar records, not a new FV, HVP, PCG, or forecast computation.

## Parent status defect and later fix

The archived parent status is `failed`, and the result summary generated from that record also reports `execution=failed`. The saved resource object shows an exit code of 0, elapsed time below its own 780-second limit, RSS below its 1-GiB cap, no termination, and no monitor/cleanup error. The prior shared classifier instead compared against its old 300-second default, so it rejected this valid 780-second receipt.

The post-run fix adds a continuation-specific classifier using the plan's 780-second and 1-GiB limits. Its offline receipt, `REPEAT_DUAL_CLASSIFICATION_FIX_20261006.json`, records that the same resource is `failed` under the old classifier and `completed` under the new one, while preserving the historical parent status. The current producer and test hashes are `c4bfa13b92d188a4201ccfe524552910cfd28c5a5c22fe48c050953a689c2fb6` and `c8b4ac3b353b80712c83aa2143c55b7d90bf2885ea0fc9494fe16b9e830a0304`; they differ from the historical run's pinned producer/test copies. The offline classification regression reports **30 passed, 18 existing warnings**, with **0 type errors, warnings, or notes**. The current source therefore needs a new plan before any future launch. These checks neither replayed the FV work nor retroactively changed the raw receipt.

## Scope and remaining limits

This result establishes two accepted optimizer updates under the fixed objective and input, with actual `J` and `Φ` decreases and strict endpoint checks. It does not establish final normality, a stationary or eligible point, full-Hessian/global SPD, a smooth branch path between accepted endpoints, an adjoint or reanalysis, or improved forecast skill. The raw output keeps `full_root_claim=false` and `eligible_stationary_point=false`; response and forecast score remain uncomputed. The internal-budget stop is not evidence that a root is absent.

## Evidence reviewed

- [Frozen plan](</Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/E0B_DUAL_MERIT_CONTINUATION_PLAN_20261006.json>) and archived producer copy beside the run output.
- [Child raw receipt](</Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/e0b_repeat_dual_20261006_attempt1/step.json>), [historical parent receipt](</Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/e0b_repeat_dual_20261006_attempt1/step.run.json>), and [resource receipt](</Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/e0b_repeat_dual_20261006_attempt1/step.resource.json>).
- [Independent arithmetic result](</Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/E0B_REPEAT_DUAL_RESULT_20261006.json>), [scientific-check script](</Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/REPEAT_DUAL_SCIENTIFIC_CHECKS_20261006.py>), and [classification-fix receipt](</Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/REPEAT_DUAL_CLASSIFICATION_FIX_20261006.json>).
- [Post-fix affected tests](</Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/REPEAT_DUAL_POSTFIX_TESTS_20261006.log>) and [post-fix type check](</Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/REPEAT_DUAL_POSTFIX_TYPES_20261006.log>).
