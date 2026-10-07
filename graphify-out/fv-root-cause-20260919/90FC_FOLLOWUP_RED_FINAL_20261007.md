# 90fc follow-up continuation — final RED review

Date: 2026-10-07
Scope: saved-record audit of the single 90fc follow-up. I did not rerun FV, HVP, PCG, tests, forecasts, adjoints, or reanalysis.

## Disposition

**The one-step continuation result is supported as recorded; the requested bounded run ended at its cooperative internal deadline.** The outer execution completed within its guard. One candidate was committed from the pinned 90fc start; the second linear solve began but did not finish. This supports one additional accepted dual-merit step, not repeated convergence or a stationary point.

## Committed numerical change

The accepted control is `d31ecd393002a1e098c939838ce22e9a7526c1b7f0f426effbb3f3a408ccdeeb`, from start `90fc45552d068ae4f1b83eb54ab095362f223dc10b884695ccdbe1e63c9825fc`. The saved values are:

| Quantity | Start 90fc | Accepted d31ecd |
|---|---:|---:|
| Original objective J | 0.06301158390 | **0.06211898777** |
| Phi | 0.40098104405 | **0.32527671850** |
| Gradient infinity norm | 0.54803068273 | **0.42672561812** |

The independent arithmetic receipt reports J down **1.42%** and Phi down **18.88%**. The accepted step had length `0.0125`, used `alpha=0.1472228629`, and was the third evaluated candidate. Its live solve used 29 completed PCG iterations and 31 HVP calls, with true relative residual `1.3093e-11`. The candidate passed both actual Armijo conditions and its own complete strict endpoint branch check.

The endpoint branch signature changed from the start signature. The receipt establishes strict branch validity at the accepted endpoint; it does not establish a smooth branch along the path. The direction’s linearized gradient still differed from the actual endpoint gradient by relative norm `0.4965`, and the actual-to-predicted J reduction ratio was `0.76145`. These limit how far the local quadratic model can be trusted.

## Budget stop and unfinished second solve

The second solve started from the committed d31ecd point with 31 cumulative HVPs already used. A further 23 HVP calls completed before the 720-second internal deadline, bringing the total to **54 started and 54 completed HVP calls**. The second PCG iteration count is correctly `not_recorded`; these HVPs do not constitute a completed direction and were not used for a candidate.

The child records `execution_status=completed` and `numerical_status=budget_refusal`; the parent records the same statuses with no child-read error and a matching child hash. The guarded run used `722.154 s` of its 780-second wall limit, sampled peak RSS `379,125,760` bytes against 1 GiB, exited 0, and records no SIGTERM, resource termination, or monitor error. The single committed control is preserved.

## Integrity and independent checks

The child’s source hashes are unchanged and `fixed_input_unchanged=true`; the changed control is the intentional optimizer output, so the full input identity hash is expected to differ across start and accepted points. The parent child SHA-256 matches the raw step receipt. `90FC_FOLLOWUP_RESULT_20261007.json` and `90FC_FOLLOWUP_SCIENTIFIC_CHECKS_20261007.py` identify their scope as saved receipt verification and independent NumPy arithmetic; no FV, HVP, or PCG regeneration was performed. The recorded checks validate the exact plan hash `1c3cc494667ca6f9a3a0f57d714f78479d18249a44c1b3a397223484bd7f8067`, source/archive pins, accepted controls, Armijo thresholds, residual, and accounting.

## Remaining limits

The final maximum gradient is still about `4.27e9` times the `1e-10` threshold. No accepted-point curvature, eligible stationary point, global SPD claim, adjoint, reanalysis, or independent forecast score was produced. The budget stop alone gives no evidence of negative curvature, root nonexistence, or future convergence. The run demonstrates one further accepted step and preserves it for any separately planned continuation.
