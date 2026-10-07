# 90fc bounded continuation — final GREEN review

Date: 2026-10-07
Disposition: **GO for the one completed bounded follow-up run.** This review reads saved run records and independent arithmetic only. I did not rerun FV, HVP, PCG, tests, forecasts, adjoints, or reanalysis.

## Run outcome

The run starts from the final committed 90fc point, control SHA-256 `90fc45552d068ae4f1b83eb54ab095362f223dc10b884695ccdbe1e63c9825fc`, using frozen plan SHA-256 `1c3cc494667ca6f9a3a0f57d714f78479d18249a44c1b3a397223484bd7f8067`. It committed one step, to control SHA-256 `d31ecd393002a1e098c939838ce22e9a7526c1b7f0f426effbb3f3a408ccdeeb`:

| Quantity | Start at 90fc | Committed point | Change |
|---|---:|---:|---:|
| Original objective J | 0.06301158389782033 | **0.062118987765887354** | 1.4166% decrease |
| Φ = ||g||²/2 | 0.400981044045754 | **0.3252767184999386** | 18.8798% decrease |
| ||g||∞ | 0.548030683 | **0.4267256181192793** | decreased |
| Step length | — | 0.0125 | α = 0.1472228629 |

The accepted trial passed both actual-J and Φ Armijo checks and its strict endpoint branch gate. Its completed PCG solve used 29 PCG iterations and 31 HVP calls including the independent true-residual action; the recorded true relative residual was `1.3093e-11`, below the frozen `1e-10` tolerance. Three candidates were evaluated for that step.

A second solve started at the newly committed point and stopped at the cooperative internal deadline. It used 23 additional HVP calls, bringing the run total to 54 started and 54 completed HVPs. Its PCG iteration count is recorded as `not_recorded`; the 23 HVP calls must not be described as 23 PCG iterations. The incomplete second direction was not committed or reused.

## Execution and integrity

The child receipt reports `phase=finished`, `execution_status=completed`, and `numerical_status=budget_refusal`, with one committed iteration. The parent receipt also reports completed execution and the same numerical stop. The child exited 0; the outer guard recorded 722.154 seconds under its 780-second wall limit, sampled peak RSS 379,125,760 bytes under 1 GiB, no monitor error, no resource termination, and no SIGTERM. The child’s elapsed time was 720.595 seconds; the stop was the continuation’s own cooperative budget refusal.

The raw result SHA-256 is `ade22e99afc1a19e662cf66efd89356fd1b99c1d39dacb26aad8182bbaec5cc0`. The result records unchanged source/runtime and fixed-input identity. The plan still has 98 source pins and 19 archive pins; I rechecked that every listed path exists and every current SHA matches. The old f82c Hessian remained scoped to the block inverse preconditioner; the current-point operator produced this run’s directions.

## Independent saved arithmetic

`90FC_FOLLOWUP_SCIENTIFIC_CHECKS_20261007.py` records independent NumPy checks over the saved arrays and receipts. It verifies control hashes and reconstructed step vectors; recomputes each completed solve’s true linear residual and directional slopes; reconstructs both Armijo thresholds; checks the accepted trial is the final trial in its search and carries strict branch and both merit-pass flags; and checks actual J and Φ decrease at the committed point. It also checks the raw result SHA, plan SHA, current source/archive pins, fixed-input/source receipts, and HVP accounting. This is verification of saved evidence, not a second numerical model execution.

## Limits

The final maximum gradient component is still about `4.27e9` times the `1e-10` stationarity threshold. One accepted improvement and a deadline stop do not establish convergence or a stationary point. No global positive-definiteness claim, adjoint result, nonlinear reanalysis, independent forecast score, or physical validation was produced. Keep those milestones open and report this run as one committed optimizer step followed by a cooperative budget refusal.
