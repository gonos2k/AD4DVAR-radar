# Three-point terminal forecast observation

The sealed observation plan is `9f7e281b626292126f3345546d9a350996d6b6f72b9b8b35a68d36a35cfd6b05`.
It inherits all 237 source pins and 306 archive pins from the signed-span run,
adds eight signed-run/case receipt pins, and keeps the exact controls 2a4d,
fb0a, and 96108. The guarded child completed in 9.971 seconds with a sampled
peak RSS of 772,358,144 bytes. The runtime, inputs, and all inherited source
pins remained unchanged.

The observer made exactly three `problem.score` calls. Each call captured its
18 forecast lead fields during that same forward, without optimization,
candidate evaluation, HVP, or selection. The terminal dBZ metrics are:

| Point | Terminal MSE (dBZ²) | RMSE (dBZ) | MAE (dBZ) |
|---|---:|---:|---:|
| 2a4d | 7.384565484459929 | 2.7174557005515156 | 2.3803717784314884 |
| fb0a | 7.384323633729144 | 2.7174112007072364 | 2.380326853483219 |
| 96108 | 7.3842515162645155 | 2.7173979311585037 | 2.380311593461612 |

The saved fields are at 10-minute leads from 10 through 180 minutes after the
analysis endpoint. Only one original terminal verification array is available;
it is a 4×5 grid. No aligned 18-lead truth arrays are present, so lead-wise
MSE/MAE and persistence comparisons are unavailable. The terminal MSE/MAE
values were independently recomputed from the saved final frames and this
terminal truth.

The raw `forecast_validation.json` contains one annotation error: its
`score_domain` string says “fixed 4x5 off-grid verification points.” The actual
score domain is the **full 4×5 terminal verification grid**, as confirmed by
the fixed problem’s `verification_dbz`, captured terminal forecast, and exact
recomputation of each score. The raw receipt is preserved unchanged; consumers
should use the corrected domain stated here.

This is a terminal-score observation at three fixed controls. It does not
establish a lead-time skill curve, operational forecast skill, or a global
improvement claim.
