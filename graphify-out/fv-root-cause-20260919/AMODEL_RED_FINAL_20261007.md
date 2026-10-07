# Model-guided full-space continuation — RED final audit

Date: 2026-10-07
Disposition: **The one guarded three-step attempt completed with three committed candidates that each passed the declared actual dual-Armijo and own-endpoint strict-branch gates.** The cumulative J, Φ, and gradient norms improved, but this is not a root, a same-branch trajectory, or a forecast/response result.

This is a read-only audit of saved arrays and receipts. I did not rerun FV, gradient, HVP, PCG, score, forecast, adjoint, reanalysis, or tests.

## Receipts and work

Plan SHA-256 is `a75b5e474e68fb74475c3fd7fb2f31e85c891edcb06c40e83b4287b16a7b3136`. I independently checked all **111 source pins and 43 archive pins**; all match. The raw child SHA `92497bab…` equals the parent `child_sha256`, the parent resource object equals the resource sidecar, and before/after snapshots preserve source, fixed input and runtime. The resource receipt records exit 0, no resource/monitor/signal error, **82.199 s** guard time and sampled peak RSS **381,239,296 bytes** under the 300 s / 1 GiB limit.

The child committed three full-control steps, each using one fresh HVP at its current point; total HVPs are 3 and PCG solves are 0. All three first candidates passed, so there was no backtracking. The alpha values were 0.0002434285, 0.0011829460 and 0.0003597724; actual full-control moves were 0.0001627662, 0.0004182921 and 0.0001041651, each below the 0.05 radius. No cached HVP was reused.

## Measured progress and redistribution

| Point | (J) | Φ | ∥g∥₂ | ∥g∥∞ | (Q_y[3,2]) |
|---|---:|---:|---:|---:|---:|
| PR #258 start `05889584…` | 0.061555713011 | 0.223540152425 | 0.668640640 | 0.510128497 | (-5.91061\times10^{-7}) |
| Step 1 | 0.061486718357 | 0.062517193837 | 0.353602019 | 0.165249192 | (+1.82986\times10^{-5}) |
| Step 2 | 0.061377878815 | 0.041913938066 | 0.289530441 | 0.171123583 | (+4.79366\times10^{-5}) |
| Step 3 | 0.061369252443 | 0.018131732819 | 0.190429687 | 0.097245034 | (+1.37948\times10^{-5}) |

Cumulatively J fell **0.30291%**, Φ **91.88883%**, ∥g∥₂ **71.51988%**, and ∥g∥∞ **80.93715%**. The first full-space (-g) step crossed the selected face from negative to positive flux; later steps stayed positive. This releases the temporary tangent restriction as intended.

The per-block gradients fluctuate. Step 1 reduces the initial-field, flow and growth blocks. Step 2 then raises the maximum gradient from 0.165249 to 0.171124 even as Φ falls. By step 3 the field, flow and growth block norms are about 0.155531, 0.098748 and 0.048192. The final maximum gradient is still **0.097245**, about (9.72\times10^8) times the (10^{-10}) pending-root threshold. The result is a meaningful full-space correction, not stationarity.

## Model, Φ policy and branch limits

Each direction is the fresh (d=-g) at that iterate. The saved slopes are resolved negative in all three steps:

| Step | (g^Td) | (g^THd) |
|---|---:|---:|
| 1 | (-0.4470803) | (-1303.4288625) |
| 2 | (-0.1250344) | (-55.0057825) |
| 3 | (-0.0838279) | (-52.1683270) |

The αΦ cap sizes the starting trial from a local linearized-gradient model. It does not guarantee actual Φ decrease. The implementation also checks actual full-gradient Φ Armijo and a roundoff-resolved actual Φ decrease at every candidate; all three actual checks pass.

The saved independent model arithmetic shows J quadratic actual/predicted reduction ratios of **0.98265, 0.99467 and 0.32209**. The linearized-gradient relative errors grow from **9.88%** to **95.66%** to **149.92%**, and the predicted-gradient/actual-gradient angles are **5.51°, 61.77° and 78.23°**. These are finite-step model differences, not evidence of an AD defect. The Φ computed from the linearized gradient is 0.0648943, 0.0299828 and 0.0325296 versus actual Φ 0.0625172, 0.0419139 and 0.0181317. Actual nonlinear values, not these model predictions, govern acceptance.

Both the analysis and future branch-partition hashes change on every committed step, even though each candidate passes its own strict branch/margin gate. The first target-face sign switch occurs on step 1. Thus the HVP/model slope is local to each base branch; the three accepted endpoints do not certify an event-free path. The direction changes every iterate and all 26 controls move, so the cumulative improvements cannot be attributed solely to the (Q_y[3,2]) crossing.

The saved-array angle between the initial gradient at `05889584…` and the final gradient at `69c4a794…` is **63.1336°**. This is the actual old-to-new gradient angle, not a model-prediction angle. The separate per-step predicted-gradient/actual-gradient angles are **5.51°, 61.77° and 78.23°**.

## Scope and remaining limits

The runner preserves prior commits if a later iteration refuses or reaches budget, uses no J decrease floor, and emits only `root_pending_audit` at the declared gradient threshold. This run ended `max_iterations_completed`; it makes no full-Hessian SPD, full-root, active-face minimum, adjoint/VJP/reanalysis, forecast-skill or physical-boundary claim. The original full 26-dimensional root and independent future verification remain open.
