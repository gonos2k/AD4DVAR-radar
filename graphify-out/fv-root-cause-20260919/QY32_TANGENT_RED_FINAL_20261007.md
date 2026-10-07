# Qy[3,2] fixed-η tangent correction — RED final audit

Date: 2026-10-07
Disposition: **The single bounded chart-tangent attempt completed and passed its declared actual-J / actual-Φ Armijo, own-endpoint strict-branch, radius and final-closure gates.** It is one exploratory correction on the original full objective. It is not evidence of overall stationarity or same-branch convergence.

This is a read-only audit of the saved child, parent, resource, plan and summary. I did not rerun FV, AD/HVP, PCG, score, forecast, adjoint, reanalysis or tests.

## Run and receipt closure

The child finished with `one_qy32_tangent_dual_armijo_step_accepted`, one committed candidate, **one HVP and zero PCG solves**. Child JSON SHA `061c51d2…` matches the parent `child_sha256`; parent resource data equals the resource sidecar. The process exited 0. Parent-guard time was **59.294 s**, child time **57.952 s**, and sampled peak RSS **521,453,568 bytes** under the 300 s / 1 GiB guard. Source and fixed-input snapshots matched before and after. The plan is `70f9560c…`; the base is the pinned e29 control `e29c348d…` and accepted control is `05889584…`.

The accepted chart preserved the same nonzero production face value on both endpoints:

\[
Q_y[3,2]=-5.910612738066479\times10^{-7}.
\]

The stored actual full-control displacement was (3.90625\times10^{-4}), within the 0.05 radius and only **0.78125%** of it. Seven larger candidates were rejected; the eighth, at α `0.0005224020701`, passed. The child checks actual chart displacement before each expensive candidate evaluation.

## What improved and what worsened

| Quantity | e29 base | Accepted chart point | Change |
|---|---:|---:|---:|
| Original (J) | 0.061699268708 | 0.061555713011 | **0.23267% lower** |
| Actual Φ | 0.279198206009 | 0.223540152425 | **19.93496% lower** |
| ∥g∥₂ | 0.747259267 | 0.668640640 | **10.52093% lower** |
| ∥g∥∞ | 0.413558059 | 0.510128497 | **23.35112% higher** |
| Euclidean tangent-gradient norm | 0.747244423 | 0.651284054 | lower |
| Euclidean unit-normal gradient | +0.004709951 | −0.151358467 | sign changed |

The block norms explain why the maximum gradient worsened despite the L2 merit falling. The field block improved from 0.702241 to 0.392114, while flow rose from 0.049130 to 0.181922 and growth rose from 0.250681 to 0.510128; growth is the final maximum-gradient component. The negative unit-normal component and reduced tangent residual do not make the endpoint stationary: the tangent residual is still about **0.6513**, and the raw maximum gradient is **0.5101**.

The one-HVP base slopes were g·d = −0.5587502088 and g·Hd = −1087.1234794, both resolved negative. At the accepted α, the dual-Armijo thresholds use these current-point first derivatives and the actual candidate passes both. The actual Φ reduction is much smaller than the linear first-order scalar extrapolation suggests: that extrapolation gives Φ ≈ −0.2887, an unphysical finite-step value, while the evaluated candidate Φ is 0.22354. The stored linearized-gradient model g + αHd is close to the fresh candidate gradient (error norm about 6.20e−4, relative about 0.093%). Keep these model fields labeled as first-order diagnostics. Acceptance correctly used the actual full-gradient Φ.

## Branch and interpretation limit

The base and accepted endpoints both pass their own complete strict branch and margin gates, but their branch signatures differ. Both the 360-stage analysis partition and 3240-stage future partition hashes changed. The step receipt retained hashes and partition hashes, not the full branch arrays, so this audit cannot identify which limiter choices changed. Consequently the HVP is a valid local derivative at e29, while the finite accepted transition is not certified to stay on that branch. Report this as a finite fixed-η exploratory transition, not a branch-preserving smooth tangent continuation.

The target face remains at a small negative nonzero flux; this step does not cross (Q_y[3,2]=0). It also does not impose a permanent model constraint or remove the pivot/prior contribution. The accepted point has lower (J), lower Φ and lower L2/tangent residuals, but a higher max-gradient component and a changed branch partition. No root, constrained face minimum, final curvature, response, forecast score or physical boundary claim follows.

## RED conclusion

The attempt is internally consistent with its narrow policy: one fresh live HVP at e29, actual dual-Armijo checks at curved fixed-η candidates, an actual full-control radius cap, and fresh final endpoint/identity closure. It produces a real local correction, but the maximum gradient worsens and the endpoint crosses other branch partitions. Keep it as one bounded exploratory result; do not call it convergence or use it as a qualified normal point. The original 3-hour normal-root, adjoint/reanalysis and independent forecast-validation milestones remain open.
