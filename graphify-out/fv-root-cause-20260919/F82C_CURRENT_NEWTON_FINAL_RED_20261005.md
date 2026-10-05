# F82C current-point Newton step: final RED review

**Disposition: GO; no blocking discrepancy found in the completed one-step diagnostic.** Review used the archived current-curvature matrix, candidate report, and guard records. No new Hessian or HVP was computed.

## Pinned execution

- Newton plan SHA-256: `d7f639e82dcd9445eeef0e32472194b57f31088dc1247e22b52ab0acc37f4a8f`. All 80 source pins and 4 archive pins still match. The step’s source/input integrity flags are true and runtime identities match.
- The child and parent agree on completion and on the child hash. The one guarded child exited 0 in 18.663 s with a sampled peak RSS of 358,252,544 bytes, inside the 300 s and 1 GiB limits. No timeout, cancellation, monitor error, or process cleanup error was recorded.
- The solve used the newly archived f82c curvature audit (`3058fec5…`) and its Hessian, with the same f82c control and fixed parameter hashes. The direction audit reports `mu = 0`, original-system relative residual `8.58e-14`, and `g·s = -0.04099246`; no new HVP was requested.

## Accepted step and diagnostics

- The first radius-scaled candidate was accepted with `alpha = 0.19365644`; the standardized control displacement has norm `0.05`. Its own full endpoint branch and all 3,600 margins passed. Original `J` fell from `0.07811088` to `0.07161498`; Armijo passed. `Phi` fell from `1.76705799` to `1.11297362` and is reported as a diagnostic, not an acceptance gate.
- The local quadratic model predicted a `0.00716979` reduction in `J`; the actual reduction was `0.00649590` (actual/predicted `0.9060`). The candidate gradient’s relative linearization error was `0.8019`. The measured endpoint supports this accepted step, while the sizeable gradient mismatch limits extrapolation from the base Hessian.
- The minimum-curvature mode has eigenvalue `0.07362217`. It accounts for `0.5997` of the squared standardized direction norm and the same fraction of the accepted displacement norm. This is coordinate-space movement, not physical uncertainty.
- The branch signature changed. The separately captured accepted-point full-signature hash matches the Armijo trial hash. Its partition records 360 analysis-replay and 3,240 future-forecast stages; both partition hashes differ from the base point. These are endpoint branch records, with no connecting path certificate.
- Recorded physical-coordinate deltas are limited to initial model reflectivity/echo proxy, configured flow/face speeds, and interval log echo growth. The largest absolute initial dBZ change is `0.0856`; the largest echo-proxy change is `0.692`. They do not establish water mass, observed wind, forecast skill, or physical validity.

The candidate remains nonstationary (`gradient_inf = 0.7980`). The report correctly records one accepted optimizer step, no new HVP, no score or response, and false full-root/physical-validation claims. The next required evidence remains a stationarity assessment and any separately scoped response or forecast validation.

Reviewed artifacts: [step report](f82c_unshifted_newton_20261005_attempt1/step.json), [guard resource record](f82c_unshifted_newton_20261005_attempt1/step.resource.json), [parent run record](f82c_unshifted_newton_20261005_attempt1/step.run.json), and [frozen Newton plan](F82C_NEWTON_PLAN_20261005.json).
