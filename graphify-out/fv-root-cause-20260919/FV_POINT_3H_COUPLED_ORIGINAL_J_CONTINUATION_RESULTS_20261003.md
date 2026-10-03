# Four-epoch coupled original-J continuation

One guarded reusable continuation ran from the accepted R8 control (SHA-256 `3aa913cd71d5916b4f3146efad75b0f5dd6766f6bdc81ee6e8c6706cee03fc32`, initial `J=0.1077570519`). It kept the original 26-control objective, all zero-centered priors, 13 parameters, observations, data/time/boundary setup, and terminal target unchanged. The six dynamics and 20 field controls remained search coordinates.

At each of four accepted epochs, the driver freshly reconstructed the full Hessian and its Hff/Schur blocks, then selected a shifted block inverse because the current original Hessian/Schur was not resolved SPD. The dynamics-only shifts were `11.92543`, `36.97624`, `17.27101`, and `8.90576`. In every epoch, the PCG operator was the fresh original-objective HVP plus the shift on the six dynamics coordinates; the SPD block inverse was only a preconditioner. Each modified system converged in one PCG iteration, with true relative residuals `1.0553e-12`, `4.4227e-12`, `2.0735e-12`, and `2.0726e-13`. The original-H residuals were separately recorded as `0.42553`, `12.02885`, `0.59630`, and `0.52391`, so these are modified search solves, not original Newton solves.

Original-J Armijo accepted one strict endpoint per epoch at alphas `0.125`, `0.015625`, `0.0625`, and `0.0625`. J decreased monotonically from `0.1077570519` to `0.0838971809`. The full-gradient infinity maxima by epoch were `3.47245`, `1.83542`, `3.16276`, `1.51078`, and `1.05882` at the final accepted endpoint; Φ values were `18.59198`, `5.44925`, `11.78273`, `3.78709`, and `1.76360`. The infinity norm and Φ increased at epoch 2 despite the original-J Armijo acceptance, which is permitted by this policy. All accepted endpoint branch checks passed, with signatures recorded; no connecting-path certificate follows.

The child stopped at the declared four-epoch cap with `epoch_limit`. The final full-gradient infinity norm `1.058816824` remains far above `1e-10`, so no stationary candidate was issued. The terminal full-Hessian/root, adjoint, response, and physical-validity claims remain false.

The guarded child completed in `992.643` seconds with exit code `0`, no external termination or monitor error, and peak sampled RSS `375,242,752` bytes under the 1,500-second / 1-GiB limits. Git HEAD remained `3cf06409c2982db0a39909921eeb50b640a507bf`; all 64 captured source paths, the R8 raw, fixed parameters, and input/data/time/prior identity matched before and after. The older R8 driver and test remain byte-for-byte unchanged.

Authoring verification was seven focused pure tests, offline basedpyright with zero errors, clean diff-check, and targeted Graphify AST extraction for the new producer/test (29 nodes, 74 edges). The shared graph and report hashes were unchanged. No response/adjoint/forecast evaluation or follow-on run was performed.

## Scope

This is one reusable four-epoch coupled search continuation. It reduced original J but did not meet the full-gradient stationarity gate. No additional scientific phase is authorized by this result.
