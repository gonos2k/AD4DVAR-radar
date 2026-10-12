# GREEN final — bounded 7839 radius check

**The guarded run closed successfully and correctly skipped candidate evaluation.** The newly admitted 7839 point's fresh local KKT entry for the second direction lies outside the frozen radius cap, so this plan did not make an expanded-radius move.

## Frozen plan and endpoint lineage

The plan hash `466c4ed8c8eb594e6c4f3293a204a9692dd145448eb0c85fc79c110e108f0c03` and source bundle hash `ac3dd6ece38bd23425adbca6d5c436529cc4054e4b5e8fedb709ef4f7723bbed` were verified before launch. The bundle has exactly 222 planned sources with matching hashes; the plan contains 291 inherited archive pins. Its only numerical source transitions are the shared runner and rank-aware subspace solver.

The receipt-only loader admitted 7839 from the archived PR301 run and closed both repeated arms to the d19e starting endpoint. The run then freshly checked the native pair, carry-forward P2, face-release condition, and minimum-mixture P2 at 7839. Both new directions and their four HVPs were generated at this control; `current_KKT_radius_selection.old_d19e_operators_reused` is false.

## Fresh KKT gate and run outcome

The current normalized V is full rank, with singular values `[64.0963793, 1.5690640]`. At the base radius `1.8757177916958825e-5`, axis 0 passes KKT with `lambda=11661.7991242` and `mu1=+0.1049055310`; both signs are resolved against their recorded scale-aware budgets. Independent arithmetic from the stored fresh `G`, `D`, and `V` reproduces the KKT free-coordinate stationarity residual `2.78e-17`. The fresh inactive-axis multiplier reaches zero at `5.0012972662e-5`, which is `2.6664x` the base radius and `1.3333x` the plan cap `3.7514355834e-5` (`2x`). The driver returned `no_resolved_subspace_entry_within_frozen_radius_cap` before any candidate evaluation.

The child closed after 77.86 seconds; the guarded wall was 79.70 seconds, with 1,096,499,200 bytes sampled peak RSS (about 1.021 GiB). It completed four fresh HVPs, 24 GN row VJPs, and one dense solve. Candidate evaluations, repeats, and commits are all zero. The current and last-confirmed control hashes both remain `7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a`; source, fixed-input, and runtime closure pass. Each fresh direction and both side HVP vectors differ from their archived d19e counterparts, and every new HVP event carries the 7839 control hash. The stored `verify_expansion.py` audit reports all checks true.

This is a successful abstention under the frozen gate, not a failed model run. No new actual J/R candidate was evaluated. No score was computed. The d19e KKT trend did not transfer to fresh 7839 directions, which is why the current-point gate was necessary. No convergence, exact minimum, response improvement, or forecast-skill claim follows.

The prelaunch suite reports 67 passed in 24.43 seconds with 18 existing TorchScript deprecation warnings. Expansion-config basic Pyright reports zero errors and warnings. No physical evaluation was run by this GREEN audit.
