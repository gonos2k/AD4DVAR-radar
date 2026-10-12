# GREEN preflight — bounded 7839 radius trial

## Status

**GREEN preflight passes for the frozen plan.** I reviewed `radius_expansion_trial.py` statically and called its receipt-only `load_endpoint()` once. No fresh observation, FV evaluation, HVP, candidate evaluation, score, or guarded run was launched. The endpoint admission returned the archived committed control `7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a` with control/full-G shapes 26/27 and theta `0.6938343734`.

The loader reads PR301 source/receipt members from the pinned archives, then checks each repeated eligible arm and the selected commit against d19e using `_close_step`. It verifies the selected endpoint control, theta, and full-G against the archived child. The plan allows exactly two numerical source transitions: the rank-aware solver and shared runner. At runtime the driver calls `trial._run_child(..., experiment=...)`, so it reuses the shared current-P2, four fresh HVP, 24 GN row/solve, two-arm candidate and repeat gates instead of copying a second numerical loop.

The fresh KKT callback is correctly placed after fresh current P2, four fresh HVPs, 24 GN row/solve work, and fresh direction/model construction. It requires full-rank V, resolved current axis multipliers, and a future axis-entry radius below the `2x` cap, then solves the current local model at the selected radius and skips all candidate evaluations unless the added coefficient and predicted residual-merit gain exceed scale-aware `128 * eps` screens. This keeps the d19e KKT result out of the 7839 operator path.

The frozen plan SHA256 is `466c4ed8c8eb594e6c4f3293a204a9692dd145448eb0c85fc79c110e108f0c03`; its source bundle SHA256 is `ac3dd6ece38bd23425adbca6d5c436529cc4054e4b5e8fedb709ef4f7723bbed`. I independently re-admitted it through `radius_expansion_trial.load_plan()`, verified that the source tar has exactly 222 planned members with matching hashes, and confirmed 291 inherited archive pins. The source transition contains exactly the shared runner and rank-aware solver.

The current radius policy keeps one shared 600s internal / 660s outer / 2GiB limit, HVP4, rows24, solve1, up to 6 candidates per arm, 2 repeats total, one commit and radius cap `3.751435583391765e-5`. It requires both arms independently eligible and uses repeated actual R then J for selection.

The expansion-specific `/tmp/advar_rank_pyright.json` includes the work directory in `extraPaths`; the recorded expansion type run reports zero errors and warnings. The full affected suite reports 67 passed in 24.43 seconds with 18 existing TorchScript deprecation warnings.

I found no remaining GREEN blocker. `RED_PREFLIGHT_EXPANSION.md` still contains a pre-freeze note that the plan/archive closure is pending; refresh the RED record to the frozen hashes before launch. The actual chart, original J/R, repeat P2/full-G, source/input/runtime and resource gates remain inherited from the shared runner.

The d19e audit supports at most one bounded fresh-point diagnostic after requalifying 7839. It does not support a production radius change: the d19e local model’s predicted gain at `1.5x` is only `3.4053e-5` of gradient-axis R, and no above-radius finite candidate exists in the stored receipts.
