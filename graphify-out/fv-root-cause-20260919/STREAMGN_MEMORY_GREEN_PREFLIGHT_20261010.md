# GREEN preflight: new 2-GiB GN continuation from the f2ca endpoint — 2026-10-10

## Disposition

The new adapter and frozen plan are clear for the root-owned bounded run. This is a separate experiment of up to three new commits starting from the saved f2ca endpoint; it does not retroactively finish the original 1-GiB plan, which remains at two of three commits. The predecessor's third-point rows and top-level dense audit are retained as provenance only. The new adapter returns only the last independently P2-closed endpoint, so common continuation code must build fresh rows, solve, and paired HVPs at its first point.

## Plan and saved-base checks

The final plan SHA is `7c93aa28219c73e7a437feaac7341887da8fc6fc0b9f0b8e4633ad4424224c67`, with 144 source and 173 archive pins. I independently recomputed every source/archive hash; there were no mismatches. The plan raises only the bounded RSS cap to 2 GiB, keeps the other point-level budgets and thresholds fixed, and explicitly says its maximum is three **new** commits.

The read-only plan and base loaders passed. They admit control `f2ca572936a8bf646c920c542b3780ff65a0f94c0d715aa61ba00a5cec57313d`, theta `0.4830637745604483`, J `0.06119270839098437`, and F² `0.004479439946247095` from the second P2-closed endpoint. Reusing the NumPy collector verifies both earlier endpoints, their two point-level GN solve receipts, and the third point's local row-space audit. That prior audit reconstructs successfully, but it has no paired HVP or candidate/endpoint repeat; the adapter does not return its rows, solve, direction, or audit arrays in the base payload.

The restart input anchor derives both `control_sha256` and `shifted_flow_fractions = tanh(control[20:25])`, with all other fixed input identity fields preserved. Its provenance and runtime anchor match the frozen plan. Fresh input/runtime equality is checked at the new process boundary before the model observes the base.

## Implementation and verification

The new producer and test match their final source pins `6c884d94…` and `3271f050…`. The focused suite passed 7/7 according to the implementation receipt; no production work was run during this GREEN review. The analysis successor at `STREAMGN_MEMORY_ANALYZE_20261010.py` is syntax-checked and its saved-only base-preflight path passes: it separately reports the old two-step prefix and tail audit, then analyzes only the new run's own rows, HVPs, solves, candidates, resource, and source receipts.

The remaining evidence is the root-owned attempt's raw/run/resource/archive receipt and subsequent saved-array analysis. A successful three-new-step sequence would support only those new local steps. Neither the 2-GiB allowance nor lower memory use would establish convergence, an optimum/root, exact curvature, response, adjoint, reanalysis, or forecast skill.
