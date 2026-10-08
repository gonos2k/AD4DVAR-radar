# Exploration/correction cycle RED final audit

Date: 2026-10-08
Scope: read-only audit of the saved single cycle run and its child, parent, resource, source, and historical-reference receipts. I did not rerun FV, AD, HVP, PCG, or tests, and made no implementation edits.

## Receipt and execution closure

The run used the frozen plan SHA-256 `a91588d2281325c95518c79e677043823b18ed52e51ee0557d55c6e3e51dec12` (117 source pins, 66 archive pins). The child SHA-256 is `275107930a951925726f91a68300123ddeebcf302283a26e3d72aca0c5274fae`; the parent receipt names that exact child hash, reports `completed`, has no child read error, and records `cycle_recovered`. The resource receipt matches the parent copy: exit code 0, no monitor/resource termination, 37.74 s elapsed, and sampled peak RSS 378,912,768 bytes under the declared 300 s / 1 GiB guard. The internal child elapsed time was 36.61 s.

The child reports two committed corrections, two HVP starts/completions, zero PCG solves, and no active candidate. Each iteration’s base control equals the prior committed endpoint; each accepted control/J/Φ/gradient exactly matches its accepted trial; the final current control and state match the second commit. Both candidates passed J Armijo, Φ Armijo, and strict endpoint branch checks. Source-before equals source-after for all 178 union entries; the map exactly matches the frozen plan pins plus the plan itself, and every pinned source/archive digest still matches on disk. Fixed inputs and runtime close, with the final control hash recorded as `6c0fe485…`.

The cycle base and comparator are correctly distinguished. Correction started at PR262 control `26e830dc…`, J `0.061317125140052595`, Φ `0.02995458290516701`. The pinned PR260 pre-exploration comparator is `2cdccade…`, J `0.061347638323085395`, Φ `0.012673246502950312`. The child’s cycle assessment exactly matches an independent recomputation from those serialized values and the final state, including the FP64 roundoff budgets:

- Final J is `0.0613117140406184`, which is `3.592428246699392e-5` below the PR260 comparator and resolved beyond the `1.7436047506602505e-15` comparison budget.
- Final Φ is `0.010815707540872811`, which is `0.0018575389620775016` below the PR260 comparator and resolved beyond the `3.601953296467375e-16` comparison budget.
- Both J and Φ also decreased from the correction start by resolved amounts. The serialized `recovered=true` assessment is therefore supported by the final child state and pinned comparator.

## Numerical and branch limits

Final gradient infinity norm is `0.11651289508164255` (L2 norm `0.14707622201343637`). The infinity norm remains above the configured `1e-10` stationarity gate and above its PR260 comparator value `0.0885108925278776`. This run supports a bounded J/Φ recovery result only; it does not establish stationarity or forecast/response evidence. Score and response flags are false.

All recorded base and accepted endpoints passed strict branch eligibility with 3,600 choice stages and 3,600 face-sign stages. Full branch signatures changed from cycle start `aecc7f2e…` to correction 1 `6448e050…` and correction 2 `7f210f96…`. The analysis partition changed at correction 1 (`973742e8…`) and returned to the cycle-start partition at correction 2 (`48f3297d…`); future partitions changed at both corrections. The endpoints are individually strict, but this does not show an invariant branch along either path or isolate a face as the cause of the objective changes.

## Parent semantic check and disposition

The parent receipt validates execution, child hash, and generic iteration/source/input/runtime closure; it does not independently recompute the cycle assessment. I checked that metadata directly against the raw final child state and pinned comparator, and it matches. Keep this parent-receipt limitation explicit; the raw-child semantic audit is the evidence for `cycle_recovered`.

No actionable discrepancy was found. Treat the result as one bounded exploration/correction cycle with two committed corrections and a verified endpoint recovery comparison. Do not describe it as a stationary solution or attribute its gains to one stable branch or face mechanism.
