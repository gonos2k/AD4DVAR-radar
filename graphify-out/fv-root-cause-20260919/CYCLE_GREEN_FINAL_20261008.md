# GREEN final audit: exploration-correction cycle

The saved cycle completed as `cycle_recovered` after two accepted full-space corrections. I audited the saved JSON receipts and their recorded arithmetic; no FV evaluation or HVP was rerun.

Plan SHA: `a91588d2281325c95518c79e677043823b18ed52e51ee0557d55c6e3e51dec12`.

| Receipt | SHA-256 |
| --- | --- |
| `step.json` | `275107930a951925726f91a68300123ddeebcf302283a26e3d72aca0c5274fae` |
| `step.run.json` | `1bc20c78729c631d44f2eead29b795b92ece15318bb5b5a5e9203dae69300372` |
| `step.resource.json` | `ccdf33208489e58c96cd0aedd1ad56eaca530cd0b80e86fe30911d1936406e4f` |
| `step.log` (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The parent receipt is `completed`, has no child read error, and its child SHA matches `step.json`. Its resource object matches `step.resource.json`; exit code was 0, no resource termination or monitor error occurred, wall time was 37.74 seconds, and sampled peak RSS was 378,912,768 bytes of the 1 GiB cap. The child recorded 36.61 seconds, source-before equals source-after, fixed inputs remained unchanged, and runtime-before equals runtime-after.

The cycle starts from accepted exploration control `26e830dc…` and references the pinned pre-exploration control `2cdccade…`. Final `current_control` hashes to `6c0fe4858631513f665f91c77006d4641e564862d8141111c0030c91d2bb74bd`, equals the last committed iteration endpoint, and matches the accepted trial SHA. Both correction iterations used the fresh negative base gradient direction, started and completed one current-point HVP, and accepted their first candidate with strict-branch passage, resolved Phi decrease, and both Armijo tests passing. Candidate counts were `[1, 1]` (2 total); the recovery gate stopped after correction 2, within the three-correction cap. The correction loop used 2 HVPs; with the pinned exploration step’s 1 HVP, the combined cycle used 3.

| State | J | Phi |
| --- | ---: | ---: |
| Pre-exploration 2cd reference | 0.061347638323085395 | 0.012673246502950312 |
| Correction start 26e | 0.061317125140052595 | 0.02995458290516701 |
| After correction 1 | 0.06131658772348851 | 0.022464730748383992 |
| Final 6c0f | 0.0613117140406184 | 0.010815707540872811 |

The final endpoint is below both references in both recorded metrics. Relative to pre-exploration 2cd, J changed by `−3.592428246699392e−5` and Phi by `−0.0018575389620775016`; relative to correction start 26e, J changed by `−5.411099434193822e−6` and Phi by `−0.0191388753642942`. The child assessment reports `recovered=true`, `resolved_correction_decrease=true`; its roundoff budgets are `1.7436e−15` for J and `3.6020e−16` for Phi, so both comparisons are well resolved.

This is a recovered merit cycle, not a root result: final `||g||∞ = 0.11651289508164255`, versus the `1e−10` root threshold, and `full_root_claim=false`. The parent receipt checks generic execution and iteration closure but does not independently recompute cycle metadata; this remains the previously recorded nonblocking limitation.
