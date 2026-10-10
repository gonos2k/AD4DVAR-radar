# RED code preflight: three-step coupled-GN resume — 2026-10-10

## Clearance

No actionable RED blocker remains for the one planned root-owned guarded run. RED did not run FV, seed reconstruction, production gradients/HVPs, tests, or the guard.

Frozen plan SHA-256 is `1f0eff706a21982407f79b51375687f41aca85ee34433229d8faa8e060a19ce0`, with 139 source and 148 archive pins; I independently checked all 287 hashes against the current files, with zero mismatches. Frozen adapter/core/test hashes are `abb2539fcd1562d8fbcd413af01a26844c51744e11dd833702715a007020595f`, `eca4eb6ee24abd840e89f0ab8d1fd84da615279e16066cc1cb04a293c3a2d55f`, and `4550823578644cf50837e66018e8ddf2a8da9bb0c78d97a35c871230333715e5`. The predecessor control is `4035904e46c11c67a6f937039fe3c88b0516330b05f3ba9d57c4cfa619423ab7`; the accepted base loader verifies the full PR273 raw/gzip/run/resource/manifest/plan and final-repeat P2 closure, then retains only the required input/runtime anchor and terminal accepted closure during the new trajectory.

The wrapper calls the shared coupled-GN factory but returns only the `robust_gn_coupled` arm. At each modeled point it therefore produces 24 fresh row reverse products, one 12×12 solve, and one fresh two-sided HVP pair; no baseline arm/HVP is run. The shared resume chain links each current control/theta to the preceding accepted closure and verifies the one-arm GN direction/HVPs, per-point row and solve history, candidate/final closure, actual commit count, and last-confirmed state. The saved-array analyzer, not a duplicated parent formula, recomputes per-point theta and merit.

Two refusal/closure edge cases are explicitly handled. A terminal full row-parity refusal after one commit may complete as a stopped partial sequence with exactly one terminal 24-row Cartesian side/row batch, no HVP/solve/commit at that attempted point, and the prior closed control/theta preserved. A GN solve that completes without returning a model/solve-history receipt fails closure and preserves the last commit; it cannot become a ghost candidate or false stop. The optional `row_budget` type guard was fixed before freeze.

The frozen test log reports 88 passed with 18 existing TorchScript deprecation warnings in 42.19 seconds; type checking reports zero errors/warnings/notes. The plan remains capped at three commits, at most 16 candidate slots per point, 72 row VJPs, 6 current HVPs and 3 row-space solves, 600/660 seconds and sampled RSS 1 GiB. The preceding one-step PR273 run used 984,743,936 bytes sampled RSS, leaving limited headroom; per-point state does not inherit old row/HVP operators or the full prior raw history.

This preflight supports only launch readiness. Any resulting three-step trajectory remains a bounded local record: no exact-curvature/minimum, stationarity, global smoothness, response, adjoint/VJP, reanalysis, or forecast-skill claim follows. The per-step 0.05 chart radius does not bound the total path by 0.05.
