# GREEN final audit: streamed partial coupled-GN resume — 2026-10-10

## Disposition

The saved attempt confirms **two accepted, independently closed continuation steps** from the saved PR274 endpoint. It stopped during the third current point at the 1-GiB RSS guard. The archive and parent receipts match, all 142 source and 168 archive pins in frozen plan `529110d25ca28ac85ab3ce3fa2fd23a8b2a8ba0122c2f184ab5e26a3c20b4f17` match, and the confirmed child prefix matches the saved control/theta checkpoint. This review used the NumPy saved-array analyzer once; no FV, seed, AD, HVP, optimizer, or guard was rerun.

## Confirmed points

| Point base | Confirmed endpoint | Per-point work | Candidate slots | Model and prediction checks |
|---|---|---:|---:|---|
| `7cec4c…` | control `4eede849…`; θ `0.48300428290133907`; J `0.06119357252204779`; F² `0.004483245047494389` | 24 rows, 2 HVPs, 1 recorded 12×12 solve | 11 | χ `0.8692761`; accepted α `0.0007131432` (1/1024 of α₀); full residual relative error `0.5279%`, angle `0.3015°`; delta-vector error `100.17%`, angle `85.612°`; components 12–14 hold `99.93%` of squared error. |
| `4eede849…` | control `f2ca5729…`; θ `0.4830637745604483`; J `0.06119270839098437`; F² `0.004479439946247095` | 24 rows, 2 HVPs, 1 recorded 12×12 solve | 12 | χ `0.8694199`; accepted α `0.0003567105` (1/2048 of α₀); full residual relative error `8.30e-7`, angle `0.0000475°`; delta-vector error `0.1823%`, angle `0.1033°`; components 12–14 hold `20.00%` of squared error. |

The full-residual error is normalized by the actual endpoint residual norm; the delta-vector error is normalized by the actual residual-change norm. χ is a dimensionless `cos²(F, DF)` diagnostic, not an alpha or an acceptance criterion.

Both accepted repeats pass their saved endpoint/P2 checks and fresh minimum-θ recomputation. These are local step receipts, not solver convergence or a minimum/root certificate.

## Interrupted third point and resources

The third point at control `f2ca5729…` completed its 24 row VJPs. The HVP counters and receipts are both 4/4, so no third HVP was started. There are no third-point candidates or commit. The dense-solve counters say 3 started and 3 completed, while only two point-level solve-history receipts were stored. The third solve does have a top-level 12×12 audit. The NumPy analyzer rederived `t` from the same-control last-confirmed P2 gradients, rebuilt B from the third point's saved rows, and checked S, the right-hand side, solve vector, correction, uncorrected direction, and charted direction. The B reconstruction relative error is `1.18e-16`; the solve vector relative error is `1.17e-13`; the recomputed solve residual is `1.40e-15` against a saved budget of `2.15e-12`; and the minimum eigenvalue of S is `1.00166`. The audit's context matches the current/last-confirmed control and theta, though the audit object itself has no control hash. This supports a locally closed third row-space solve and direction algebra. It does not close a full GN model point: there is no per-point solve-history entry, paired HVP, complete direction-model arm, candidate search, or independent endpoint repeat. The analyzer's `unmodeled_tail_solves=0` counts missing solve-history entries; it should be read together with the separately validated top-level tail audit and the 3/3 counters.

The corrected attempt ran for `437.161` seconds and exited `-15` at sampled RSS `1,091,633,152` bytes, which is `17,891,328` bytes over the `1,073,741,824`-byte limit. It saved a two-step prefix, not the configured three-step cap. The attempt continued through two commits and third-point row work with streamed checkpoints in place. The saved run cannot quantify how much RSS the writer saved or establish that serialization caused the crossing. `comparison_progress` is absent from the latest checkpoint; while a next point is uncommitted, progress may still be absent before direction progress is emitted. The writer lifecycle regression separately covers omission from the committed snapshot and preservation on write failure.

The superseded attempt 1 is preserved as a preflight-only failure: `4.808` seconds, zero row products, zero HVPs, no dense solve receipt, candidate, or commit. It is not numerical optimization evidence. Across the session the archive records one failed preflight and one corrected numerical guard attempt.

## Provenance and limits

The corrected attempt's `source_before` matches the frozen source/archive maps and all current pins. It has no `source_after`, and top-level input/runtime before/after fields are absent because RSS interrupted the child. The derived restart-anchor provenance is recorded, but this archive therefore does not establish whole-run source/input/runtime unchanged closure. Do not extend the predecessor's closure flags to the interrupted run.

The analysis artifact is [STREAMGN_RESULT_20261010.json](STREAMGN_RESULT_20261010.json). Its archive manifest confirms raw/gzip round-trip and raw/run/resource hashes; the saved child digest, parent digest, resource receipt, and two-step terminal control/theta reconcile. Resource closure and terminal execution closure are false. No causal memory conclusion, convergence claim, exact-curvature result, response, adjoint, reanalysis, or forecast-skill claim follows from this path.
