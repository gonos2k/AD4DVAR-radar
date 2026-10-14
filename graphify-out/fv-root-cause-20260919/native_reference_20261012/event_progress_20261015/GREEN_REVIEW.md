# GREEN review — event progress attempt 1

Read-only evidence review of `attempt1/PLAN.json`, `attempt1/RESULT.json`, `attempt1/PLAN_SHA256.txt`, and `attempt1/source.tar.gz`. No objective, gradient, Hessian, score, or forecast was recomputed for this review.

## Outcome

The run located two distinct positive-slope limiter equalities, selected the earlier located stage-48 event, and accepted two lower-native-J Lagrangian-search steps on that active event. The terminal candidate is **not stationary and is not qualified as a minimum**. The run status `no_native_Armijo_progress_at_current_event` means the remaining proposed trials were refused after an unsupported intervening limiter transition; it does not establish that every feasible move has nondecreasing J.

The initial requalified native objective was `0.0601191808341518`. The terminal objective is `0.0601191807868374`, a decrease of `4.731439545713201e-11`. The terminal 24D projected-gradient infinity norm is `1.1416882854509208e-3`, far above the `1e-10` stationarity threshold. The stored selected-branch reduced Lagrangian model has minimum eigenvalue `0.8590889593022794`; the record labels it a same-point selected-AD search model. It is not the full four-sector/event-sector P2 required for a minimum claim.

## Event and accepted-step evidence

The recorded roots are separate actual-control evaluations with one active limiter event each:

| Stage | Ray alpha | Native J | Its gap | Other stage gap |
|---|---:|---:|---:|---:|
| 47 | `0.0006178200128488243` | `0.06011918080010134` | `4.8672e-13` | stage 48: `-2.44687e-6` |
| 48 | `0.0004443460493348539` | `0.06011918080517632` | `8.8818e-13` | stage 47: `+2.44385e-6` |

Both events are x slopes at cell `[1,1]`; the opposite-stage gaps show these are not a simultaneous stage-47/stage-48 limiter tie. The normalized matrix of the two flow-face normals and two event normals has numerical rank 4, with singular values `[1.4196891, 1.3784704, 0.2903483, 1.6474e-6]`. Each event normal is independently resolved from the two flow-face normals. The small last singular value shows the four directions are poorly conditioned relative to the largest singular value, so the evidence supports rank under the recorded tolerance, not strong separation.

The stage-48 event control hash is `39ad956e664909ed99a5232c5850a5d416e7392769f0a06932372e5ec3e09042`. Two subsequent stage-48 constrained Newton steps were accepted:

| Step | Alpha | Native J | Armijo threshold | Candidate control SHA |
|---|---:|---:|---:|---|
| 0 | `0.001953125` | `0.060119180790486296` | `0.06011918080517485` | `312603da9eddc0115b7adf49db4a2426dec2db790ee02ae2057ed2ee98585506` |
| 1 | `0.00048828125` | `0.0601191807868374` | `0.06011918079048593` | `30bed163abb770fb4cb708cabd168852d895892da05d0655bc9ec604430044f2` |

Both recorded candidate values are below their Armijo thresholds and below the preceding point's native J. The final accepted control bytes reproduce the stored control hash; the terminal native-J record, final control hash, terminal branch audit, and selected-branch curvature all refer to that same control.

## Why continuation stopped and what remains unresolved

The terminal point still has the stage-48 x-cell `[1,1]` near-tie and is classified as the two flow faces plus a limiter tie. Its branch record lists `unique_near_tie_slope_entities: ["x:1:1"]`; accordingly `first_order_complete=false`, `P2_complete=false`, and `minimum_claim=false`. The event-constrained gradient test is not met.

There were 32 rejected trial records. Each reports an unsupported intermediate y-limiter selector transition at cell `[2,3]`, stages 123 or 124 (`right → inactive zero`): 29 stage-123 and 3 stage-124. Therefore the stopping status should be read as the bounded search refusing those changed active sets, not as evidence that the native objective cannot decrease beyond the final point.

No forecast/score calls were made (`score_calls=0`); top-level `minimum_claim` and `response_claim` are both false. The active two-face parameter-response base gate remains unmet because the current receipt does not carry same-point first-order qualification or four-sector P2. See the sibling `RESPONSE_READINESS.md` for that gate and source-hash handoff.

## Resource and artifact closure

The run completed in `98.85294162482023` seconds under its newly declared 900-second phase. Counts were 82/82 direct-J calls, 25/25 gradients, 192/192 Hessian rows, and 15/15 profile forwards. Cumulative tracked direct-J and gradient counts are 594 and 159 respectively; the result does not claim completeness of earlier Hessian/profile totals.

The frozen plan SHA256 matches its sidecar (`9eec4a0d16335df8614a843ab2c0ee7a0e40c53457a35e0c3a9b71bcc7ec689c`). All 117 source pins and five input pins were rechecked against disk, and all 117 files in `source.tar.gz` match the plan hashes. The result reports `source_unchanged=true` and `inputs_unchanged=true`. Its transport transition records `f6c680da…` → `51721e91…`, the current observer API source.

The evidence supports successful event localization and two actual Armijo decreases. It does not support limiter-event stationarity, terminal P2, a two-face minimum qualification, or parameter-response readiness.

## Supplement — replay closure and conditional event status

A separate one-time terminal check compared the four stored retained (`replay=False`) flux-sector gradients with fresh gradients through the original native (`replay=True`) objective at the same `30bed163…` terminal control. `TERMINAL_REPLAY_CHECK.json` records 4/4 comparisons passing; the maximum infinity-norm difference is `4.343747583845925e-15` against each row's component-operation budget. The check used 4 additional gradients in `3.9994475829880685` seconds, with 0 direct-J, Hessian, and score calls. It left the raw attempt result unchanged. This closes the replay/retained gradient implementation gap at that point; the native minmod tie still uses its selected AD derivative, so it does not supply the missing one-sided limiter-sector gradients or stationarity/P2 evidence.

The current stage-48 event is a **conditional search equality**. The rank-four normal audit establishes numerical independence of the two face normals and two event-gap normals, but it does not show that a native minimizer should remain on the stage-48 tie. The search did not establish event-outward multiplier weights or a one-sided event-sector Clarke/KKT condition. Stage 47 is inactive at the selected point, and the final branch still has the stage-48 x-cell `[1,1]` near-tie. The event's optimal active status therefore remains unverified.

The checker executed bytes are preserved in `terminal_replay_executed_source.tar.gz` with SHA256 `1e8bef974658dc540f50cccc1f2df1cf57f35b93e0e54b7e74a1fa7a1bf6b750`. Its current file differs only by a `typing.Any` import and the corresponding result-dictionary annotation, as recorded in `TERMINAL_REPLAY_SOURCE_TRANSITION.json`; no calculation was repeated after that annotation edit. The raw plan/result remains pinned and unchanged.
