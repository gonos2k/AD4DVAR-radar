# RED final review: adaptive minmod event attempt 1

## Outcome

The bounded maintain/release run completed its 12-step limit in 349.11 seconds. It accepted 12 actual native-J decreases, moving from `0.0601191807868374` at the 30bed input to `0.06011917924338052` at control `9c106af45be7141fb70727e0993842c3408ba6c13a83f47d66c1c5398e16be5d` (decrease `1.5434568831e-9`). The terminal point is not first-order qualified and is not a minimum: its two-event chart hull infinity norm is `1.1755243e-7`, above `1e-10`; `first_order_complete=false`, `P2_complete=false`, and `minimum_claim=false` are correctly retained. The explicit terminal reason is `current side-by-flux limiting family is nonstationary`.

## Event transitions and Armijo evidence

The controller used event identities from the current full replay trace, without a stage whitelist. It released and re-encountered the stage-47 x-cell `[1,1]` positive-gap event, and located stage-124 and stage-123 y-cell `[2,3]` right-operand zero events. The final active union contains stage 47 x-gap and stage 123 y-right-zero, both with positive other operands. Each of the six recorded transition roots has a single active identity in its root union; the terminal point then maintains both current constraints.

The accepted effective step fractions include root truncation (for example, iteration 0 uses raw alpha 1 with effective alpha `0.00351510`; iteration 2 uses raw alpha 1 with effective alpha `0.000344181`). Every accepted step has a negative recorded slope, a strictly lower native J, and a value below the recorded Armijo threshold computed from the effective fraction. The revised release policy checks the exact stepped direction against every selected-side event normal, retained-event tangency, and all selected-side flux-sector gradient slopes before proposing release. The saved check also confirms these slope calculations against the stored family records.

At the final same-point family, the two chart event normals are numerically independent in all 16 side/flux records; their minimum normalized singular value is about `0.999745`. Both event-side outward derivatives are positive for both events, so maintain is supported by the normal test. That result does not close tangent stationarity: the chart hull residual remains `1.18e-7`. Terminal normal-cone/product-mixture and P2 calculations were correctly skipped after this stationarity failure. No minimum claim follows from the positive normal tests or the search Hessians.

## Saved verification and limits

I reviewed `verify_saved.py` and independently reproduced its file/hash, trace-hash, native-J chain/Armijo, count-cap, prior-cancellation-scale, and exact stepped-direction checks in memory. All passed. The receipt reports 311/384 direct-J calls, 327/768 gradients, 336/2048 Hessian rows, and 131/256 profile forwards; source, input, and trace hashes close. No FV/objective/gradient/Hessian callback was run for this review.

Transition location is supported by actual trace samples at the proposed endpoint and fractions 0.25, 0.5, and 0.75, followed by bisection of the first changed sampled interval and an active-union check at the root. This is a sampled path guarantee, not a global proof that no event crossed and returned between samples. For example, a scalar event residual `h(t)=(t-0.1)(t-0.2)` has the same sign at `t=0`, `0.25`, `0.5`, `0.75`, and `1` while crossing zero twice between samples. The attempt makes no global first-transition claim; preserve that limit in any summary. A stronger claim would need interval bounds or denser adaptive event checks.

## RED disposition

The attempt supplies bounded native-J progress through the currently observed event family and stops with a precise nonstationary terminal reason. It does not establish a local minimum, a complete P2 result, or exhaustive event ordering along every line segment. Keep the terminal claim gates as recorded and describe event detection as sampled/adaptive.

## Addendum: packed trace archive

The 311 trace paths referenced by the saved evaluations are all present in `TRACES.tar.gz`: 311 members, no missing or extra members, and every decompressed member hash matches its recorded trace digest. The plan sidecar and source/input closure remain valid. `VERIFICATION.json` binds to the unchanged `RESULT.json` SHA256 (`f29020c8…`) and reports all checks passed. `verify_saved.py` now reads trace bytes from the archive; this packaging change adds no model evidence and required no physics run.
