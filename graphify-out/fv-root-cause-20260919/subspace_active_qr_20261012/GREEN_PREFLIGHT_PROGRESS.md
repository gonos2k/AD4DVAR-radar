# GREEN preflight progress: cached-gradient J/R continuation

The cache side is closed. `cached_operators.admit()` checks the pinned PR302
plan, manifest, source and receipt bundles, all 222 archived source members,
the 291 inherited archive hashes, the exact five receipt members, and the
closed 7839 snapshot. Current-source drift is limited to the two task-authorized
search files, `fv_point_3h_subspace_model.py` and
`fv_point_3h_netflux_continuation.py`. The admitted expansion receipt records
24/24 row VJPs, one GN solve, four historical HVPs, zero candidate evaluations,
zero commits, and 79.7018435 seconds of cache cost.

The continuation driver performs a fresh native side pair at every point and
checks the accepted P2 before proceeding. At point zero, it reuses the cached
pair only after matching the 7839 control, fresh gradient direction, full
residual, theta, input anchor, runtime anchor, and later the cached `DGd`
against the rebuilt residual model. For point indices after zero, the cache hook
is not called; the driver computes and records two fresh side HVPs before model
closure. Each accepted candidate is repeated and checked against fresh full
`G`. The driver has no GN preparation path and performs no future-score work.

The alpha-contraction path now passes the charted candidate control to the
backend and carries the actual contracted alpha into the evaluation facts,
receipt key, selected trial, and proposal. The four new bounded tests cover
initial-only cache use, fresh HVPs at a changed point, cached-operator refusal
on direction or `DGd` drift, and contracted-alpha receipt linkage. The saved
test log reports these tests passing; two older endpoint tests fail earlier in
their legacy live ancestry loader because the continuation source hash changed.
The new progress driver avoids that stale chain through the pinned closed
endpoint snapshot.

`current_model_audit.json` now labels every offline ray sample by direction.
The baseline tangent model gives predicted `R=0.002539063767` at one base
radius and `R=0.002532303213` at two base radii. The GN ray is also labeled and
does not enter the no-GN coefficient gate.

## Frozen plan and guarded-run closure

The frozen plan now closes with SHA-256
`5410fd4ae986d07a8b9a1d112c106fd79acdc6d228d18153d6adb24f361eb866`; its
source bundle is pinned by
`0166b2a66e55c4a3e182b96f24a79f4d903147dd1d5076f4871b30d1fa281b3a`. I
independently verified all 228 source members against the source tar, every
current source hash against the frozen plan, and all 296 archive hashes. The
only declared source transitions are the subspace solver and continuation
driver, and both current bytes match their planned `after` hashes. The driver's
read-only `load_plan()` also passed.

The guarded receipt chain is closed: the run exited 0 with no resource
termination, monitor error, or SIGTERM; its child SHA matches `step.json`, and
its embedded resource record matches `step.resource.json`. Nine of nine
accepted steps have passing previous-P2 and minimum-mixture P2 checks, no face
release requirement, a passing fresh full-G check, and a selected candidate
receipt whose hash and actual alpha, objective, F-squared, and theta match the
proposal and repeated endpoint. The accepted control hashes form a continuous
prefix. Source, runtime, and fixed input anchors remain unchanged. The final
current control, full G, and theta equal the last confirmed endpoint.

The cache boundary also closes in the run: point zero reused exactly two
cached side HVPs after fresh base qualification and passed its cached-DGd/model
check. Each of points one through eight completed two fresh side HVPs. Totals
are nine commits, nine candidate evaluations, sixteen completed fresh HVPs,
and one additional HVP start at point nine. That final start stopped at the
precheck before the JVP; it produced no HVP result, candidate, or commit.

From the qualified base through nine commits, J changed from
`0.061001393469310765` to `0.06099005811764152`, and R changed from
`0.002548715176361377` to `0.002441269566693863`. These are bounded
original-J/R progress results, not a stationarity, minimum, response, forecast,
or convergence claim.

The internal deadline was reached at point nine. The child elapsed time plus
charged cache cost is `607.1600788` seconds; guarded elapsed time plus cache is
`611.6366405` seconds. This is `11.6366405` seconds over the 600-second internal
budget, while remaining `48.3633595` seconds under the 660-second outer cap.
The recorded precheck timeout means no JVP began at the final partial point.
Peak sampled RSS was `497696768` bytes under the 2 GiB guard.

The saved test record reports 54 passed and two legacy endpoint ancestry
refusals: those older frozen loaders stop on the intentionally changed
continuation source hash. They do not validate the new snapshot-based path; no
tests were skipped or hashes substituted. The new cache/alpha tests and the
27-case QR/rank suite passed. This compatibility limit remains explicit.
