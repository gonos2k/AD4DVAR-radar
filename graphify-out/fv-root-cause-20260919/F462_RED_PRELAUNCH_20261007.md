# f462 inexact continuation — RED prelaunch review

Date: 2026-10-07
Plan SHA256: `499e89862d86c8e4258e5199194674714c9001c59fca00e5436c935d98a3e33b`
Scope: final read-only review of source pins, selected-endpoint receipt closure, policy, counters, and metadata-only loader integration. No FV, HVP, PCG, forecast, adjoint, reanalysis, or test suite was run in this review.

## Disposition: GO for the one declared guarded continuation

The earlier RED blocker is resolved in the current snapshot. The plan points to the comparison parent `step.json`, whose `step.run.json` and `step.resource.json` sidecars exist and are pinned. The adapter verifies the completed inexact comparison selector and closes the selected arm's endpoint data against the top-level endpoint before passing it to the generic continuation. The selected control is f462 (`f462a496…`); the strict arm is not eligible as a fallback.

I independently loaded the exact plan through `_validated_loader()` using `.venv/bin/python`. This was a metadata-only load: it checked all source/archive pins, receipts and endpoint metadata, and reconstructed the cached preconditioner; it did not evaluate the FV objective or perform HVP/PCG. It returned the declared f462 start, `selected_arm=inexact`, `comparison_complete=true`, inexact mode, and the expected policy. Plan SHA was `499e8986…`; the plan contains 104 source pins and 28 archive pins, and all listed digests matched the current files during this review.

The comparison top-level raw, parent, and resource receipts close: parent child SHA matches `step.json`; parent reports one completed accepted iteration and 53 aggregate HVPs; resource reports exit 0, 695.530 seconds under the 780-second limit, sampled RSS 373,604,352 bytes under 1 GiB, and no resource termination, SIGTERM, or monitor error. The inexact arm delta is separately recorded as 22 HVP, 20 completed PCG iterations, and one solve. The top-level 53 HVP / 49 PCG figures are the paired comparison totals, not the selected inexact arm's incremental cost.

## New-run policy and accounting

The plan pins exactly the existing bounded inexact policy: at most 3 iterations, 90 HVPs, PCG maximum 40, 720-second internal limit, 780-second outer limit, 1 GiB sampled RSS, one guarded launch, original dual-merit gates, and final root threshold `||g||∞ <= 1e-10`. It states that the new run's counters start at zero and prior partial solve products are not reused.

The wrapper calls the generic continuation without `shared_counts` or an inherited absolute deadline, so a fresh invocation creates fresh counters and the ordinary 720-second internal deadline. At every iteration the runner reconstructs J, full gradient, strict branch/margins, and a current-point HVP direction. It records the forcing tolerance from that iteration's gradient. f462 starts with gradient infinity norm about 0.416619, so its first eta is the 1e-3 ceiling; the plan correctly says the later tightening zone below 1e-2 is still unverified. The f82c Hessian remains the preconditioner only. I found no automatic original-J-only fallback or change to the prior or final root tolerance.

The saved focused verification log reports 70 passed with 18 existing warnings, and the saved type-check log reports zero errors/warnings/notes. These are existing records; I did not rerun them. The selector regression covers non-inexact selection, incomplete comparison, and an endpoint hash mismatch. The actual metadata-only loader integration passed independently in this review.

## Remaining limits

This is prelaunch authorization for one bounded research continuation under the pinned plan; it does not predict that the 3-step/HVP/time budget will suffice. Reaching `||g||∞ < 1e-2` and observing eta tightening remain unverified. The current endpoint is still far from the final `1e-10` stationarity threshold. No result from this run can establish final curvature, adjoint validity, reanalysis, or forecast skill. Preserve budget refusal, numerical refusal, and external resource termination as distinct outcomes; do not retry within the same declared launch or reinterpret old counters as new-run usage.
