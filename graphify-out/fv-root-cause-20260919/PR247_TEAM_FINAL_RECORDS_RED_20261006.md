# PR247 final records review: RED

**Disposition: the resource-status fix closes the reported validator gap; the archived attempt remains complete. One replay constraint remains: the frozen plan no longer matches the edited checkout.** This review did not rerun the FV objective, HVP, response, or tests.

The revised `execution_status()` now requires the reviewed 300 s / 1 GiB declarations, a nonnegative integer sampled peak, and fails closed on malformed or mismatched values. An observed peak above 1 GiB or elapsed time above 300 s is `resource_limited`, even if a resource termination flag is absent. Existing cancellation, monitor, cleanup, and exit-code checks remain fail-closed. The added cases cover an over-cap peak, a mismatched wall declaration, a mismatched RSS declaration, and missing, negative, or boolean peaks. The main agent reports 51 affected tests passed and zero type errors; this RED review did not repeat those checks.

Reading the unchanged archived `audit.resource.json` and `step.resource.json` through the updated helper returns `completed` for both. The base curvature run recorded 235.878 s and 373,686,272 sampled bytes; the Newton step recorded 18.663 s and 358,252,544 sampled bytes. Each reports the 300 s / 1 GiB limits, clean exit, and no cancellation or monitor error. Existing raw, parent, resource, and scientific-result records were not edited.

The frozen `F82C_NEWTON_PLAN_20261005.json` now has four stale source pins in the working tree: `fv_point_3h_bounded_coupled_step.py`, `fv_point_3h_current_newton_step.py`, and the two associated tests (`test_f82c_curvature_experiment.py` and `test_fv_point_3h_current_newton_step.py`). Consequently, the old plan will correctly fail `load_base()` source-pin validation if used to rerun the caller from this checkout. This does not change the provenance or completion of the already archived attempt; a future execution needs a new plan pinning its current source set and must retain the old plan and attempt records as historical evidence.

Resource fields remain sampled observations at 0.25 s intervals, not a hard OS memory cap. The actual archived samples are comfortably below the declared cap.
