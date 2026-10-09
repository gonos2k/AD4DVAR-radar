# PR267 RED audit: state and error transactions

Reviewed `main` at `63610e9` on 2026-10-09. Read `AGENTS.md`. The initial investigation was read-only; the later resolution changed the runner and added a focused regression file. Checks used mocked callbacks and temporary output files. No FV/weather/guard/HVP job was launched.

## Findings

### P2 — Durable progress snapshots leave the public current-point fields stale

`_run_child_impl` initializes `current_control`, its digest, and `current_theta` to the PR266 base (`examples/weather_scenarios/fv_point_3h_tangent_continuation.py:535-546`). A successful fresh repeat updates `last_confirmed_control`, digest, theta, and count (`:722-735`). But `committed_progress` persists the iteration and counters without updating the `current_*` fields (`:748-753`). Thus, while the next potentially long iteration is running, a durable file can report one or more accepted commits and the latest `last_confirmed_*`, alongside the original base as `current_*`.

If the child is terminated externally while that later work is in progress, `_run_child` cannot run its Python exception recovery. The parent reads the partial JSON but does not reconcile its current-point fields when the guard reports failure (`:865-915`). The confirmed point itself is retained, so this is not loss of the commit; it is an internally inconsistent partial receipt that downstream readers can misinterpret.

**Mock reproduction:** serialize a running record with the archived PR266 base in `current_control`, one accepted iteration, and a different correctly hashed control/theta in `last_confirmed_*`, matching the fields written by `closed_repeat` plus `committed_progress`. The resulting snapshot has `accepted_iterations == 1`, `current_control_sha256 == BASE_CONTROL_SHA`, and `last_confirmed_control_sha256 != current_control_sha256`; current theta remains `0.3504554198817298` while confirmed theta is `0.4`. An external stop during the next iteration leaves this shape on disk.

**Resolution:** fixed in the tangent runner. Each atomic progress write now copies the confirmed control, digest, theta, and accepted count into both the `current_*` and `last_confirmed_*` fields. `tests/test_fv_point_3h_tangent_checkpoint.py::test_interrupted_child_checkpoint_matches_latest_confirmed_commit` exercises the real nested final-repeat and progress callbacks, then simulates an external stop before the final report. It verifies both field groups and counters identify the committed state.

### P3 — Exception recovery accepts unvalidated durable theta and commit count

`_run_child` trusts `last_confirmed_control_sha256 != BASE_CONTROL_SHA` as the sole signal that a commit occurred, then converts `last_confirmed_iterations` to an integer without checking that it is nonnegative, bounded, or consistent with the stored iterations. It restores `last_confirmed_theta` without checking finiteness or the required `[0,1]` domain (`:815-838`). A valid control digest therefore does not establish a valid transaction receipt.

**Mock reproduction:** monkeypatch `_run_child_impl` to write a 26-element control with its correct digest, `last_confirmed_theta=99.0`, and `last_confirmed_iterations=-4`, then raise `TimeoutError`. Before the fix, `_run_child` promoted the invalid theta/count. Recovery now requires finite theta in `[0,1]`, an integer non-boolean count in `[0,3]`, a matching control digest, and a commit/count combination that agrees. Invalid checkpoints return `numerical_status="invalid_checkpoint"`, zero top-level commit counts, and do not promote current or last-confirmed candidate fields. Parametrized regressions cover out-of-range theta, negative count, and boolean count. The report retains the base digest; it includes base control/theta only when the stored control can be verified against the base digest.

## Checks that held

- `.venv/bin/python -m pytest -q tests/test_fv_point_3h_tangent_continuation.py tests/test_fv_point_3h_tangent_checkpoint.py`: **14 passed**, 18 existing TorchScript deprecation warnings.
- Graphify AST extraction incrementally refreshed the runner and new checkpoint regression: 35 nodes and 156 edges, recorded in `TANGENT_CHECKPOINT_GRAPHIFY_INCREMENTAL_20261009.json`. This was a targeted extraction, not a shared full-graph rebuild.
- Existing mocked continuation test confirms three accepted corrections use exactly six HVP calls, with durable progress callbacks at counts `[1, 2, 3]` (`tests/test_fv_point_3h_tangent_continuation.py:154-193`). The shared callback rejects calls after the plan limit (`fv_point_3h_nonsmooth_coupled_probe.py:496-513`), and the loop hard caps accepted corrections at three (`tangent_continuation.py:302-303`).
- Existing failure cases confirm a failed fresh final repeat leaves the earlier confirmed state intact and a current-products timeout on iteration two preserves the first commit (`tests/test_fv_point_3h_tangent_continuation.py:196-268`). `_run_child` also preserves a valid two-commit receipt on a Python `TimeoutError` (`:270-291`).
- Mocked candidate rejection evaluated all 16 allowed backtracking candidates and returned no acceptance when `branch_pair_passed` was false. Mocked final-repeat checks rejected failures in native objective, F-squared, trace, source, and deadline facts. In production, `fresh_final_closure` additionally compares both repeated side-gradient vectors with the proposal (`tangent_continuation.py:257-282`).
- Plan loading pins the exact plan path/digest, inherited producer sources/archives, and frozen tangent source snapshots (`:379-437`). Final repeat checks proposal objective/F, both side gradients, branch traces, fixed input, runtime, source, and deadline (`:687-720`); a failed repeat does not advance `closure_state` (`:722-735`).
- Investigated the root's chart-backtracking concern. The actual chart uses the max-derivative pivot, eta zero, and a 0.05 displacement cap. At the base, the distance in pivot argument to its real-domain boundary is at least half the pivot derivative; the four retained-coordinate derivatives are each no larger than that pivot derivative. The radius cap therefore leaves margin for this fixed five-coordinate tanh chart. An arbitrary mocked `chart_candidate` can raise on its first alpha and abort the grid, but that is not a supported-path counterexample under this plan's geometry/radius, so I do not record it as an actionable PR267 defect.

## Verification limits

No external guard termination, filesystem failure, or real model calculation was exercised. The progress checkpoint regression simulates interruption with `SystemExit` after the real nested callback writes its snapshot; it verifies file state, not an OS-level kill. The malformed-receipt regressions exercise defensive recovery only, not a malformed checkpoint produced by the supported writer. Neither establishes forecast accuracy or model validity.
