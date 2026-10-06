# PR249 records and execution audit

**Result:** one actionable parent-receipt defect reproduced with a mocked child; one durable HVP-count gap on unexpected child exceptions. The recorded successful run's HVP accounting, input/source pins, and execution/numerical status separation otherwise agree. No live FV/HVP run was started for this audit.

## Findings

### R1 — malformed child JSON object can leave a completed parent receipt

At [fv_point_3h_hvp_newton_step.py:315](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_hvp_newton_step.py:315), the parent indexes `child["numerical_status"]` before validating that decoded JSON is an object. The handler at line 319 catches `OSError`, `ValueError`, and `KeyError`, but not `TypeError`. A valid JSON array such as `[]` therefore raises before the second parent write. The only durable parent receipt remains the preliminary record written at lines 311–313 with `execution_status: completed`, `numerical_status: not_reached`, and no `child_read_error`.

**Reproduction:** a toy harness monkeypatched `load_base` and `run_guarded_diagnostic`; the fake guard returned a valid completed resource record and wrote `[]` to `step.json`. `main()` raised `TypeError: list indices must be integers or slices, not str`; `step.run.json` stayed marked completed/not-reached. No objective, FV trajectory, or HVP was evaluated. Validate `isinstance(child, dict)` inside the guarded parse path and persist the failure; add the malformed-child regression at [test_fv_point_3h_hvp_newton_step.py:43](/Users/yhlee/ADVAR/tests/test_fv_point_3h_hvp_newton_step.py:43).

### R2 — unexpected HVP/PCG exception loses the in-memory HVP counters

The counted operator updates `hvp_calls` and `hvp_calls_completed` in a `finally` at [fv_point_3h_hvp_newton_step.py:209](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_hvp_newton_step.py:209), but that path does not write the record. The next write is only reached after `solve_newton_direction` returns at line 222. If the operator or PCG raises an unexpected programming/runtime exception, the durable `step.json` remains the `linear_solve` checkpoint written at line 196, with both counters still zero; the parent can correctly report a failed process while its child receipt exposes stale numerical status and counts. Persist the counter snapshot on exception before propagating, or through a finalization path that preserves the original exception and records failure status. A mocked operator exception should distinguish started from completed calls.

## Evidence checks

- The completed `step.json` reports 24 PCG iterations and 26 started/26 completed *new live* HVPs. These counts are consistent with the implementation: 24 Krylov operator applications, one fresh-residual product inside PCG (`matrix_free.py:468–479`), and one caller-side true-residual product (`fv_point_3h_schur_newton_step.py:113–115`).
- The archived f82c curvature separately reports 26 explicit Hessian columns and 27 total HVP calls. `load_base` requires both values and checks the archived Hessian against the completed checkpoint (`fv_point_3h_hvp_newton_step.py:107–119`). That matrix is used only as the block inverse preconditioner; the operator being solved at a35 is the live original-J Hessian action.
- The successful attempt's child and parent records agree: child `phase=finished`, `numerical_status=one_live_HVP_original_J_step_accepted`, one optimizer step, fixed input and sources unchanged; parent `execution_status=completed`, zero exit, no child read error, 219.13 s elapsed, and sampled peak RSS 370,491,392 bytes. The resource receipt calls RSS a sampled guard, not a hard allocation limit.
- The one accepted candidate passes the strict point/Armijo gate and lowers J from 0.0716149827 to 0.0668157467. Phi rises from 1.1129736171 to 1.8573840876 and gradient infinity norm rises from 0.7980151516 to 0.9880409692. The result explicitly leaves stationarity, new-point curvature, response/score, physical validation, and global SPD unclaimed; the one-step result is not evidence of convergence or forecast skill.
- Hashes verified against the current workspace: all 88 plan source pins, all 7 plan archive pins, and all 19 entries in `NEWA35_HVP_MANIFEST_20261006.json` match. The source and archive closure covers the Newton caller, solver/objective dependencies, focused tests, accepted f82c step/curvature, their parent/resource receipts, and Hessian checkpoint.
- Existing test/type records say 44 focused tests passed and the type check reports 0 errors, warnings, or notes. Those are recorded evidence, not rerun in this records-only audit.

## Limits

The completed run is internally consistent at the artifact level. This audit did not independently recompute the FV objective/HVP, verify stationarity, qualify the Hessian at the accepted new point, or validate forecast skill. The two failure-record paths above need targeted toy regressions and a durable receipt fix before relying on receipts for malformed child documents or unexpected solver exceptions.
