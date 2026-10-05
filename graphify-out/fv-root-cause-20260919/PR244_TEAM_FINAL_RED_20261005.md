# PR244 final bounded-step RED review

Status: the narrow source-integrity and refusal-classification fixes are accepted. This review covers only the current bounded-step producer and its toy tests; no FV/HVP or new candidate run was performed.

Current producer SHA-256: `71c4d56f0b497f63ed8f016db99aa225c98a9ad807d4d3c154c804a1364752e0`
Current test SHA-256: `3856c68c9d9abe21bdcc5a9294fb485f7b640e00343b35dd7c4bdf991196d80a`

The pre-fix source-list gap is closed: `_source_paths` now unions all validated `plan.source_files` with the required live dependencies, including the resource guard and endpoint diagnostics helper. Tests assert those dependencies are required and audited before and after execution. The block direction wrapper converts only the known `block_step.StepRefusal` into a bounded numerical refusal; unknown exceptions still propagate as execution failures. The child CLI preserves an initial typed refusal record when no report yet exists.

The current delta adds cooperative deadline checks between cache load, runtime/input reconstruction, branch replay, base J/g evaluation, and direction construction. A known `BudgetRefusal` before the report exists is saved as `step_budget_refusal` with `integrity_status=not_verified`; the parent deliberately marks execution failed rather than presenting unverified base identity as a completed numerical result. A toy regression verifies an expired setup stops before source-path or direction work.

The latest focused log records 44 tests passed, 18 existing warnings, and zero error-level type diagnostics. Historical raw evidence and the prior actual one-step result remain unchanged and retain the earlier producer/test pins `0c6ab9c3…` / `e723afef…`. The frozen historical plan still pins those earlier files, so it cannot authorize a run of the current source; any future run needs a separately reviewed plan. No new actual run occurred in this delta.
