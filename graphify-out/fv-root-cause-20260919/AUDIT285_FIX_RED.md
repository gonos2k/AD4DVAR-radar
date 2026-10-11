# PR285 progress-accounting fix — RED review

Disposition: no actionable state or accounting defect found in the prospective revision.

- `fv_point_3h_search_progress.py:21-32` records each started grid slot before chart/evaluation work. It separately tracks evaluator starts/completions (`:46-49`) and closes normal refusals/acceptances (`:56-59`). An evaluator timeout is persisted as an interrupted slot and re-raised (`:60-65`), retaining earlier trial receipts.
- `fv_point_3h_gn_continuation_progress.py:382-397` persists the cumulative budget as the prior total plus this search's `slots_started`; its final summary recomputes the same value from `evaluated_count` (the same started-slot count), rather than adding twice. The global candidate cap remains enforced.
- Search progress writes candidate/search fields only. The current/last-confirmed controls and `closed_steps` move only through `_commit_closed_point` after `fresh_final_closure` (`fv_point_3h_gn_continuation_progress.py:163-180`, `:413-430`); an interrupted prospective search therefore leaves the confirmed P2 prefix intact.
- Focused tests cover a timeout on the third evaluation after two refused evaluations, a chart failure before evaluator invocation, first-pass parity with frozen search, and cumulative slot accounting (`tests/test_fv_point_3h_gn_continuation_progress.py:25-78`). I ran the focused file: **4 passed**. No FV/production run was performed for this review.
- The change is prospective and pinned by `GN_CONTINUATION_PROGRESS_PLAN_20261011.json`; the existing PR285 producer and its saved run were not modified by this review. Candidate accounting distinguishes grid slots from actual evaluator calls, so chart-only refusals are not represented as expensive evaluations.

Remaining limit: tests exercise progress callbacks with in-memory receipts, not a full child-process timeout plus on-disk receipt recovery. The production callback uses the same helper state and atomic JSON writer; this is a coverage gap, not an observed correctness failure. No optimizer-convergence or model-validity claim follows from these accounting tests.
