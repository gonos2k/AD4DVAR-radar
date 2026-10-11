# GREEN audit: PR285 interrupted-search receipt fix

## Disposition

No actionable correctness issue found in the saved progress helper, continuation adapter, tests, or plan preflight. This change adds resumable search receipts without altering the frozen three-step producer or its full-R calculation.

## Checked

- The prospective plan preflight passes with a closed parent, 165 source pins, and 228 archive pins. Its source snapshots include the new progress adapter, helper, and tests; the earlier producer is not a new source pin and remains unchanged.
- `fv_point_3h_search_progress.search_candidates` uses the same `candidate_alphas`, chart-candidate callback, radius/zero-movement guards, evaluator callback, and acceptance-key tuple as the previous prepared search. Actual J/R and Armijo bounds remain in the caller’s evaluator; the progress helper does not recalculate or replace them.
- Each slot is persisted as started before chart construction. Evaluator start/completion counts are separate from slot counts, so chart rejection/failure is not misreported as an objective evaluation. Timeout/failure receipts retain started versus completed states and propagate the original exception.
- The continuation callback recomputes total slots as `slots_before + active slots started`, then the completed return path records `slots_before + evaluated_count`; it does not add the same slots twice. Candidate-trial receipts update without changing the confirmed-control/P2 prefix.
- Focused progress tests report 8 passed; basedpyright reports 0 errors, warnings, or notes. Tests cover interrupted evaluation, chart failure, first-pass parity with the frozen search, and aggregate slot accounting.

No FV, seed, optimizer, or production operator was run for this review.
