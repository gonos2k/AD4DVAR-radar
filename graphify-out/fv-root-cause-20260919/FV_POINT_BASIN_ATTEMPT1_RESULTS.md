# R2 point-objective basin search attempt 1: strict-search refusal

One predeclared branch-aware L-BFGS search used the same fixed 4×5,
full-valid, correlated point objective and warm control as the successful
no-solver preflight. It ran in a guarded child with a 1200-second wall and
sampled 1-GiB RSS cap. The child exited 2 with
`numerical_status=basin_incomplete` and
`basin_status=line_search_refused`; the parent recorded
`execution_status=completed`. Actual child time was 6.465 seconds, with 25
RSS samples and sampled peak 313,622,528 B. Source, input, preflight and
plan hashes were unchanged.

The objective decreased from **0.07993508078523576** to
**0.0064467507825562076** in 13 accepted steps. At the last accepted
point, `||grad_c J||_inf=0.28903172116088516`, so it did **not** reach the
predeclared `1e-4` basin handoff gate, much less the final `1e-10`
stationarity gate. The 69 trial evaluations comprise 13 accepted steps, 52
candidate branch-gate refusals and four finite Armijo/roundoff refusals.
No accepted step changed the recorded limiter/face-sign signature.

At iteration 14, all 16 permitted backtracks were refused by the combined
stage/slope/face gate before objective merit evaluation. The last accepted
point had scaled slope margin `0.00010001667666401268`, close to the search
gate `1e-4`, while its face-flux margin was `0.00835141555813997`. The
current refusal record does **not** separately identify which gate first
failed at each of those 16 candidates, so proximity is a diagnostic clue,
not proof that every refusal was due to the slope margin.

No candidate SPD Hessian audit, exact Newton polish, adjoint or signed
reanalysis ran. `response_validation` remains `not_performed`. This run
does not show that a stationary point is absent. It shows that a policy
requiring the **final sensitivity margin `1e-4` at every exploratory search
candidate** could not progress from this seed beyond the observed point.

The raw child report, resource report, log and exact uncommitted search,
producer, runner and plan snapshots are preserved in
`point_basin_attempt1/`. Any next attempt must be separately predeclared
and distinguish a search-stage mathematical smoothness policy from final
response eligibility; changing the historical attempt or its threshold
after seeing this result would misstate the evidence.
