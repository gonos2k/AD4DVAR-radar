# Proposed G3b nominal-refinement diagnostic, attempt 2

Status: **prepared, not approved or executed**. Run 1 is preserved in
`FV_PARTIAL_REANALYSIS_RUN1_RESULTS.md`. Its GN phase completed, but nominal
Newton correction refused at iteration 4 before any tangent, adjoint or signed
reanalysis. The old exception did not expose whether its final backtracks
were rejected by the fixed branch, nonfinite candidate/evaluation, or finite
Armijo decrease. This proposed attempt answers that narrower question.

Use exactly the run-1 one-lead 4×5/26-control partial problem: the same two
missing cells, 58 detected observations, fixed 0.25-dBZ common-bias error,
known state/boundaries, theta/pattern, and detached conditional verification.
Before execution, require the current preflight's fixed-problem and input
tensor SHA256 to match run 1's guarded preflight record. The source hash is
expected to differ because the diagnostic now reports candidate refusal
counts; keep both source trees separately identified.

The numerical method, objective, branch signature/margin gates, PCG tolerance,
8 Newton iterations and 16 backtracks are unchanged. A diagnostic-only mode
stops after nominal refinement even if it succeeds. No tangent, adjoint,
signed endpoint or response-validation comparison is authorized under this
plan. On line-search failure, `refine_stationary` now reports, for the final
iteration, the number of branch refusals, nonfinite candidate/evaluation
refusals, and finite Armijo refusals. Their sum must equal the attempted
backtracks. This is a reason count, not proof that another branch has no root.

Run one serial guarded preflight with its existing 120-second cap and one
serial **nominal-only** child with a **240-second wall cap and sampled 1-GiB
RSS cap**. The proposed command is:

```text
.venv/bin/python examples/weather_scenarios/fv_partial_reanalysis_runner.py --execute --nominal-only --wall-seconds 240 --directory graphify-out/fv-root-cause-20260919/partial_refinement_diagnostic_attempt2
```

If GN/refinement refuses, preserve the raw phase checkpoint, error counts,
outer exit and resource report. If it reaches a strict stationary point, stop
with `response_validation=not_performed`; do not treat this diagnostic as
G3b closure. An external cap/monitor failure likewise leaves G3b open. No
branch tolerance or objective rescaling will be selected after observing the
result. A later response experiment would need its own frozen scope and
resource decision.

The 240-second cap is based on run 1's 133.185-second child time and permits
diagnostic overhead. It is a ceiling, not a convergence guarantee. The user
approved **one** 1,200-second numerical attempt for the earlier plan; that
attempt is complete. This second child requires separate authorization.
