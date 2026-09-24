# G3b final-backtrack branch-gate detail, attempt 3

Status: approved for autonomous checklist investigation by the user's
2026-09-24 19:02 JST instruction; **not yet executed** at plan freeze.
Attempts 1 and 2 are preserved. Attempt 2 found 16/16 final Newton
backtracks refused by the branch callback, with no nonfinite candidate and
no finite Armijo evaluation. It did not distinguish strict tracer failure,
quantitative slope/face margin, or exact GN-signature mismatch.

Use the same one-lead 4×5 partial problem, two missing cells, fixed
common-bias whitener, warm start and synthetic verification. The current
preflight must match attempt 2 on fixed-problem and input tensor identities.
Do not change the objective, branch gates, Newton/PCG tolerances, 8×16
iteration/backtrack policy, or physical operator. The sole numerical-source
change is that `refine_stationary` appends a count of each branch callback's
raw `ValueError` message to its existing line-search failure message. Tests
show the branch and nonfinite categories without running FV optimization.

Execute **one** `--nominal-only` guarded child, with a 120-second preflight
cap, a 240-second numerical wall cap, and sampled 1-GiB RSS cap:

```text
.venv/bin/python examples/weather_scenarios/fv_partial_reanalysis_runner.py --execute --nominal-only --wall-seconds 240 --directory graphify-out/fv-root-cause-20260919/partial_branch_gate_detail_attempt3
```

If it refuses, report the raw reason counts and preserve the checkpoint,
outer exit and resource record. If it unexpectedly reaches a strict
stationary point, stop there with `response_validation=not_performed`.
No tangent, adjoint or signed endpoint is part of this diagnosis. A gate
change would require a separate mathematical justification and experiment;
in particular, do not simply remove the GN signature lock and reuse a
branch-local Armijo slope across a minmod kink. G3b remains open until
a newly qualified nominal point and independent response comparison exist.
