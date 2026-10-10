# PR274 RED state and receipt audit

## Scope

Reviewed the coupled-GN resume loader and the shared continuation/parent
closure paths, with focus on committed-prefix recovery and receipt
classification. I did not change source files during this audit; the bounded
patch reviewed below was applied separately by the root agent.

## Reproduced classifier defect and reviewed fix

The source-reachable initial row-only refusal could be rejected incorrectly.
Fresh base gradient replay allows FP64 roundoff, so the newly observed
minimum-mixture theta can differ from the carried theta. Coupled GN builds row
VJPs at that fresh working theta. The old terminal row-only check compared row
theta to carried theta and could reject a valid refusal receipt. The new check
ties it to `base_mixing_minimum.reference_carried_theta` and
`working_theta_star`, which are emitted by the current-point factory; that is
the correct state pair for this receipt.

Reproduction: the new `initial_roundoff_refusal` synthetic fixture perturbs the
replayed side gradients within their admission budget while preserving the
initial carried merit gate, then raises after the complete 24-row batch but
before any HVP or dense solve. The child correctly has zero commits, a valid
row-only refusal, and no solve counters. The regression confirms the checker
accepts the actual working theta and rejects a substituted carried theta.

The second fix defaults absent dense-solve counters to zero when matching
`len(solves)`. This is valid for the initial pre-solve row refusal; when any
modeled solve exists, a missing counter still fails because the solve-history
length is positive. The regression asserts the zero-solve state. I found no
false-positive path in these changes.

The earlier NaN-row reproduction was malformed receipt mutation. The producer
validates generated rows and serializes with `allow_nan=False`; I am not
classifying that external/corrupt-report case as a mandatory calculation fix.

## Actual PR274 RSS stop: confirmed prefix preserved

The captured child report is explicitly `phase=running` and
`execution_status=running`; it records one confirmed commit in
`last_confirmed_iterations`, and `current_control_sha256` equals
`last_confirmed_control_sha256`. The next point's Jacobian row 26 is `started`,
with 25 of 26 rows completed. The guarded parent receipt classifies the launch
as `rss_limit`, and the integration record says `terminal_execution_closed=false`
with `confirmed_prefix_steps=1`. This preserves the committed prefix without
classifying the interrupted child as a completed terminal result.

## Requested candidate gaps checked

- The parent invokes the chain checker with the pinned model name
  `robust_gn_coupled`; optional helper defaults do not affect this caller.
- Terminal row-only progress is supported only for the final unaccepted point,
  requires no HVPs, validates the full 24-row side/index set and finite row
  data, and requires dense-solve counters to match solve history.
- The coupled-GN base loader uses the accepted producer final repeat as the
  carried-theta endpoint, then the fresh base check recomputes the carried
  residual. At each modeled point, the factory separately records the current
  minimum working theta and its residual norm. No false commit was reproduced.
- The pinned PR273 endpoint's carried theta and current minimum differ, but are
  at different control states, so that comparison alone is not evidence of a
  post-commit drift bug. The reproduced source-reachable case is the initial
  gradient-replay roundoff path described above.

## Verification

Ran the focused synthetic suites:

```text
.venv/bin/python -m pytest -q \
  tests/test_fv_point_3h_tangent_coupled_gn_resume.py \
  tests/test_fv_point_3h_tangent_mixing_minimum_resume.py
10 passed (rerun after the classifier regression was added)
```

The final affected CPU suite also passed 89 tests with 18 existing warnings;
the type check reported zero errors, as recorded in the shared findings memo.

No FV seed production HVP or guarded launch was run.
