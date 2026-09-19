# PR163 follow-up: forwarding, comparison integrity and limiter counterexamples

Baseline: merged main 5723161a. No product transport, inverse solver or exact
response gate changed. The saved 180-minute comparisons are reused, not rerun.

## Closed engineering gaps

- A 16x16 one-lead minmod regression instruments the actual finite_volume_step
  and JVP _advance calls. Omitting reconstruction in either path fails the test;
  global/row metadata must agree. Existing donorcell default tests remain.
- The comparison publisher validates report/row scheme labels, 48 km domain,
  600 s interval, all 18 evaluation times, 32/64/128 spacing, fixed echo threshold
  0.1, unique 3-flow x 3-grid coverage, matching CFL schedules, and finite displayed
  metrics. A panel-level test ensures rendering cannot bypass validation.
- Shared source hashes are bound to measured revision c192c0df. All four producer
  files were checked with git show before pinning their canonical source-map
  fingerprint e619a8bf0fdf4d8da492ca4beec84ede1cb06242ec0aa92ce998a91011889cfb.
  Validation therefore also works in offline/shallow checkouts without fetching
  historical commits. Current archived files are unchanged.

The archive predates direct initial/boundary tensor hashes. The common-input
claim is supported by the same deterministic producer revision and matched
reported cases/settings, not newly recorded tensor hashes or an attestation of
arbitrary callbacks. Future experiments should record input tensor identities
at execution rather than relabel historical evidence.

## Actual transport counterexample

The new tests call finite_volume_step(reconstruction='minmod') itself on a 5x9
CPU FP64 full-support field, fixed known-zero exterior, x-flow CFL 0.2 and
SSPRK2. Only the initial field is varied in these small cases.

| Ramp tie, central output | Result |
|---|---:|
| AD-selected JVP | 0.835 |
| Right difference quotient | 0.895 |
| Left difference quotient | 0.810 |
| JVP/VJP inner-product discrepancy | 0 |

The one-sided split persists for h=1e-4,1e-5,1e-6. The full-field split norm is
about 0.16552945. Thus a consistent transpose of the selected AD linearization
is insufficient to establish a classical two-sided derivative. These are actual
product-function checks, not a copied limiter implementation.

A quadratic positive profile has matching one-sided differences (errors below
6e-11 at h=1e-5 in the measured environment). A ramp with flow fixed identically
zero has exact identity output/JVP despite its slope ties. This does NOT mean
zero nominal flow is safe when flow is itself perturbed: the test fixes flow.
It prevents a blanket policy rejecting every structural zero/tie.

Reproduction: tests/test_fv_minmod_derivative_branches.py.
Raw values: minmod_tie_probe.json. This is a small directional counterexample,
not a general automatic branch classifier or a finite-path certificate.

## Remaining scientific checkpoints

- [ ] Track relevant sign/tie/selector states through every RK stage, accounting
  for permitted initial/flow/growth perturbations and whether flux/output can
  actually depend on each branch.
- [ ] On stable cases check those directions separately, their mixed derivatives,
  replay agreement, and two-sided Taylor residuals. Do not demand a factor-four
  ratio when the residual is already at roundoff or exactly linear.
- [ ] Only then assess a small minmod inverse problem with matched analysis,
  forecast, objective, stationarity, true adjoint residual and reanalysis.
- [ ] Typed precision/support, finite-impact tolerances and whole D7 remain open.

The existing HTML was republished through the stronger validator. Both complete
HTML files are byte-identical (comparison_publish_unchanged.json), so no new
browser/layout claim or replacement animation is made. Full CPU CI and the
240-grid experiment were not restarted. Final GREEN/RED review found no blocker
in this bounded scope.

## Verification

Final bounded run: 25 passed (6 probe/forwarding, 16 publisher, 3 branch tests),
3.60 s pytest; wrapper 5.05 s and sampled peak RSS
332611584 bytes. Limits 60 s/2 GiB, exit 0. Development
reruns overlap these cases and are not added. Log/record: minmod_branch_publisher_tests.
Graphify refreshed cached code structure only: 6530 nodes, 64812 edges,
231 communities, zero semantic LLM tokens.
