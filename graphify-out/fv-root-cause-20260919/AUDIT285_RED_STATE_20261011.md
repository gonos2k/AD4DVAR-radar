# RED state audit: PR285 GN continuation — 2026-10-11

## Disposition

No P1 state-loss or commit-order defect was found in the completed PR285 run. The three saved steps form a valid P2-closed prefix; atomic commit writes advance current/last-confirmed control and count together, and the failure tests preserve the prior file when P2 or the atomic write fails. No FV or production work was run for this audit.

## P2 finding: interrupted search can undercount candidate work

In [fv_point_3h_gn_continuation.py](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_gn_continuation.py:350), `evaluate` checks the deadline and may raise while `prepared._search_candidates` is still collecting a point’s local `trials`. The shared record is updated with `candidate_slots_used`, `trials`, and `search` only after that call returns ([lines 381–388](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_gn_continuation.py:381)). If an internal timeout interrupts a later point’s search after some trial evaluations, the timeout handler finishes with the last P2 prefix, but the interrupted point’s attempted candidate slots and partial trial receipts are absent. The 3×24 structure still bounds total work, and prior P2 commits remain intact; the issue is that a timeout receipt can underreport candidate evaluations. The tests cover P2/write-failure prefix preservation and commit cap, but not interruption during a partially evaluated search ([test file](/Users/yhlee/ADVAR/tests/test_fv_point_3h_gn_continuation.py:52)).

## Actual PR285 run

The completed run is unaffected: all three per-point searches returned, with 9, 10, and 11 evaluations (30 total), then each selected endpoint passed P2 before commit. The final state has three closed steps, 72/72 row VJPs, 3/3 solves, 6/6 HVPs, 3 optimizer steps, and cap status `gn_continuation_cap_reached`; no fourth point or readiness operators ran. Source, input, runtime, and deadline closures are true. The saved resource receipt reports exit 0, `331.8552 s`, and sampled peak RSS `476,413,952` bytes. This cap outcome is not a stationarity, minimum, response, or forecast claim.

Plan SHA `319bb7798348643c0198fd09ae9bbe19ddfb88ae7bcfc94c50a80745b29c2d65` pins 164 source and 228 archive files. The raw/gzip/run/resource hashes and lossless round-trip match the archive manifest. Prelaunch checks report 14 tests passed and zero type errors/warnings.
