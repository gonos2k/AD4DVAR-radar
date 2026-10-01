# Full CPU regression: reviewed shard candidate

Official attempt 3 (`36813848349`, head `39156508`) is incomplete. GitHub reports cancellation and an explicit two-hour maximum-time annotation; the raw test log ends at 73%, without terminal pytest counts or failure tracebacks. Wheel/CLI and UI completed successfully at that head. Historical audit and collection failures remain separate evidence.

The candidate retains the hashed CPU closure, strict isolated audit, package install/check, product types, CPU thread settings, and 120-minute ceiling. It partitions the complete freshly collected inventory by whole modules into six deterministic jobs with fail-fast disabled. Every worker re-collects exactly its assigned subset before running it. Phase reports are persisted atomically; successful accounting requires setup/call/teardown evidence or an explicit valid skip/xfail lifecycle. Collection-time skips fail the complete-inventory gate. Reconciliation compares source/runtime bindings and actual assignments/outcomes against the entire inventory; missing records or hard timeouts remain incomplete.

Final post-filter-guard focused tests: 29 passed in 4.35 seconds, including real isolated collection and execution in temporary toy Git repositories. They cover pass, setup skip, xfail/XPASS and teardown xfail, import failure, collection skip, assertion failure, and pre-execution selection mismatch. No FV numerical solve was run by these helper checks. The earlier five field-guard fixture repairs remain test-only: 29 native and 29 simulated-metadata checks passed, while the real archive runtime-drift gate still rejects incompatible hosts.

Final basedpyright 1.39.9 analysis of both exact source-byte copies reports 0 errors and 343 lower-severity warnings. Copies outside the hidden `.github` directory ensure both targets were analyzed; the direct-path attempt analyzed only one file and is retained as partial evidence. Warning-only exit status uses the repository checker's existing policy. This is not a warning-free claim.

The native Mac isolated inventory contains 2,311 unique IDs in 165 modules. Its six assignment counts are [386, 385, 385, 385, 385, 385], with a disjoint exact union. The prior 2,304-ID inventory was unchanged across duplicate-conditional cleanup; six filter-guard regressions and one skip-reason regression account for the final increase to 2,311. This is local collection/partition evidence, not Linux execution or a performance-balance guarantee. Each official Linux job generates its own fresh inventory at the candidate SHA; the six inventories and terminal records must be reconciled before V1 closes.

Workflow actionlint passed. GREEN and RED reviewed the coverage and lifecycle path; the inherited-options filter gap was fixed and both worker paths rechecked. Graphify added the two new code files using bounded AST extraction: post-filter code AST and cache counts are recorded in CPU_SHARDING_GRAPHIFY_AUDIT_POSTFILTER_20261001.json. Unrelated node and link metadata remain exactly unchanged; no semantic rebuild or visualization ran.

The new official run is pending. No full CPU pass, scientific-response accuracy or physical skill is claimed from this candidate's local checks.

Current verification uses FINAL inventory/AST and RELEASE test/type artifact names; older FINAL artifacts belong to the superseded prefilter snapshot, not the latest code.

Runtime skipped phase reports retain nonempty pytest skip reasons; reconciliation rejects reasonless skipped phases while preserving the existing skip/xfail policy.
