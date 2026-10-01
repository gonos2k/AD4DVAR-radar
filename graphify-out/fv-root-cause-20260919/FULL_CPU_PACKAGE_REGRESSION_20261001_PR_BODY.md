The full CPU job exceeded its two-hour limit after 73% progress, and synthetic research fixtures assumed an archived host runtime. This change runs the complete suite in six whole-module shards, verifies exact collection and per-test terminal outcomes, and separates portable test vectors from strict production archive certification. It also fixes checkout imports under isolated pytest and updates the two CI lock blocks for urllib3 2.8.0 so strict audits run.

All audit, typecheck, install, CPU-thread, package/CLI and resource gates remain intact. Collection skips, missing test calls, mismatched assignments and incomplete worker records cannot count as full coverage. Source/runtime-bound records are retained for reconciliation.

Validation: 29 focused protocol tests; fixture checks 29 native/29 metadata-simulated; native isolated collection of 2,311 IDs with exact disjoint partition; actionlint passed. Targeted types: 0 errors, 343 lower-severity warnings. Wheel/CLI passed at the prior head; the new official Linux shard/package run is pending. V1 remains open until all terminal records pass. R4 research work is in a separate local branch.

Inherited PYTEST_ADDOPTS is removed in both isolated workers. Configured filters/ignore options and collector deselections fail closed; runtime skips and xfail remain explicit. The prefilter candidate is preserved as superseded evidence.

Runtime skip reasons are retained and verified from phase reports.
