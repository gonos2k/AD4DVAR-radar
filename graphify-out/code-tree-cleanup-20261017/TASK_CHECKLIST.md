# Code-tree cleanup — 2026-10-17

- [x] Pin current main4bfe783 (PR307 merged), inventory pre-existing AGENTS.md change, create fresh cleanup branch.
- [x] Consult cached Graphify structural graph; inventory546 tracked Python modules under src/examples/tests; no duplicate complete modules.
- [x] Remove10 obsolete/test-only event operations, unused constants/import and fixed-stage metadata. Port useful tests to actual dynamic paths.
- [x] Use existing shared digest validator for four exact duplicates; retain original aliases and input/error contract. Remove two repeated imports.
- [x] Keep numerical operators/controller AST unchanged; verify actual prior source snapshot remains archived.
- [x] Preserve public reexports, pinned producer imports, legacy schemas, historical results and pre-existing untracked files.
- [x] Complete focused tests/types and record broad-test interruption/inherited historical-pin limit separately.
- [x] Incrementally refresh changed AST in Graphify; no full semantic rebuild.
- [x] GREEN/RED final review of changes and verification.
- [x] Commit/publish Draft PR308; existing automatic CI is inspected in the handoff.

This is a code cleanup. It does not progress the optimization problem, rerun archived production cases, alter budgets, certify a minimum, or update forecast/adjoint results. Current source differs from old frozen experiment source: reproduce those records from their original archived/commit sources. Never rewrite their pins to authorize cleaned code.

Results:11 Python files changed; source273 lines and all code308 lines net removed. Focused103 tests passed plus41 subtests; extra restored producer-pin test passed. Both type checks report0 errors. Four digest aliases match old validators on14 cases each;20 live numerical/control operations have exact AST equality, and native audit differs only by removed legacy metadata. No full-suite pass: the broad core attempt stopped after131 passes/114 subtests; historical paired-face pin mismatch was already present at main4bfe783.

Raw test/type/collection logs, including interrupted/exploratory failures, are preserved byte-for-byte in TEST_LOGS.tar.gz (extract in this directory). The rebuilt root graph and standard Graphify cache are updated. The temporary13MB AST delta was removed after merging; the PR records scoped source hashes/counts in KG_UPDATE.json. Loose test logs were removed only after verifying their archived bytes matched. Historical archives/results/pins remain unchanged. Reproduce their exact-source checks from their original archived sources/commit; cleaned current files correctly fail their old source-closure gate.

Draft PR308: https://github.com/gonos2k/AD4DVAR-radar/pull/308 . Base4bfe783; no merge or deployment performed.
