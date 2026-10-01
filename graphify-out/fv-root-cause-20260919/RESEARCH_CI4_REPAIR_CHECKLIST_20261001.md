# Research repairs from official CPU attempt 4

User steering: research use, not deployment; reduce unnecessary CI. No new full CPU, wheel/CLI, or deployment run has been launched after this direction.

Official run `36836402831` at `a9e321d` executed all six Linux shards. All six selected inventories share the same digest and runtime; their disjoint union covers all 2,311 node IDs. Pytest reported 87 failed/error node IDs. The old logger additionally marked legitimate subtest traces invalid. This run remains **failed**, while its Wheel/CLI and UI jobs passed. Local repairs do not retroactively change it.

| Finding | Applied repair | Local evidence | Limit |
|---|---|---|---|
| Subtest call reports treated as repeated parent calls | Record actual SubtestReport instances separately, with context, skips/xfails, failure text and sections; aggregate failed subtests into failed parents | 32 focused protocol tests; GREEN review | No whole Linux rerun |
| Synthetic runner/lifecycle tests entered a host-specific archive certificate gate | Host-local or synthetic fixtures; archived controls are data, never rebound response certificates; separate production drift rejection | Response protocol group 111 passed; face/sector group 59 passed; field audit 7 passed; GREEN/RED reviewed | These are protocol checks, not new FV analyses |
| Composite identity byte hashes depended on host reductions | Compare old/current definitions with identical current inputs; retain archived tensor/source hash cross-checks and mutation sensitivity | Portable identity group 23 passed; original-input/archive controller group 36 passed; merit error path 11 passed | Historical reports remain unchanged |
| Algebraically equivalent score means failed bit equality | Relative roundoff allowance 2*N*eps on nonnegative reductions, zero absolute floor; stronger unequal-lead checks retained | Two affected tests passed; GREEN/RED math review | No score or physics change |
| Archived support summaries required obsolete source bytes to equal current source | Hash-pinned historical snapshots keyed by exact path/hash; mark current_source_match=false where needed | Support matrix 8 passed; snapshot bytes checked against historical Git blobs; GREEN/RED | Historical proof is not current model certification |
| Strong theta direct/indirect cancellation magnified PCG stopping error | Explicit adjoint_relative_tolerance in (0,1e-10], default unchanged; tight cross-check requests1e-12; fresh residual budget check and zero-RHS path retained | Same-point controlled diagnostic; 14 guarded affected tests in96.66s,339,787,776B sampled RSS; GREEN/RED | CPU FP64 research response; FD thresholds unchanged |
| Required CPU check name lost in matrix naming | Lightweight opt-in aggregate preserving original required context; repository protection unchanged | actionlint passed | No extra full CI dispatch |

A final local integration check for support/identity/source bindings passed 36 tests (18 existing warnings). Product sensitivity typecheck reports zero diagnostics. These test groups overlap and are not summed into a full-suite total.

Graphify consulted the existing graph and extracted only 22 changed code files (409 nodes/1,044 edges) into `LOCAL_RESEARCH_FIXES_AST_20261001.json`. The shared cache was restored byte-for-byte when preexisting duplicate raw multigraph keys were found to collapse during parsing; no forced rebuild or semantic extraction occurred. Current source-only AST evidence remains available; the cache integrity limit is explicit.

R2-O-R, R4-R, R5-R-R and production/general physical-input validation remain open. No new root, sensitivity reanalysis, physical skill or deployment acceptance is inferred from these engineering repairs.
