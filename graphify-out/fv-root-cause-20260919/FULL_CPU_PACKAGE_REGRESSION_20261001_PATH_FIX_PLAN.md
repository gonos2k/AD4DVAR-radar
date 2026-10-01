# Isolated pytest collection repair and third official run

Attempt 2, run `36813275231` on candidate
`aa56f65b4560b398076ee685a4017a12774429b3`, passed both strict dependency
audits and the complete Wheel/CLI job. The CPU job passed product types
then failed collection with 53 `ModuleNotFoundError: examples` errors.
No CPU tests executed. Preserve its job/step/log evidence independently
of attempt 1's audit failure and of the next run.

The existing `python -I -m pytest` command excludes implicit current
directory imports. Research tests deliberately import checkout-only
`examples.weather_scenarios` and fixture modules in `tests`. Reproduce
the same 53 collection errors locally under `-I`. Add only pytest's
explicit `pythonpath = ["."]` configuration beside its existing
`testpaths`; preserve Python isolation, dependency audits, workflow
commands, product/runtime dependencies and the installed CLI isolation.
The added path applies while pytest is running; it does not change the
earlier isolated pip-audit or later wheel CLI processes.

Validate isolated collection succeeds and matches every one of the
2,281 baseline test identifiers in the same order. This is collection
evidence only. GREEN/RED review the one-line configuration fix and its
isolation boundary. Commit/push the candidate, then dispatch the official
workflow once on that exact head. Inspect its full CPU test summary and
the package job independently, retaining any further failures for a
targeted correction. Do not drop tests, remove `-I`, loosen type/audit
gates or count prior package success as full CPU success.
