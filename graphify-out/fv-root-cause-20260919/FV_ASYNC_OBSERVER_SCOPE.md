# G9 asyncio observer ownership

The minmod-stage and PCG diagnostic observers use `ContextVar`. Two sibling
`asyncio` tasks that each enter their own observer scope keep separate stage
and PCG records, with unchanged small FV-step and diagonal-SPD results. An
ordinary call in the parent task while both scopes are active enters neither
collector.

The earlier implementation had a lifetime gap: `asyncio.create_task()` made
inside a live observer scope inherited its callback, and a delayed child
could append a stage or PCG record after the parent had exited the scope.
The pre-fix reproducer in `async_observer_pref_fix_repro.json` uses the exact
`main 918e1c0` source and records `[1.0, 2.0]` for both stage and PCG
collectors despite the parent scope ending before the `2.0` work began.
The observer closures now check whether their registering scope remains
active. After exit, newly starting calls inherited from that scope skip its
collector, while nested dispatch can still reach an
outer observer whose scope remains active. The same boundary is tested with
an `asyncio.to_thread()` worker delayed until after scope exit.

This controls **diagnostic ownership**, not the numerical computation. No
transport formula, PCG recurrence, tolerance, objective, or AD operation was
changed. Tasks created within a live scope still inherit and intentionally
share its collector while that scope is active. Services that require
independent request records must register observers inside each request task
and await child work before publishing the report. A call already in flight
when its scope ends is not cancelled by the observer guard; final records
must still be finalized only after the associated work completes.

The affected async, stage-observer, PCG-observer, local-response and
local-refinement tests passed **51/51** with 18 existing TorchScript warnings.
Pinned basedpyright 1.39.9 reported **0 errors, 0 warnings, 0 notes** for the
two changed modules and new test. These are small-task and algebraic checks;
the 4×5/8×10 process-isolated run in `FV_PROCESS_ISOLATED_RESPONSE_RESULTS.md`
was not repeated after this diagnostic-only change. Simultaneous in-process
forward-AD FV responses remain unsupported, and no async service or
concurrent GN/refinement run is certified here.
