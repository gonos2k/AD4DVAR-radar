# RED code preflight: three-step mixing-minimum resume — 2026-10-10

## Clearance

No actionable RED blocker remains in the frozen resume adapter/shared chain path for the planned single guarded run. This is source/receipt preflight only; no FV, seed, production gradient/HVP, test, or guard was run by RED.

Frozen plan SHA-256: `a3f24f93e84ec107c36a8888fbe2e644f9a5be3a71bff855ec219bf9be7c0bba`, with 135 source and 132 archive pins; all plan pins match the current files. Relevant frozen sources: adapter `a61000eb639d932450da7c5d120bb7663ea4a92779ac1159409bac1894f90f51`, shared continuation `32113dd56cdf9cf44b01f0b20ceb896a827b407273015d23ef11af5b3fdaa998`, resume test `f1f44e95ee997ad692ee3a50e6ef9dbe05e990ff680cbb97e40d7392278d4cf`.

The adapter validates the pinned accepted PR271 endpoint and its raw/gzip/run/resource/source/input/runtime closure before using control `e801cf…`, theta `0.48316874590279574`, and J `0.061240252230254005` as the new base. The predecessor's saved outer limit is correctly checked as 300 seconds; the new run is separately bounded to 300 seconds internal / 360 seconds outer, sampled RSS 1 GiB, at most three commits and six fresh side HVPs, with no row VJPs/PCG/dense solve.

The shared parent chain now admits up to three ordered rows and checks base-control/theta transfer, current-point HVP side/direction/working-theta labels, unique accepted nonzero commits, independent repeat closure, last-confirmed terminal state, and partial refusal preservation. Resume status distinguishes cap reached, partial progress, and zero-commit refusal. The independent saved-array analyzer recomputes each point/candidate theta-star and merit/Armijo arithmetic; this is receipt arithmetic only, not model replay. RED's analytic non-descent example remains covered by the existing merit gate: a valid interior `d=-t` can increase `Psi`, and clean refusal is correct; zero tangent similarly remains a no-commit refusal.

No claim follows beyond the planned bounded local sequence. Per-step chart radius `0.05` does not bound the total three-step path to `0.05`. Stationarity, minimum/curvature, global smoothness, response, reanalysis, and forecast claims remain out of scope. Frozen verification receipts now show 79 tests passed with 18 existing warnings in 18.97 seconds and 0 type errors/warnings/notes. The frozen source and plan hashes still match. RED has no blocker to the one planned root-owned guarded run; runtime execution evidence will remain separate from this preflight.
