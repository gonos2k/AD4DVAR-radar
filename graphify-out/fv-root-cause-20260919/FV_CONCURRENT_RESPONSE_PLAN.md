# Bounded G9 concurrent FV local-response check

Purpose: verify that the new call-context minmod-stage and PCG diagnostic
observers do not cross-contaminate **two actual FV local-response calls** in
different Python threads. This reuses the archived 4×5/26-control and
8×10/86-control stationary points under their existing fixed observations,
verification fields and strict branch signatures. No GN analysis,
stationary refinement, signed endpoint reanalysis or physical truth run is
authorized by this plan.

The two predeclared workers are:

| Worker | Fixed control | Direction | Expected local branch |
|---|---|---|---|
| `fv4x5` | `minmod_middle_time_bias_final.json` nominal control | +1 dBZ on 20 middle-time cells | archived 54-stage selector/face signature |
| `fv8x10` | `fv86_seed_a.json` refined control | +1 dBZ on 80 middle-time cells | archived 108-stage selector/face signature |

Each worker rebuilds its current problem, checks fixed input/control hashes
against the corresponding archived report, enters its **own** PCG observer
context, waits at one start barrier, and calls `compute_local_response`.
An outer minmod-stage observer is bound **only inside that worker's ordinary
branch-check forecast**, outside all subsequent `torch.func`/replay derivative
calculations. Both first stage callbacks rendezvous inside their own observer
scopes; the stage callback windows must overlap and each must count exactly
54 or 108 stages. The worker records
timestamped HVP/PCG events, actual branch signature, nominal
gradient, true transposed-adjoint residual and direct/indirect/total
directional response. These must not contain the other worker's stage count
or solve events. The two workers also rendezvous before their first PCG HVP;
their HVP event windows must overlap, not merely their whole-call intervals.
The calculation compares the total response to the
archived same-problem value at relative tolerance `1e-6`, checks nominal
gradient `<1e-10`, true **relative** transpose-adjoint residual `<=1e-10`, and requires exact
per-worker archived branch choices/face signs. It also requires the two
worker execution intervals to overlap. These are regression gates, not new
independent nonlinear-response validation.

The named direction vectors and 4×5 verification field are checked against
pinned archived response files; the 8×10 direction is checked against the
pinned signed-reanalysis record. Worker reports retain full direction,
control, parameter and verification identities (shape/dtype/bytes), plus
absolute/relative true adjoint residuals and PCG-reported residuals.

The child process is guarded at **300 seconds wall time** and **1.5 GiB
sampled RSS** with `fv86_resource_runner.run_guarded`; a resource cutoff,
nonzero exit, incomplete source/input recheck, failed gate or missing worker
is recorded as refusal. The producer hashes relevant current core/adapter
sources and fixed inputs before work and checks them again before writing a
completed report. Raw archived reports are read-only and pinned by SHA-256.
The external runner hashes the current source set **before launching** the
child and after it exits; the child independently hashes that set before
and after its work. Historical archived source hashes are recorded as old
provenance and are not required to equal the observer-refactored current
source. These source checks cannot prove against a transient edit between
Python import and hashing, but they fail on persistent source changes.
The runner captures child exit, resource state, sampled peak RSS and log
separately from the local numerical statuses.

This is one bounded concurrency integration case. Passing it would not prove
arbitrary grids, other branches, async task inheritance, a finite path of
minmod derivatives, general FSOI, or forecast skill.
