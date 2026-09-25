# R7 bounded process-response job lifecycle plan

The existing FV local-response numerics and two-process overlap evidence
remain unchanged. This is a **research coordinator** for two fixed
`fv4x5` response jobs, not an operational radar service or generic
request API. PyTorch forward AD continues in separate Python child
processes; no in-process simultaneous forward AD is attempted.

Add an optional, default-inactive cancellation callback to the existing
sampled wall/RSS process guard. A cancellation request terminates only
that child's process group, waits for its exit, and reports a distinct
`resource_termination=cancelled`. Existing callers without a callback
must retain their behavior. Test normal, wall, RSS and cancellation
paths with tiny subprocesses before touching FV.

Each job gets a unique empty directory and fixed case id `fv4x5`.
Before spawn, pin the worker/diagnostic source hashes, the four
archived reports, fixed input identity and resource policy. The worker
computes its existing one-score matrix-free response and records strict
54-stage branch, stationarity and true-adjoint residual. The parent
checks child exit/PID/command, sampled wall/RSS status, source/archive
identities before/after, fixed input identity, response values versus
archived reference, stage and PCG/HVP attribution, and completion of
all in-flight worker/diagnostic callbacks. Treat a partial child JSON
or monitor failure as **not publishable**.

Write a final `published.json` only after `run_guarded` has reaped the
child, every validation passes, and cancellation is still unset. Write
it to a temporary file then atomically replace. Its timestamps must
come after the worker's response/HVP window. Separately write a
`lifecycle.json` with execution, numerical and response statuses for
completed, cancelled, resource-limited or failed attempts. A cancelled
job may leave a partial raw child report, but must have no published
response. The cancellation/publication **decision** linearizes under
one token lock: a cancellation accepted before publication acquires
that lock wins; once publication has acquired it, a concurrent
cancellation waits and returns `False`, even if the temporary file is
still being written before atomic replacement. After publication it
cannot retroactively retract the result. Record the decision timestamp
inside that lock, separately from the later completed write timestamp.

In one guarded integration, launch two isolated fixed jobs concurrently:
let one complete and publish; request cancellation of the second
after its raw `running` report appears but before completion. Require
distinct PIDs, one eligible publication, one cancelled/no publication,
source/archive stability, separate resource records and no callback
or result leakage between jobs. Each child has 300-second wall and
sampled 768-MiB RSS cap; the parent records its own elapsed time but
does not call the sum of sampled peaks a system memory bound. No new
nonlinear reanalysis or physical forecast validation is performed.
