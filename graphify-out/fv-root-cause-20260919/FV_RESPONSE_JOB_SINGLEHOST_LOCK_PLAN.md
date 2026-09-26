# R7-P first increment: exclude duplicate launches of one fixed job

The PR #206 lifecycle proves one finishing and one cancelled fixed
`fv4x5` response in **different** job directories. It does not
serialize two coordinators targeting the **same** directory. The
current `directory.exists()/any()` check followed by
`mkdir(exist_ok=True)` permits both callers to pass before either
creates the directory. They can then collide on raw worker, resource,
log and atomic-publication paths. This change closes only that
single-host duplicate-launch race; it is not durable crash recovery
or a general queue.

Acquire a nonblocking OS `flock` on a persistent sibling lock file
before the job directory emptiness check. Resolve the parent path
before deriving the lock name so symlinked **parent** aliases of one
job directory share the same lock. Hold its file descriptor
through child launch/reap, numerical validation, cancellation versus
publication decision, atomic `published.json`, and final
`lifecycle.json` write. A second caller of the same fixed job
directory receives a distinct `ResponseJobBusyError` before it can
launch a worker or write any job artifact. Different job directories
retain independent locks and the existing process-isolated behavior.
Never unlink the lock file: removing a held inode could let a third
caller lock a newly created inode under the same name. OS closure
releases the lock after normal return, exception or process death.
The existing nonempty-directory guard still rejects reuse after a
crash; it must not be described as automatic recovery.

Validate request IDs as before. Reject preexisting symlinked job
directories and lock-file paths; these checks are not protection
against a hostile actor changing paths after validation. Use a
predictable sibling lock path on POSIX macOS/Linux local filesystems;
Windows and network filesystem behavior are outside this contract.
Keep original child wall/RSS limits, source/
input/archive/branch/adjoint gates, cancellation token ordering and
result schema unchanged. The lock is advisory and same-host only;
cross-host/shared-filesystem semantics are not claimed.

Regress the defect without running FV: block one fake job after it
acquires the lock, start a second call for the same directory, and
prove exactly one reaches the worker body while the other is busy.
Also test separate processes contend on the OS lock, release after
exception, no symlink following, and different directories can
proceed independently. Run the affected lifecycle/process-guard
tests and typecheck; preserve prior raw FV evidence. Update the
checklist/KG as a narrow R7-P subitem, leaving durable journal,
restart/reconciliation, cross-host queueing, generic requests,
concurrent GN/refinement and whole-system memory bounds open.
