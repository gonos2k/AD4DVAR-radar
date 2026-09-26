# R7-P first increment: one local job directory, one coordinator

The prior PR #206 lifecycle established one finishing and one
cancelled fixed `fv4x5` response in **different** directories. This
follow-up closes a separate race in `run_job`: two coordinators could
both observe the same job directory as empty, then write the same raw
worker, resource, log and publication paths.

`run_job` now takes a nonblocking POSIX `flock` on a persistent sibling
file **before** the empty-directory check. It holds the lock through
worker launch/reap, numerical validation, cancellation/publication
decision, atomic `published.json` write and final `lifecycle.json`
write. A concurrent same-directory call raises
`ResponseJobBusyError` before its worker body runs or its job
artifacts are written. Different directories still use different
locks. The lock file is never unlinked, because replacing a held lock
inode could split ownership. Normal return, exceptions and process
exit release the OS lock; a crash can still leave a nonempty job
directory that current `run_job` refuses to reuse.

The parent path is resolved before deriving the lock name. Thus
`real/job` and `alias/job` through a symlinked **parent** contend on
the same lock. Preexisting symlinks at the job directory or lock file
are rejected; this check is not adversarial protection against
another actor swapping path components after validation. Contention
errors `EAGAIN`/`EWOULDBLOCK` become `ResponseJobBusyError`; other
open/lock errors, including `EACCES`, propagate as operational errors.

The new lock regressions use fake jobs or tiny lock-only subprocesses,
with **no new guarded FV response worker run**. The wider affected
suite also exercises existing fixed-case forward/branch checks. The
lock tests show one of two same-job
callers enters the fake worker, cross-process contention and later
reacquisition, release after an exception, independent directories,
parent-path alias contention, and preexisting symlink rejection. The
affected lifecycle/process-response/concurrent-response selection
passed **38 tests**, and pinned basedpyright 1.39.9 reported zero
diagnostics. A code-only Graphify refresh recorded 7,771 nodes,
67,256 edges and 264 communities. The existing PR #206 numerical
report and publication evidence were not changed; no full CPU/package
regression was run.

This establishes **single-host duplicate-launch exclusion for the
fixed research job** on a POSIX macOS/Linux local filesystem. It is
not a durable queue or recovery mechanism. Cancellation state still
lives in one process, and a parent crash between worker completion,
publication and lifecycle recording still needs reconciliation.
Generic inputs, cross-host locking, concurrent GN/refinement,
in-process forward AD, and whole-system memory bounds remain open in
R7-P.
