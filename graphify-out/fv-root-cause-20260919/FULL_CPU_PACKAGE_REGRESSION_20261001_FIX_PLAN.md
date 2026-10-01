# Dependency-audit repair and second official regression run

Attempt 1, GitHub Actions run `36798315747`, completed with failure
at `main f7ee32b6a4e658065981010a3c4502aec7277090`. Both CPU and
package jobs installed their hashed CI closure, then the strict audit
reported three urllib3 2.7.0 vulnerabilities (CVE-2026-97687/97688/97689),
with 2.8.0 as the fix. CPU tests, wheel build/install and CLI smoke
were not reached. Preserve the original plan, terminal run record and
failure log as evidence; do not relabel this attempt passed.

The [upstream changelog](https://urllib3.readthedocs.io/en/stable/changelog.html)
documents the three 2.8.0 security fixes. Verify PyPI 2.8.0 metadata,
wheel Python requirement and SHA256 of downloaded wheel/source files.
Change only the urllib3 version/hash block in the two existing Linux
CI locks, from 2.7.0 to 2.8.0. This is a targeted patch, not a lock
regeneration: retain all other package entries and both runtime locks.
Keep the strict audit, workflow gates, job timeouts, shared `.venv`,
user `uv.lock` and user `AGENTS.md` unchanged.

Run the existing lock synchrony checker and compare all non-urllib3
blocks byte-for-byte to the baseline. GREEN/RED review the patch and
release verification. Commit/push the candidate and open a PR; then
dispatch the same official workflow once on that exact branch commit.
Bind its run ID/head SHA/event/job/step results and logs to the new
candidate. Poll until terminal and inspect the CPU test summary and
package build/offline-install/CLI-validation steps individually.
Do not relax or skip a new failure. Further defects need their own
evidence and targeted correction before a recorded rerun.

Merge only after the candidate's relevant official jobs pass and
review findings are resolved. Record source-tree equivalence between
the tested candidate and merge for code/workflow/dependency files.
Full-suite/package success remains software evidence; R4/R2/R5
qualified responses, R7 service and R6/R8 physical inputs remain open.
