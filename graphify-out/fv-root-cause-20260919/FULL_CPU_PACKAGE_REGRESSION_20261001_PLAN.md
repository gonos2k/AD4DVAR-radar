# Full Linux CPU and package regression, 2026-10-01

Run the existing official `CI` workflow once by `workflow_dispatch`
at `main f7ee32b6a4e658065981010a3c4502aec7277090` (PR #227 merge).
The repository is public. Bind the resulting run ID, event, head SHA,
workflow source SHA256 and all job conclusions to this plan; do not
count earlier automatically skipped Python/package jobs as passes.
Do not modify the workflow, shared local `.venv`, dependency locks,
user `AGENTS.md` or existing numerical archives to obtain a pass.

The Linux CPU job must complete its hashed Python 3.12 closure install,
isolated dependency/lock audit, editable candidate installation,
dependency consistency check, product-source basedpyright check and
whole `python -I -m pytest -q` suite at one OMP/MKL thread. Preserve
its existing 120-minute job timeout. The package job must complete
its geodetic generator source/environment checks, hashed build/runtime
closures, source/wheel build, offline isolated wheel installation,
dependency check and installed CLI output-contract smoke. Preserve
its existing 15-minute timeout. The ordinary UI job is separate.

Local preflight collected 2,281 tests using the existing Python
3.12.13 / Torch 2.13.0 / NumPy 2.5.2 environment. This Mac has MPS
available; collection is neither execution nor CPU-only evidence.
Do not duplicate the full suite locally merely to repeat the official
run. Local fixes/tests may be needed if a concrete CI defect appears.

Poll the exact authoritative run handle until terminal; a monitoring
timeout is not a failed or stopped workflow and must not trigger a
duplicate dispatch. On failure retain job/step/log evidence and fix
the cause before a separately recorded rerun. Success requires both
CPU and package jobs to execute and pass, not only overall/UI green.
Save execution records/log hashes and update checklist/KG with the
precise commit/platform/scope. These regressions establish software
verification only; they do not close R2/R4/R5 stationary-response
research, R7 production service, or R6/R8 external physical evidence.
