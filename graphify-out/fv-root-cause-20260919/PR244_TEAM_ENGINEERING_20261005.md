# PR244 engineering review

The bounded one-step driver now handles the block solver's declared numerical
refusal as a driver refusal while allowing unexpected runtime errors to
propagate. Cooperative deadline checks stop setup before the next stage after
cache loading, runtime/input reconstruction, branch evaluation, merit
evaluation, and shifted-direction construction. These checks do not interrupt
an operation already in progress.

Source auditing now includes every path pinned by the bounded-step plan, as
well as the explicitly required source dependencies, before and after the
attempt. An early refusal before those integrity checks complete is recorded
with `integrity_status: not_verified`; the parent marks execution failed and
states that child completion or integrity was not verified. The record makes no
source-unchanged claim for that path.

The independent toy-only review reproduced the block-refusal exception-type
mismatch and the setup deadline crossing. Regressions cover refusal
normalization, propagation of unexpected errors, source-map coverage, explicit
unverified integrity status, and stopping before source-path work after the
deadline. No FV trajectory or new physical calculation was run. The prior
PR243 plan and its historical attempt artifacts remain unchanged; they pin the
pre-review source bytes and must not be reused as current-source proof.

Validation on the current working files:

- `.venv/bin/python -m pytest -q tests/test_fv_point_3h_bounded_coupled_step.py` — 28 passed.
- Offline basedpyright on the driver and test — 0 errors.
- `git diff --check` on the driver and test — clean.

Current SHA-256 values:

- Driver: `71c4d56f0b497f63ed8f016db99aa225c98a9ad807d4d3c154c804a1364752e0`
- Tests: `3856c68c9d9abe21bdcc5a9294fb485f7b640e00343b35dd7c4bdf991196d80a`
