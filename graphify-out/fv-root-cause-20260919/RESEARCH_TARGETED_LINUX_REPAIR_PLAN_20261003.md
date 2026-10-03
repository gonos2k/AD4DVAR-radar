# Targeted Linux research repair verification (2026-10-03)

## Registration

Purpose: verify only the two resume-controller cases still failing in targeted
Linux run 37096279408. This is research evidence only. The historical full
Linux run `36836402831` remains failed (2,311 IDs, 87 failed/error); this plan
does not replace or relabel it. No deployment or full-suite claim follows from
two tests.

Source anchor: current main commit
`42891a34859924ff1bd04a35dc1e8690b8bbe971` (as inspected 2026-10-03). The
workflow records the actual checked-out commit separately, fetches the exact
anchor, and permits changes only to this workflow, this plan, and
`tests/test_fv_minmod_local_path_probe.py`. It asserts SHA-256 fingerprints for
the lockfile, FV implementation sources, local-path probe, and exact current
test source (`bc9a37f5c1df1f7a7821bc34d17d8629837e2e46bc86447345ebf4ac283825c8`)
before running. The one test-file change is the registered
cross-checkout fixture correction below; production sources and cache
certificates stay byte-identical to the anchor. Any further target-source or
lockfile change requires a new registration.

Environment: GitHub-hosted `ubuntu-24.04`; CPython `3.12.14`; pytest `9.1.1`;
PyTorch `2.13.0+cpu`; dependencies from the hash-pinned
`requirements/ci-py312-linux.lock`; one OMP/MKL thread. This matches the
historical run's reported OS family and Python/Torch/pytest versions. Runtime
and source SHA-256 values are emitted before execution. No local Linux engine
is available, and no equivalent targeted Linux run was found in recent
read-only Actions metadata; recent PR CI runs skipped CPU jobs.

## Fixed test inventory

Run exactly these two pytest node IDs, with no selector or retry:

1. `tests/test_fv_minmod_local_path_probe.py::test_new_direction_clears_old_pairs_and_computes_its_cross`
2. `tests/test_fv_minmod_local_path_probe.py::test_completed_direction_resume_validates_without_repeating_reanalysis`

The previous targeted run 37096279408 already passed the theta, portable
identity, 4x5 input fixture, runner, and lifecycle cases; this attempt must not
repeat those five. Both remaining tests exercise controller behavior with
current-runtime synthetic checkpoints. They must keep archived p-tensor,
saved-report, source, and cache payload checks exact. Do not weaken or rebind
the production cache identity.

## Budget and result rules

One manually dispatched job; 15-minute hosted-job maximum, 10-minute pytest
maximum, no retry, no rerun of the five passed cases, and no full-suite,
package, wheel, UI, deployment, or extra job. The workflow installs only the
locked test closure and editable source needed for imports, then runs the two
registered IDs. Preserve the provenance and pytest logs as a seven-day
artifact.

Pass requires JUnit to report exactly two cases with zero failures, errors,
or skips, plus expected runtime and source hashes and no timeout. Failure,
timeout, or wrong runtime is reported without retries as a failed/incomplete
targeted check. In all cases, run `36836402831` remains failed and no
whole-Linux-pass claim is made.

## Amendment A1: corrected selector after collection failure

Run `37095628674` checked out source `0238b78fd886806eda5402117d97ed7b9a24b548`.
The lock, runtime, and source fingerprints passed, but pytest exited 4 with no
tests run because the third selector named the removed
`test_case_archives_direction_branch_and_nominal_gradient` function. This is a
selector/collection failure, not a numerical failure. The run JSON, full log,
and uploaded provenance, pytest log, and empty JUnit report are preserved under
`graphify-out/fv-root-cause-20260919/linux_targeted_repair_attempt1/`.

The current 4x5 test is
`test_portable_case_uses_frozen_control_data_with_current_inputs[fv4x5-archive_names0-direction_slice0-54]`.
It retains the two exact archived-report SHA checks, archived 26-control seed,
middle-time direction slice, and 54-stage branch count, while constructing
verification and parameters from current inputs. It intentionally no longer
claims that regenerated inputs reproduce the archived branch signature or
stationary gradient. The separate production constructors retain exact
archived input/hash rejection checks. This is the registered portable-fixture
repair case, not a replay or certification of the historical response.

Workflow-equivalent local collection-only command, `.venv/bin/python -I -m
pytest --collect-only -q`, with the corrected selectors collected exactly seven
node IDs in 0.70 seconds; no tests executed. Attempt 1 remains a failed
collection record. Run 37096279408 was the subsequent targeted attempt and is
documented in Amendment A2; the current registration now contains only its two
remaining failures.

## Amendment A2: localized resume-test source identity

Run `37096279408` at source `7b18c2d6d462ca9a4c4734761b23e15cf1726cf7`
collected and executed all seven cases. Five passed. Only the two registered
resume cases failed, both with `KeyError` while the test helper indexed saved
source hashes by an absolute checkout path. The runtime, lock, and registered
source fingerprints passed. The complete run metadata, log, and uploaded
provenance, pytest, and JUnit artifacts are preserved under
`graphify-out/fv-root-cause-20260919/linux_targeted_repair_attempt2/`.

Both failures were reproduced with the repository's existing `.venv` Python
from a temporary copied checkout rooted at a different absolute path. The
helper no longer finds the original archived machine's absolute path and
raises the same `KeyError` before controller validation. The test-only repair
copies the archived JSON into `tmp_path`, localizes only its `source_sha256`
keys by known repository-relative suffixes, and points controller tests at
that copy. Stored hash values and all numerical/certificate fields are
preserved; the raw archived JSON is never rewritten. The synthetic resume
checkpoint derives its saved-report digest and path map from the localized
copy. An alternate-root regression checks this mapping explicitly.

The focused current-root run passed 8 tests in 3.76 seconds, including both
resume cases, alternate-root mapping, exact legacy archive anchoring, versioned
payload mutation rejection, and source/branch/fixture drift gates. The same
three resume and alternate-root tests passed from the temporary checkout in
3.13 seconds. Isolated basedpyright 1.39.9 reported 0 errors (403 warnings).
No production file or archived evidence changed. No new Linux run has been
dispatched. After GREEN/RED review, the next Linux attempt must run only the
two IDs above; the five cases from run 37096279408 are already passed evidence.

## Available command after review

After the test-source update is reviewed and checked in, one manual dispatch
on a ref containing the workflow and registered fingerprints can be made
with:

```sh
gh workflow run targeted-linux-repair.yml --ref <workflow-bearing-ref>
```

Do not dispatch as part of plan preparation.
