# Targeted Linux research repair verification (2026-10-03)

## Registration

Purpose: check whether the seven previously failing cases pass on the Linux CPU
runtime that produced run `36836402831`. This is research evidence only. The
historical run remains failed (2,311 IDs, 87 failed/error); this plan does not
replace or relabel it. No deployment or full-suite claim follows from seven
tests.

Source anchor: current main commit
`42891a34859924ff1bd04a35dc1e8690b8bbe971` (as inspected 2026-10-03). The
workflow records the actual checked-out commit separately, fetches the exact anchor,
and requires its Git tree to differ only in this workflow and plan. It asserts SHA-256
fingerprints for the lockfile, three FV implementation files, local-path
probe, and six target test files against this anchor before running tests.
This permits the workflow and plan to be checked in without silently changing
the registered test sources. Any target-source or lockfile change requires a
new registration.

Environment: GitHub-hosted `ubuntu-24.04`; CPython `3.12.14`; pytest `9.1.1`;
PyTorch `2.13.0+cpu`; dependencies from the hash-pinned
`requirements/ci-py312-linux.lock`; one OMP/MKL thread. This matches the
historical run's reported OS family and Python/Torch/pytest versions. Runtime
and source SHA-256 values are emitted before execution. No local Linux engine
is available, and no equivalent targeted Linux run was found in recent
read-only Actions metadata; recent PR CI runs skipped CPU jobs.

## Fixed test inventory

Run exactly these seven pytest node IDs, with no selector or retry:

1. `tests/test_fv_observation_response.py::test_parameterized_background_matches_polished_centered_reanalysis`
2. `tests/test_fv_research_problem.py::test_common_objective_score_and_derivatives_match_frozen_reference[4x5]`
3. `tests/test_fv_concurrent_response.py::test_case_archives_direction_branch_and_nominal_gradient[_small_case-archive_names0-direction_slice0-54]`
4. `tests/test_fv_partial_face_event_runner.py::test_runner_launches_exact_guarded_command_and_records_completion`
5. `tests/test_fv_response_job_reconcile.py::test_consistent_terminal_lifecycle_allows_candidate_and_is_idempotent`
6. `tests/test_fv_minmod_local_path_probe.py::test_new_direction_clears_old_pairs_and_computes_its_cross`
7. `tests/test_fv_minmod_local_path_probe.py::test_completed_direction_resume_validates_without_repeating_reanalysis`

These sample the cancellation-sensitive adjoint tolerance, portable identity,
archived input fixture, synthetic runner and lifecycle fixtures, and both
resume controller cases. The last two exercise current-runtime synthetic
checkpoints; archived p-tensor and saved-report hashes remain exact cache
contracts and are checked by their separate tests. Do not replace the test
fixtures with saved certificates or weaken/rebind cache identity.

## Budget and result rules

One manually dispatched job; 15-minute hosted-job maximum, 10-minute pytest
maximum, no retry, no full-suite, package, wheel, UI, deployment, or extra job.
The workflow installs only the locked test closure and editable source needed
for imports, then runs the fixed seven-node command. Preserve the provenance
and pytest logs as a seven-day artifact.

Pass requires JUnit to report exactly seven cases with zero failures, errors,
or skips, plus expected runtime and source hashes and no timeout. Failure,
timeout, or wrong runtime is reported without retries as a failed/incomplete
targeted check. In all cases, run `36836402831` remains failed and no
whole-Linux-pass claim is made.

## Available command after review

After this workflow is checked in, a manual dispatch on a ref containing the
workflow and the registered target-source fingerprints can be made with:

```sh
gh workflow run targeted-linux-repair.yml --ref <workflow-bearing-ref>
```

Do not dispatch as part of plan preparation.
