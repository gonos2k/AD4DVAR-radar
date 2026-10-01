# Portable identity regression repair (2026-10-01)

## Scope and diagnosis

- Owned only `tests/test_fv_point_empty_time.py` and `tests/test_fv_research_problem.py`.
- The official Linux shard 1 failure at `full_cpu_package_ci_attempt4_shard1.log:770-793` compared the fresh fixed-problem digest `4acf88d7…` with a digest captured on another runtime, `dd8ec41e…`.
- The official Linux shard 3 failure at `full_cpu_package_ci_attempt4_shard3.log:780-789` compared regenerated 4x5 parameter bytes (`40bd28f1…`) to the archived historical tensor digest (`64cd06d7…`). The fixture's verification tensor had the same platform-sensitive byte-hash assertion. The 8x10 fixture also compared regenerated parameter and verification bytes to archive hashes.
- These failures are identity-byte portability failures; they do not establish a scientific objective, score, derivative, branch, or archive-data defect.

## Changes

- `tests/test_fv_point_empty_time.py:110-130` computes the pre-feature fixed-input identity schema from the current case inputs, leaving out the optional `empty_observation_time` key. This preserves the legacy schema check without pinning bytes produced by a different runtime. The existing changed-input identity check remains at line 89.
- `tests/test_fv_research_problem.py:20-27` continues to pin the immutable legacy adapter source artifact hashes.
- `tests/test_fv_research_problem.py:35-54` keeps the 4x5 archived parameter hash cross-check against the independent VJP artifact and cross-checks its archived verification hash against the partial-reanalysis preflight artifact. The regenerated verification and parameters feed both old and new implementations as the same current-runtime inputs; their bytes are not claimed to replay the historical archive.
- `tests/test_fv_research_problem.py:56-71` keeps exact current-runtime equality between the hash-pinned archived factory and current 8x10 fixture, and cross-checks the archived parameter and verification hashes between seed A and seed B.
- Existing objective, score, gradient, HVP, branch, layout, and unsupported-variant tests remain active.

## Verification

- Focused local command: `.venv/bin/python -m pytest -q tests/test_fv_point_empty_time.py tests/test_fv_research_problem.py`
- Result: `22 passed, 18 warnings` in 12.20 seconds. Warnings are existing `torch.jit.script` deprecations.
- `git diff --check -- tests/test_fv_point_empty_time.py tests/test_fv_research_problem.py` passed.
- `graphify update . --no-cluster` refreshed the cached code graph after edits (409 files extracted; 8,505 nodes, 426,340 edges).
- GREEN and RED reviewed the latest diff per the repository's team instructions; both found no remaining blocker. RED noted that this focused test does not duplicate independent raw-report SHA checks already present in the concurrent-response probe and FV86 execution summaries; this is not a blocker for the scoped portability repair.

## Limits and handoff

- No production source, archive report/hash, dependency lock, or shared environment was modified.
- No official workflow was dispatched, pushed, committed, cancelled, or replayed. Local Mac test success does not replace Linux CI evidence.
- The byte hashes in archived reports remain historical certification values. This test validates their archived cross-references and uses fresh same-runtime adapter comparisons for compatibility.

## Shard 4 identity follow-up

- Raw failure: `full_cpu_package_ci_attempt4_shard4.log:1471-1501`; the six separately reported tensor SHA-256 entries match the archived target, while `problem_identity.fixed_problem_sha256` is `92e83576…` on Linux versus archived `3de086e6…`.
- Diagnosis: reconstructed the `FVPointResearchProblem` class from commit `6b33392` and instantiated it from the exact current-runtime inputs returned by `preflight.fixed_problem()`. Its identity digest equals the current class digest (`3de086e6…`), and its one-lead `layout` and `support` values also equal current values. The class change in `1937493` generalizes lead counts, but its one-lead layout remains the same. This rules out a default identity schema regression.
- The digest includes tensor bytes nested throughout `frozen` and boundary schedules; the preflight's six named `tensor_sha256` values do not cover every nested tensor in that fixed problem. The available evidence is consistent with cross-runtime differences in an unreported nested tensor or serialized field, but it does not identify the exact Linux payload leaf or replay the Linux calculation.
- `tests/test_fv_point_3h_forward.py:47-91` compares the archived named tensor hashes and source hashes for the point case, spatial fixture, and fixed FV model exactly. It checks the current fixed hash against the explicit historical identity payload recipe on the same runtime, excluding optional empty-time metadata; verifies the legacy scope and one-lead layout; and verifies a changed verification input changes identity. No production or archive files changed.
- Focused local command: `.venv/bin/python -m pytest -q tests/test_fv_point_3h_forward.py::test_original_one_lead_layout_and_input_identity_stay_unchanged tests/test_fv_point_empty_time.py tests/test_fv_research_problem.py`
- Result: `23 passed, 18 existing JIT deprecation warnings` in 11.57 seconds. `git diff --check` passed for the three focused test files. Graphify refreshed after the final code edit (409 files extracted; 8,525 nodes, 570,266 edges).
- GREEN found no blocker. RED recommended reasserting the archived composite `fixed_problem_sha256`, but that would reintroduce the exact Linux failure this task is correcting (`92e83576…` versus `3de086e6…`) and contradict the requirement to preserve the legacy recipe without comparing runtime-sensitive composite bytes. The available archive stores that aggregate hash but no portable values/hashes for every nested fixed tensor. The test therefore pins the six named tensor hashes and the source hashes that generate hidden fixed tensors, checks the same-runtime legacy recipe, and verifies identity sensitivity to a fixed-input mutation.
- Remaining limit: source pins and named hashes do not independently prove every nested generated tensor is byte-for-byte identical to the historical run. Stronger cross-platform value provenance would require a portable component manifest or canonicalized archived values, outside this authorized test-only scope. No official workflow was dispatched or replayed.
