# Mathematical and numerical review — 2026-09-05

Scope: six reproducible findings from the numerical review of commit `ca551ea3`.
Preserve existing physical constraints, input contracts, and automatic-differentiation paths.

## Corrections and regression checks

- [x] N01: Preserve the background-velocity Jacobian at zero velocity, including finite second derivatives and nonzero controls. Clamp unused Taylor arguments to keep large-control gradients finite.
- [x] N02: Evaluate pseudo-Huber loss without small-residual cancellation; verify the public float32 preparation/solve path.
- [x] N03: Keep affine cell-area nominal values inside their directed enclosure for valid grids.
- [x] N04: Compute truncated Gaussian interval likelihoods in both tails with finite gradients.
- [x] N05: Validate ensemble centering independently for each observation/lead/metric; preserve centered covariance and jackknife results.
- [x] N06: Compute transport outflow directly from lost boundary cells using wider audit accumulation; verify zero and positive outflow.
- [x] Add project `AGENTS.md`: simple, clear, concise, intuitive code; minimal changes; numerical and autodiff checks.

## Validation

- [x] Regression checks reproduce the original six failures before implementation.
- [x] New regression checks pass after implementation (10 tests; 9 together plus the added two-tail gradient test).
- [x] Recheck the promotion fixture failure with fixed sources and compare the same single test with isolated HEAD. Both passed; record the transient failure below. The complete baseline suite was not rerun (ancestor `AGENTS.md`).
- [x] Review the final diff and check whitespace/type issues relevant to the changes.
- [x] Record exact commands, outcomes, and limitations below.

## Evidence

Commands run from the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_numerical_review.py --tb=short
# 9 passed, 5 subtests passed before adding the final two-tail check.
.venv/bin/python -m pytest -q tests/test_numerical_review.py -k both_tails --tb=short
# 1 passed, 9 deselected.

.venv/bin/python -m pytest -q tests/test_variational.py -k 'radial or bounded_controls or robust or huber' tests/test_nowcast.py -k 'radial or bounded_controls or robust or huber or transport or outflow or area or affine' --tb=short
# 20 passed, 39 subtests passed (the final -k expression selects tests).
.venv/bin/python -m pytest -q tests/test_ensemble_sensitivity.py tests/test_promotion.py -k 'ensemble or truncated or quantized_gaussian or observation_error_gaussian' --tb=short
# 18 passed.
.venv/bin/python -m pytest -q tests/test_variational.py tests/test_sensitivity.py -k 'gradient or stationarity or final_linearization or fso_matches_dense or solver_reuses or radial or bounded_controls or physical_motion_increment' --tb=short
# 17 passed, 26 subtests passed.
.venv/bin/python -m pytest -q tests/test_promotion.py tests/test_cli.py -k 'quantized or conditional_pit or prior_uncertainty or audit_is_optional' --tb=short
# 7 passed, 1 failed in candidate process-start receipt lineage validation.
.venv/bin/python -m pytest -q tests/test_promotion.py::NeuralPriorPromotionTests::test_unreliable_prior_uncertainty_blocks_promotion --tb=short
# Fixed-source rerun: 1 passed in 37.36s.

uvx --from basedpyright==1.39.9 basedpyright --level error --pythonpath .venv/bin/python src/advar/variational.py src/advar/nowcast.py src/advar/diagnostics.py src/advar/promotion.py src/advar/ensemble_sensitivity.py
# 0 errors, 0 warnings, 0 notes.
git diff --check
# PASS.
```

## Observations and limits

- N02: the original 4×4 case now has the correct positive loss, but its improvement remains below the existing final acceptance threshold. That threshold is unchanged. The 16×16 regression returns an analysis without fallback and reduces cost from about `1.19615e-5` to `6.11390e-6`.
- Independent review confirmed the radial first/second derivatives and the geometry, transport, likelihood and centering changes. The large-control inactive-branch issue found during review was corrected and regression-tested.
- Local tests used CPU PyTorch 2.13.0. Native MPS/CUDA execution was not validated. Existing TorchScript deprecation warnings remain.
- `basedpyright` was absent from `.venv`; the repository-pinned version `1.39.9` was run through `uvx` without changing project dependencies.
- The initial promotion selection had one lineage-validation failure while source edits overlapped the test run. The same test passed on fixed modified sources (37.36s) and on isolated HEAD `ca551ea3` (38.92s). No reproducible regression remains; the precise mismatching lineage field in the initial run is unconfirmed. Comparison command, from a temporary `git archive HEAD` extraction: `PYTHONPATH=<archive>/src /Users/yhlee/ADVAR/.venv/bin/python -m pytest tests/test_promotion.py::NeuralPriorPromotionTests::test_unreliable_prior_uncertainty_blocks_promotion -q --tb=short`.
