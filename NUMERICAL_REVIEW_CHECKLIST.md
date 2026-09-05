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

## Post-merge follow-up — `main 8b2229d`

- [x] N05 follow-up: reproduce acceptance of constant FP32 projections at `1e-8` and `1e-5`. Normalize each observation/lead/metric by its own maximum magnitude before checking the mean; zero components use a divisor of one. Validate both observation perturbations and forecast projections through the public factory. Explicitly centered zero projections have zero impact and jackknife uncertainty.
- [x] Precision boundary: restrict ensemble statistics to FP32/FP64 and document the requirement. The inherited `128 * eps` tolerance equals one for BF16, making its normalized centering test vacuous. FP16/BF16 inputs are now explicitly rejected rather than silently accepted or converted.
- [x] N01 follow-up: reproduce the incorrect zero velocity at FP32 controls `(2e20, -2e20)` and larger values. Scale latent coordinates by their maximum absolute component (at least one) before squaring. The normalized squared norm stays at most two; the original Taylor branch is unchanged near zero.
- [x] Apply the same normalization to the fallback projection for saturated/outside backgrounds. Mask inactive branches before their norm arithmetic. Confirm the output direction, limiting magnitude and analytically normalized Jacobian, not just finiteness.
- [x] MPS follow-up: scalar `new_tensor()` factories inside the transformed vector decoder raised `DispatchKey Undefined doesn't correspond to a device`. Explicit dtype/device constant construction and `ones_like`/`zeros_like` preserve the formula and allow the MPS background/control Jacobian tests to pass.
- [x] Preserve origin first/second derivatives, rotation equivariance, inward updates from saturated backgrounds, frozen IRLS and final linearization behavior.

The new regression checks failed before implementation for the small-scale
centering and norm-overflow cases. Final checks:

```sh
.venv/bin/python -m pytest -q tests/test_numerical_review.py tests/test_ensemble_sensitivity.py --tb=short
# 29 passed, 28 subtests passed; includes native MPS Jacobian checks.

.venv/bin/python -m pytest -q tests/test_variational.py -k 'radial_velocity or bounded_controls or physical_motion_increment or final_linearization_tracks_remap_branch_changes or frozen_irls_matches_true_robust_gradient' --tb=short
# 7 passed, 135 deselected, 6 subtests passed.

uvx --from basedpyright==1.39.9 basedpyright --level error --pythonpath .venv/bin/python src/advar/variational.py src/advar/ensemble_sensitivity.py
# 0 errors, 0 warnings, 0 notes.
git diff --check
# PASS.
```

Independent comparison of 128 normal-scale FP32/FP64 cases found maximum
old/new differences of `1.91e-6` / `3.55e-15`, with the same bounds for rotation
equivariance. Eight `gradcheck`/`gradgradcheck` cases covered the origin, tied
maximum components, the scaling crossover, nonzero backgrounds and outside
cancellation. Inactive-branch overflow cases also retained finite Jacobians.

Scope remains **norm overflow**, where the vector components themselves are
representable. Active componentwise multiplication/addition can still overflow
(for example `scale=2`, `limit=1`, zero background and maximum finite controls).
This change does not claim support for every finite input combination. CPU
FP32/FP64 and local MPS Jacobians were checked; full MPS/CUDA analyses and the
complete baseline were not rerun.

PR #155 is merged. The separate initial-field comparison, scored-pixel display,
bounded `shift_zero()` allocation and provenance extraction are recorded in
`SCIENTIFIC_REVIEW_CHECKLIST.md` and carried by PR #156.
