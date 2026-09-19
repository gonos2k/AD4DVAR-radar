# Phase 1 external-background evidence

The bounded run is recorded in `fv_external_background_phase1.run.json` and
`fv_external_background_phase1.json` (177.5 s, sampled peak RSS 379.7 MB, no
resource limit). The probe uses the real 4x5 CPU FP64 donor-cell FV objective
and forecast. `B_theta` is a three-parameter bounded correction to the first
observation using only its west/north spatial differences; fixed support and
observation sigma are retained. The fixed initial parameters are `(0.1, -0.2,
0.15)`, so the tested background chain has a nonzero cotangent contribution.

Verified from the saved report and targeted tests (`2 passed`):

- (g_B) chain rule: total theta response agrees with `B_theta^T g_B` to
  `7.3e-20`; total observation response agrees with the fixed-B direct term
  plus the nonidentity `B_y^T g_B` chain to `3.4e-19`. The reported
  `background_y_nonidentity_norm = 1.46e-9` means
  `||(B_y^T-I^T)g_B||`; it is a cotangent contribution, not a norm of the
  full Jacobian `B_y-I`.
- Stationarity residual is `2.1e-11`; exact-adjoint relative residual is
  `9.6e-11`; all finite reanalysis points remain in the positive face box.
- Centered finite reanalysis slopes agree with the autodiff response within
  `4.0e-14` for theta and `7.1e-10` for the tested observation direction.
- One fixed SGD update decreases training score by `4.46e-15`. The independent
  heldout score changes by `-2.16e-16` (a tiny worsening), so heldout learning
  improvement is inconclusive and is not claimed. These are observed FP64
  changes, not rigorous uncertainty bounds.
- A freshly constructed model and optimizer reload reproduces theta, score,
  control, and forecast exactly in the saved check (`0.0` absolute error).

The report explicitly keeps `production_learning_eligible`,
`typed_neural_prior_application`, and `legacy_fsoi_eligible` false. This is an
external research composition proof only; it does not extend the typed
neural-prior application, legacy FSOI, or production learning contract.
