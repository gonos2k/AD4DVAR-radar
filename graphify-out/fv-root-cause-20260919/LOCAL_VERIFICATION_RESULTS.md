# Local FV stationarity and neural learning — 2026-09-19

This record supersedes the remaining-item summary in LEARNING_RESULTS.md for
the specific experiments below. Phase 1 synthetic evidence only; no real-data
or general production eligibility is asserted.

## Why the previous work was incomplete

1. The 240×240 solver reached its configured robust-gradient tolerance, but
   `FVAnalysisResult` deliberately stayed unverified: no explicit independent
   derivative/branch/trajectory check existed. A termination reason alone was
   insufficient evidence.
2. The research FV observation response used IRLS Gauss–Newton curvature. In
   general this is not the exact Hessian needed to differentiate the stationary
   nonlinear robust objective. The selectable `exact_robust_hessian` mode now
   computes that Hessian-vector product; the default approximate mode remains.
3. Earlier learning changed a scalar observation-error scale, not a neural
   network. A real three-parameter neural error model and checkpoint replay are
   now exercised. This is not an initial-background neural prior.

For F=J_c=0 and nonsingular H=J_cc, the local response is
`dE/dy = E_y - J_cy^T H^{-T} E_c`; parameter learning uses the corresponding
`E_theta - J_c_theta^T H^{-T} E_c`. Fixed verification weights and boundaries
are retained. When error statistics depend on y, their dependence belongs in
J_cy too. Finite positive curvature encountered by PCG is not a global SPD proof.

## 240×240 result

- Existing solver, objective and tolerance unchanged. A trusted checkpoint was
  added so verification does not repeat the assimilation.
- Explicit local verification recomputes the full robust gradient and all five
  trajectory fields, requires full support/positive interior echo, and checks
  the frozen-IRLS residual derivative against the actual robust gradient.
- Gradient norm `6.726550811915307e-7 <= 1e-6`.
- Five sampled directions: one spatial cosine, three streamfunction coefficients
  and growth. Central residual-JVP relative errors ranged from `2.83e-10` to
  `1.41e-7`; declared face boxes remain interior. These samples do not cover
  every direction or enclose an unknown implicit solution path.
- Final verification: 48.48 s, sampled child peak RSS 2,583,314,432 bytes.
  Checkpoint-producing analysis/forecast: 428.50 s, 5,178,490,880 bytes.
- Input, truth, persistence, all P0/FV forecasts and metrics exactly equal the
  previous strict run. `rotation240_preservation.json` records the comparisons.
- `rotation240_verified.json` binds the new local report to these forecasts.
  Its numerical-solve source hash is explicitly the module loaded by the solve;
  verification has its own later source hash. Original unverified files remain.

This is numerical first-order verification for this declared analysis contract,
not proof of a minimum, general FV stationarity, 240×240 exact FSOI, or production
learning eligibility. Future forecast intervals are not certified by this check.

## Actual neural learning and observation response

`fv_neural_learning_probe.py` uses a 4×5 CPU FP64 donorcell problem with 24
controls. A bounded positive observation standard deviation comes from a real
`Linear(2,1)` plus tanh model (three parameters). Distinct standardized echo and
temporal-change features avoid a duplicate constant feature and bias.

- Exact parameter adjoint and central FD agree; FD error drops approximately
  fourfold when the step halves. Local Hessian minimum eigenvalue is 15.79.
- One fixed learning-rate update decreases training MSE. A separate initial
  field/noise reanalysis decreases held-out MSE from `0.00019917512618195977`
  to `0.00019917472238659394`: gain `4.037953658e-10`, about **0.000203%**.
- Local linearized error diagnostic `6.763319463e-13` is smaller than this gain;
  it is not a rigorous error bound or statistical significance estimate.
- Saved state_dict loaded into a fresh model reproduces control/forecast/score
  exactly. Run: 173.42 s, sampled peak RSS 362,725,376 bytes.
- A separate fixed-model observation probe includes feature normalization,
  observation-error statistics and `B=y[0]`. Total response `-7.8017873e-4`
  matches reanalysis central FD with errors `7.10e-10 -> 1.78e-10`.
- **Unresolved attribution:** total versus frozen-statistics response differs
  by only `4.16e-11`, below the finest FD error. The NN-specific observation
  contribution is not independently resolved (`nn_effect_resolved=false`).
  This does not negate the separately verified NN parameter update.
- Observation probe: 90.9 s, sampled peak RSS about 333 MB. No retraining.

## Tests and review

- Exact/default FV observation-response tests: 20 passed.
- Stationarity + existing residual tests before the final integrity fix:
  22 passed; after the fix, all 7 stationarity tests passed (14.53 s).
- Regression rejects stale support/streamfunction/growth, nonstationary
  controls, missing observations and deliberately incorrect residual derivatives.
- Neural feature-rank/bounds test passed; full learning probe executed once and
  its independent assertion helper passed against the saved result. The full
  pytest wrapper was not redundantly rerun.
- Luna high GREEN/RED reviews confirmed local scope. RED's stale-trajectory
  counterexample was fixed and the 240×240 verifier rerun. The unresolved
  NN-specific contribution is retained explicitly, not promoted to success.
- No full repository, MPS/CUDA, or real-data run was performed this turn.

## Remaining general work

- General boundary/missing/detection/support contracts and nonsmooth switches
  require their own local differentiability or declared generalized derivative
  rules. This verifier intentionally rejects unsupported contracts.
- Large-grid exact adjoint conditioning/solution-path verification, general
  background neural-prior integration and persistent learning remain open.
- The legacy displacement-based sensitivity/learning contract cannot be made
  valid for a spatial FV velocity field by toggling an eligibility Boolean.
- Donorcell numerical diffusion, grid convergence, unknown boundary information,
  and real-data generalization are not resolved by the low MAE of one rotation.

The original HTML displays the 240×240 local analysis result and the separate
small neural experiment with these limits. Historical video remains historical.

## HTML and graph verification

Aside 1.26.916.1741 opened the regenerated original HTML over a temporary
loopback server. The actual DOM showed the local verification label and the
expanded neural evidence with the above values. Moving the lead control to 180
minutes showed MAE 0.0198 versus persistence 0.2031 dBZ; truth/main-axis rotation
30.940 degrees versus forecast 29.683 degrees (forecast centroid 30.906 degrees).
All 13 scenario choices remain. At 1440px, body width equalled viewport width.
This turn did not repeat all 18 visual states or mobile checks; their frame arrays
were preserved exactly, and earlier visual records remain separate.

Code-only KG update extracted 240 files without LLM semantic extraction:
6,353 nodes, 64,410 edges, 216 communities. graph.json and GRAPH_REPORT.md were
updated; graph.html was skipped above the existing 5,000-node visualization limit.
No wiki directory exists to synchronize. Source hashes and 13-scenario/18-frame
assertions are in local_verification_manifest.json. No commit or push performed.

## PR preparation follow-up

The isolated feature-to-NN-standard-deviation path was independently tested with
raw observations, background, score and model parameters fixed. AD response
`-4.1615539084e-11` agrees with reanalysis FD at h=0.1 and 0.05 to
`1.94e-15` and `1.05e-15`. This resolves the **isolated local path**, while the
earlier full-input finite-difference resolution flag stays false. The earlier
total-minus-frozen value is a decomposition cross-check, not the independent FD
oracle. All endpoint gradient maxima are below 1.95e-11; face boxes stay positive.
Runtime 73.6s, sampled peak RSS about331MB. No parameter retuning occurred.
The new regression assertions passed against this executed result.

PR packaging found a static import cycle from placing verification in the solver.
The helper now belongs to fv_sensitivity.py, which already depends on the solver;
no compatibility wrapper or cycle suppression is added. The optional demo
`--verify-stationarity` flag composes this helper with the existing solve and
forecast. Previous numerical records remain evidence of their recorded versions.
Default pytest collection is limited to tests/ to exclude archived source copies.
