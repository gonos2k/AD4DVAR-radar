# Local FV stationarity and learning evidence — 2026-09-19

## What changed

The small exact-Hessian oracle now refines grad J=0 using the squared-gradient
merit, instead of an arbitrary absolute cost allowance. It checks finite,
symmetric positive local curvature and every refinement trial's donorcell branch
box. It cannot accept a zero-gradient saddle. Production DA code is unchanged.

`fv_learning_probe.py` connects a positive observation-error scale sigma*exp(theta)
to the same robust objective and FV continuation. It computes
E_theta - J_c_theta^T H^{-T} E_c with the exact robust Hessian, then takes one
fixed-rate training update. This is a parameter derivative, not observation-space
FSOI; the latter's small finite-reanalysis evidence remains in the existing
stationary-sensitivity regression. No neural network has been trained here.

Training and holdout have different deterministic noise patterns. Holdout also
has a different initial field, from which both its observations and future truth
are regenerated. Future verification is used retrospectively for training;
holdout never selects the parameter update. Geometry, masks, prescribed boundaries,
and metric domain remain fixed. This is a same-model synthetic test.

## Executed results

| Quantity | Result |
| --- | --- |
| Controls / grid | 24 / 4×5, CPU FP64 |
| Exact training dE/dtheta | 6.1843433784e-5 |
| Smallest central-difference error | 2.53047e-12; three levels show second-order convergence |
| Updated theta | -1.5460858446e-5, fixed learning rate 0.25 |
| Predicted / actual training impact | -9.56152576e-10 / -9.56136971e-10 |
| Holdout MSE before / after | 0.00019917512618195977 / 0.0001991746051878452 |
| Holdout gain | 5.20994115e-10 (~0.000262%) |
| Holdout gradient maxima before / after | 2.35783e-11 / 1.34676e-11 |
| Holdout minimum Hessian eigenvalues | 15.84844 / 15.84890 |
| Local score-error estimate with safety factor 10 | 8.16069708e-13 |
| Parameter reload control / forecast / score differences | 0 / 0 / 0 |

The error estimate uses ||E_c|| ||grad J|| / lambda_min(H), plus measured replay
difference and a score-roundoff scale. It is a **local linearized diagnostic**,
not a rigorous neighborhood bound or statistical uncertainty. Improvement is
small and does not establish practical forecasting skill. Pointwise curvature
and checked trial boxes do not enclose an unknown stationary solution path.

## Reused and new execution evidence

- Full existing stationary-sensitivity regression with the new root merit:
  12 passed in 105.60s, including actual FV Taylor/adjoint checks.
- After finite/symmetry and nonfinite-objective guards: 13 focused tests passed
  in 1.48s; the previously passed long Taylor test was not repeated.
- Initial new learning regression: 1 passed in 170.56s (agent execution).
- `fv_learning_preliminary.json` preserves the subsequently exported training/FD
  results. Its unpolished holdout and misleading repeatability label are superseded.
- `verify_learning_holdout.py` reran only baseline, updated and reloaded holdout
  solves with identical exact polishing, checked local Hessians, and merged the
  unchanged training/FD evidence into `fv_learning.json`. Theta was not retuned.
- Final strengthened assertions in `assert_learning_evidence` passed against
  this merged executed evidence. The entire revised probe was not redundantly
  rerun end-to-end. JSON explicitly names the reused evidence.
- Parameter checkpoint: `fv_observation_scale.pt`. This is parameter-only replay
  under a prescribed contract, not arbitrary saved-pipeline reconstruction.

GREEN found the local mathematical connection coherent. RED's inconsistent
holdout polishing and misnamed repeatability scale findings were resolved;
its limitation on pointwise error estimates is retained explicitly. Production
`stationarity_verified` and learning eligibility remain false.

## Demo and remaining work

`publish_learning_evidence.py` inserts the results into the original template and
HTML under “정상점·민감도·학습 검증 — 별도 4×5 합성 실험”. Embedded scenario JSON is
unchanged (SHA256 6416025107f7141021cba4d3d7c9ab9d6dc23c809b7febbb1260bbe2689372cc).
The 240×240 demo does not inherit these results. General FV stationarity eligibility,
large-grid exact sensitivity, neural/persistent learning and real-data validation
remain open; no all-case or Phase 1 100% completion is asserted.

Aside browser verification: the original page loaded through a loopback HTTP
server, the new disclosure opened by clicking its summary, and its displayed
numbers/scope matched the saved JSON. An initial cached page omitted the new
section; a fresh query URL showed it. Aside's Page does not implement the attempted
setViewportSize call, so this session does not claim a new mobile viewport check.
The existing mobile result applies to the earlier page layout only.
