# PR160 follow-up: theoretical FV scope

Baseline: PR head 0eb2a9d, merge 5d4666c, 2026-09-19.
Reuse stored numerical evidence; do not repeat the 240-grid runs unchanged.
Earlier Phase 1 completion and checkbox counts are not completion percentages
for this expanded FV scope. Real-data skill remains Phase 2.

- [x] Record final CPU run 35424239984 and bind its tested tree to main.
  Cancelled at the user's request; UI, packaging and typechecking passed, but
  the full CPU suite did not complete. Head and merge share tree
  dc06c91484ae24fae659bb253cffed1dd2678c8f. This closes status accounting,
  not full regression verification; use focused checks during follow-up.
- [ ] Connect typed background mean and precision at nonzero parameters.
  Check total observation/parameter derivatives against polished reanalysis;
  hold support fixed for differentiation and explicitly classify support changes.
  The pending external parameterized-mean API alone does not close this item.
- [ ] Connect partial observations from preparation through analysis, forecast
  and response, using one mask/support contract. Preserve grid/time and public
  output restrictions until their corresponding integration checks pass.
- [ ] Define the finite-impact domain using both signs and multiple directions.
  Report actual and linear changes, signal size, and absolute/relative error;
  declare tolerances before evaluating cases. Central derivative agreement is
  not finite-impact accuracy. Existing errors relative to actual changes are
  about 57.94%, 40.78%, 221.37% at +0.001, +0.0005, -0.0005 dBZ.
- [ ] Check field/JVP convergence over the intended forecast duration and record
  full D7 preparation, solve, refinement, forecast, adjoint, reanalysis, learning,
  serialization and retry costs. The 30-minute prescribed-flow probe and one
  490-second adjoint do not establish this whole-chain evidence.

Retain donor-cell exact-response limits unless a separately verified limiter
path is introduced. Keep exact-boundary numerical tests separate from synthetic
forecasts that use only boundary information available at issuance. Do not tune
on held-out outcomes or relax stationarity gates to obtain a pass.
