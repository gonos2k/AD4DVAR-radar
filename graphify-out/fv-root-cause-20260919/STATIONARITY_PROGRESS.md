# FV stopping-condition reproduction — 2026-09-19

The previously completed240×240 demo and its five numerical-source hashes were verified unchanged at session start. Reuse that evidence; do not treat stale session memory saying the run is unfinished as authoritative.

## Controlled small experiment

Same12×12 rotation, observations/boundaries/initialization, cap12, gradient tolerance1e-6 and PCG tolerance unchanged. Only the relative-step termination threshold changes:

| step tolerance | outer / PCG | exact final robust-gradient norm | stop reason |
| --- | --- | --- | --- |
| 1e-5 | 5 /75 | 1.8687165e-5 | step_tolerance_unverified |
| 1e-10 | 8 /109 | 2.9134010e-7 | gradient_tolerance_unverified |

See probe_step_stop.py/json (exact commands, source hashes and result metrics). This establishes that delaying the small-step exit permits further descent in this case. It does not establish a general solver defect or prove smooth-branch stationarity. RED confirmed the independent step/gradient stopping contract at variational.py:6846/6986 and robust final gradient recomputation at7462. FV stationarity_verified is intentionally always false at result materialization and remains unchanged.

## Minimal change

fv_rotation_demo.py now accepts optional --maximum-outer-iterations and --step-tolerance and records those settings. Defaults remain4 and1e-5. No production solver equations, gates, or tolerance defaults changed. Existing focused builder/adapter tests:7 passed in0.90s.

- [x] Reproduce early step exit using one controlled small case.
- [x] RED review of the stopping contract; preserve certification boundary.
- [x] Expose explicit demo settings and record configuration.
- [x] One240×240 run: cap12, step1e-10 under600s/8GiB guard.
- [x] If completed, compare against saved baseline and update original HTML; retain unverified status.

## Large-case result

The240² run completed in477.21s, sampled peak RSS5,111,447,552bytes (~4.76GiB), without exceeding600s/8GiB. Six outer iterations/66PCG iterations; stop gradient_tolerance_unverified, final exact robust-gradient norm6.72655e-7, below unchanged1e-6 threshold. Prior4outer result was1.79522e-4. stationarity_verified remains false; no branch/Hessian/FSOI certificate was added.

Inputs, independent truth and P0 outputs are bitwise identical to the saved baseline. Maximum FV forecast change1.95055e-7dBZ; +180min MAE stays0.0198dBZ. This is termination-quality improvement, not demonstrated forecast-skill improvement. Shape rotation remains29.683deg versus30.940deg truth; its discrepancy is not explained by the prior early stop at displayed precision.

Original HTML now uses fv_rotation240_strict.json, retaining the seven legacy and two small comparison cases. Eighteen FV raw score/frame records and current numerical-source hashes checked; eighteen UI lead states, gradient-threshold label, mobile390/390px and no reported JS errors verified. Screenshot strict_solver.png. Old rotation240.webm remains historical; this run did not replace it.

Next: isolate remaining shape diffusion using identical physical domain/truth with prescribed-flow versus inferred-flow and a bounded refinement comparison. NewFV stationarity certification and FSO/FSOI learning remain separate open tasks. Do not infer all-case success from gradient threshold alone.

## Follow-up diagnostics: unresolved endpoint convergence

Two bounded CPU diagnostics were completed without changing production code or the HTML results:

- `probe_cost_reduction.py/json`: the cold-start negative 5e-4 endpoint stopped with gradient norm 1.3706e-8. A full GN trial predicts a cost reduction of 7.45e-20 and reduces the gradient to 2.3151e-11, but both direct cost subtraction and factored loss-term differences report an increase around 1.43e-15. Stable subtraction alone does not resolve the objective evaluation precision floor. This is not grounds to loosen acceptance or certify stationarity.
- `c5c_warm_reanalysis.py/json/log`: initializing all four perturbed solves from the nominal control did not resolve convergence. Endpoint gradient norms were 9.8972e-7, 2.0271e-11, 7.4023e-10, and 1.0201e-7. Branch checks passed and central slopes remained close to the GN response, but these are not uniformly stationary endpoint solutions.

Stationarity certification and FSO/FSOI learning validation remain open. The next mathematical decision is whether a bounded gradient-root refinement can be justified and verified without weakening objective acceptance; no such production change has been applied.

## Exact local root refinement — applied verification change

The <=32-control oracle in `examples/weather_scenarios/fv_sensitivity_probe.py`
now solves grad J = 0 with the squared-gradient merit function. The arbitrary
`cost + 1e-14` acceptance allowance is removed. Hessians must be finite and
numerically symmetric before Cholesky; zero-gradient saddles are rejected.
Each FV refinement trial is checked against the existing sufficient donorcell
latent-box branch margin. The gradient maximum target is 1e-10.

The original failed negative 5e-4 C5c endpoint was reproduced, then refined:

- Shared solver: no_accepted_step, gradient norm 1.3706207623e-8.
- Exact local refinement: gradient max 1.5382334295e-11, norm 2.2575458461e-11.
- Exact Hessian minimum eigenvalue 15.5093987505; condition estimate 1302.4674.
- Control correction norm 7.93039257e-11; checked box face margin 0.0025107461.
- ||gradient||/lambda_min = 1.45559856e-12 is a local linearized error estimate,
  not a rigorous neighborhood error bound.

Source/results: `refine_failed_endpoint.py/json`. This run used the new root
merit before the subsequent explicit Hessian symmetry guard and squared-merit
spelling refinement. The primary full numerical regression passed 12 tests in
105.60s; after those guards and an extra malformed-Hessian regression, 12 focused
tests passed in 1.47s (the long Taylor test excluded from this second run).
Production solve_analysis and stationarity_verified are unchanged. This closes
the reproduced local oracle stopping issue, not general FV certification.
