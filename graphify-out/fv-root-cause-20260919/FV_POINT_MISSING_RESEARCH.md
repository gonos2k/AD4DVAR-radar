# G5c bounded genuinely missing point observations

The fixed off-grid dBZ-point FV binding now accepts a fixed `uint8 [3,N]`
observation status: `0` is detected and `1` is genuinely missing. Censored
and QC-excluded observations are rejected as separate, unsupported meanings.
The narrow profile requires at least one detected observation per time and
fully known model state and inflow boundaries. Stored missing values use the
canonical `min_dbz` fill, but the differentiable parameter vector keeps the
same `[3N observations, theta]` layout; finite values in missing parameter
slots have no effect. The exogenous state-grid background remains independent
of point-observation values.

For time `t`, let `V_t` be the detected point indices and let `C` be the
optional fixed same-time point correlation. The implemented data term is

```text
z_t,V = sqrt(quality_t,V) * (H(x_t)_V - y_t,V) / std_t,V
u_t,V = C[V_t,V_t]^(-1/2) z_t,V             (or u_t,V = z_t,V when C=None)
J_data = sum_t sum_i pseudo_huber(u_t,V,i)
```

The symmetric inverse square root is recomputed from each **valid principal
submatrix** before evaluating the objective. Missing residuals are never
zero-filled into the full whitener, and a slice of the full inverse or full
whitener is not used. This matters even for two points: with correlation
`rho=0.4` and only point 0 observed, the correct valid covariance is `[1]`.
The missing point gives no extra information. The same choice preserves the
previous robust-loss whitening convention; no new covariance model is
inferred from radar products.

The fixed mask, raw matrix, and cached whole/subset whiteners are checked for
in-place changes before use. Status and the changed whitening convention enter
the fixed-problem identity, so old cached results must be rebound. Detected
parameters remain inside the declared detection range; missing slots need
only be finite. This means the objective parameter gradient, mixed
control-parameter JVP, forecast JVP and score dependence are exactly zero in
directions supported only on missing slots.

The reproducible fixed-control probe in
`examples/weather_scenarios/fv_point_missing_probe.py` uses three missing
rows (one at each time), a nontrivial four-point correlation and the existing
4×5 FV warm control. Its saved `fv_point_missing_metrics.json` reports 3
detected rows per time, 54 Euler stages, **zero** missing-slot parameter
gradient, mixed JVP and forecast JVP. An active point-direction gradient is
`-0.9975093361076117`; a centered difference at `h=1e-4` is
`-0.9975092743797278`, with absolute difference `6.17e-8`. The run performs
no GN analysis, stationary refinement, adjoint solve or signed reanalysis.
It checks the actual fixed-control FV dependency, not a new forecast score.

Focused regressions cover valid-submatrix whitening at varying masks,
singleton rows, point/covariance permutation, inactive parameter payloads,
AD derivatives, status meanings, active-value ranges, mutation refusal and
all-detected parity. The affected point suites report **64 passed**, with 18
existing TorchScript warnings. Pinned basedpyright on the changed module,
probe and tests reports 0 errors/warnings/notes. The exact commands and
source hashes are recorded in `fv_point_missing_manifest.json`.

This closes **genuinely missing fixed point rows under diagonal or the
declared same-time fixed correlation**. It does not support censored/QC rows,
all-missing frames, cross-time covariance, product-defined footprint or
regridding operators, or a newly qualified stationary whole response.
Those remain distinct G5c extensions. It also gives no physical forecast
validation.
