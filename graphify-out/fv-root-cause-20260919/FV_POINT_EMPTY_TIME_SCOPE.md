# G5c one declared empty point-observation time

`FVPointResearchProblem` now has an optional `empty_observation_time` with
values `0`, `1` or `2`. Its default `None` retains the historical rule that
every one of the three times has at least one detected point. When a time is
declared, **exactly that one time** must have no status-0 detections and the
other two must have at least one each. A mismatched declaration, two empty
times or an all-empty observation window is rejected. Missing (`1`) and
externally QC-rejected (`2`) prepared rows retain their separate meanings
and canonical inactive fills; censored status remains unsupported.

For each time (t), let (V_t) be the status-0 indices. The data part of the
objective is

\[
J_{\mathrm{obs}}=\sum_{t:|V_t|>0}\sum_i
\rho\!\left(\left[C_{V_t,V_t}^{-1/2}
\operatorname{diag}(\sqrt{q}/\sigma)
(\mathcal H_t(x_t)-y_t)_{V_t}\right]_i\right).
\]

An empty time contributes **zero observation cost** and no correlation
whitener is formed for it. The all-point fixed correlation still has to meet
the existing SPD and conditioning contract; retained times use their own
principal submatrices. The FV trajectory, fully known initial state and
boundaries, forecast, verification score, control prior and smoothness prior
remain active. In particular, an empty first observation time does not
remove the exogenous initial background or its `theta` dependence.

The new 4×5 fixed-control regressions exercise an empty first, middle and
last time with a nonidentity four-point correlation. Each objective matches
an independently expressed two-time principal-submatrix sum within
`1e-12` absolute/relative tolerance. Inactive observation parameter
components have zero objective gradient and mixed control-parameter JVP;
changing only those components leaves objective, forecast and score exactly
unchanged. The fixture retains a finite, nonzero `theta` objective gradient
in all three cases. Default and mismatched opt-ins refuse, and the legacy
all-detected fixed-problem identity remains unchanged. The six affected
point-observation suites passed **74 tests** with 18 existing TorchScript
warnings; pinned basedpyright 1.39.9 reported **0 errors, 0 warnings, 0
notes**.

`FV_POINT_QC_STATUS_SCOPE.md` records the earlier QC-only source, for which
all-invalid frames were unsupported. Its historical manifest is left intact;
this later opt-in extends that limit only for exactly one declared time.

This is a **prepared synthetic observation-likelihood extension at fixed
control**. It does not mean the model state or boundary is unknown or that
forecast integration skips a time. No new GN stationary point, implicit
response, signed reanalysis, QC algorithm, censored likelihood, cross-time
covariance or independent weather accuracy was tested. The general minmod
response flag remains false and `response_validation` remains
`not_performed`.
