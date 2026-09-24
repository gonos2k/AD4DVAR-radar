# G5b fixed point-observation correlation

The bounded off-grid FV point profile now optionally accepts a fixed
`N×N` **observation-space correlation** matrix, shared independently by the
three regular observation times. In the current synthetic fixture `N=4`.
The point values are still dBZ sampled after model echo-to-dBZ conversion;
the initial grid background is still exogenous to those point values.

For each time (t), define the standardized point residual

```text
z_t,i = sqrt(quality_t,i) * (predicted_dbz_t,i - observed_dbz_t,i) / std_t,i
u_t = C^(-1/2) z_t
J_data = sum_(t,i) pseudo_huber(u_t,i)
```

`C^(-1/2)` is the **symmetric principal inverse square root**, computed once
from a fixed eigendecomposition outside the differentiated objective. The
diagonal/default case `C=I` follows the original path exactly. For the
chosen convention, the raw covariance represented at each time is
`D_t C D_t`, where `D_t=diag(std_t/sqrt(quality_t))`; quality therefore acts
before correlation whitening and can influence multiple whitened components.
Componentwise pseudo-Huber loss is not invariant to an arbitrary rotation of
whitened coordinates. Replacing the symmetric root with a Cholesky factor
would define a different robust objective, even though both give the same
quadratic Mahalanobis form.

The matrix must be CPU FP64, finite, symmetric to a `64*eps` absolute
tolerance, unit-diagonal to that tolerance, and positive definite with
`lambda_min > sqrt(eps)*lambda_max`. The dense option is limited to `N<=64`
points: construction costs `O(N^3)` and each time-row multiply costs
`O(N^2)`. No jitter, clipping or singular pseudoinverse silently changes
the declared covariance. Tiny accepted antisymmetry is explicitly replaced
by `0.5*(C+C.T)` **before** eigendecomposition, so the result does not depend
on which triangle the eigensolver reads. The raw input still has its own
identity; a roundoff-asymmetry regression checks both the computed objective
and data gradient. The matrix cannot require gradients; the problem
rejects accidental in-place changes to either the input matrix or its cached
inverse square root after construction.
Matrix bytes, shape, ordering and the whitening convention enter the fixed
problem identity.

The analytic two-point block with `rho=0.4` checks the symmetric root and
the observation-value gradient against a closed-form formula. The same
point permutation applied to coordinates, values, std/quality and `C`
preserves the robust objective and permutes its data gradient. Invalid
shape/dtype, nonfinite, asymmetric, non-unit-diagonal, singular and near-
singular matrices are rejected. In the fixed 4×5 warm-control fixture,
the diagonal objective is `0.07720456720208163`; this one correlated
example gives `0.06945751490746542`. These are **fixed-control** costs,
not analysis or forecast-skill improvements. Raw numbers and source hashes
are in `fv_point_correlation_metrics.json`.

All point rows remain detected and valid. The option does not support
missing/censored/QC rows, time-crossing correlations, arbitrary covariance,
or correlated errors inferred from resampled raster observations. If rows
are later missing, the valid principal covariance submatrix must be selected
and re-whitened; zero-filling a residual into the full matrix would be wrong.
No GN, stationary adjoint response, signed nonlinear reanalysis or physical
radar validation was executed for this extension.

Final affected validation reports **51 tests passed**, with 18 existing
TorchScript warnings; pinned typecheck on the changed problem and relevant
tests reports 0 errors/warnings/notes. The analytic checks include a
simultaneous point/correlation permutation, exact `None`/identity parity,
in-place mutation refusal for both input `C` and cached `C^-1/2`, and a
`16*eps` asymmetry case equal to its canonical symmetric counterpart while
retaining distinct raw-input identities. GREEN/RED final review found no
remaining blocker for this limited contract. Exact commands, logs, metrics
and current-source hashes are in `fv_point_correlation_manifest.json`.
