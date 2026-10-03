# S4 three-hour endpoint block/Schur diagnostic

One guarded, read-only Hessian diagnostic completed at the pinned PR204-tail endpoint. Fresh objective, merit, gradient, input identity, branch, and runtime matched the saved tail record. The endpoint remains nonstationary: full gradient infinity norm is `0.87197938` (`||g||₂=1.3800373`), with field L2 `0.2999055`, flow L2 `1.3446422`, and growth gradient `0.0806030`. No step, score, adjoint, response, or physical evaluation was run.

The exact 26-column Hessian is symmetric to `1.30e-15`; its symmetric-part eigenvalue range is `[0.8174637, 3435.6900]`. A fresh 27th HVP verifies the minimum-eigenpair residual at `1.65e-16`. For the `20+6` partition, `Hff` eigenvalues span `[0.9570084, 1371.2550]`, with condition `1432.856` and declared positive floor `1.3713e-5`; `Hdd` spans `[0.9984949, 2514.9986]`. Both `Hff` solves pass: relative residuals are `2.88e-16` for `Hff X=Hfd` and `2.53e-16` for `Hff y=gf`. The reported normalized coupling is `0.673615`.

The six-dimensional Schur complement has eigenvalues `[0.9894184, 90.00358]`, symmetry error `2.99e-15`, and the algebraically eliminated dynamics residual is `[0.8660473, -0.6761816, 0.5493044, 0.4245856, -0.3112678, 0.2828911]`. Because `gf` is nonzero, this residual is not the gradient of a profiled objective. No Newton step was computed or applied; the spectrum is pointwise diagnostic evidence, not root or minimum qualification.

The parent reports completed/exit 0, no resource termination or monitor error, `203.744` seconds and sampled peak RSS `361,955,328` bytes under the 600-second / 1-GiB guard. The 3600-stage strict branch and all source, input, truth, parameter, tail, and runtime pins are unchanged. Raw result SHA256: `ca04e5f9712667cee95de69121d889dd8f2d33f81192b70e3724ee6fa52f224b`.

This closes only the requested fresh endpoint block/Schur diagnostic. The original three-hour stationarity/root search and response remain open; no result from another endpoint, including the earlier negative-curvature audit, is transferred here.
