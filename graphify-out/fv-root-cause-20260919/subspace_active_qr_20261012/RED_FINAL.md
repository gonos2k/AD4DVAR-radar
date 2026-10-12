# RED final: active rank-two augmented-QR regression

The rank-two SPD case is materially different from the earlier rank-one
regression. With `H=[[1,-1],[-1,1+1e-8]]`, the small eigenvalue is about
`5e-9` and the normal-equation small eigenvalue is about `2.5e-17`. In the
observed LocalTorch 2.13 pre-patch run, the solver returned a finite result,
but underfilled the radius by `1.18e-8`; at perturbation `1e-9`, the underfill
was `5.68e-7`. No exception was reproduced in that environment, and the
failure mode may vary with BLAS/torch versions.

The independent oracle treats the binary64 inputs as exact values, computes
the eigenpair and active KKT root in 80-digit `Decimal`, and covers both unit
small-eigenvector `G` and the original scaled `G=(-1e-8,-1e-8)`. The padded
`(26 controls, 27 residuals)` case also checks a condition-scaled coefficient
error bound and a near-machine-precision movement-radius bound. The scaled
gradient reference solves the general spectral KKT equation rather than
assuming that the represented `G` is exactly a small eigenvector.

The final shared augmented-QR/root-closure patch passes the new checks. The
fixed-small-eigenvector scale gradcheck validates the implicit multiplier
derivative on its smooth active branch; the inactive near-singular case
requires finite gradients. Existing identity/skew and rank-one gradchecks were
reused.

Verification passed: `.venv/bin/python -m pytest
tests/test_fv_point_3h_active_qr_regression.py
tests/test_fv_point_3h_subspace_model.py
tests/test_fv_point_3h_subspace_rank_regression.py -q` — 27 passed. No FV,
HVP, candidate, or score run was performed. I did not edit solver/core code.
