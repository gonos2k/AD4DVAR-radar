# PR227 field endpoint: one fresh 20+6 Schur diagnostic

The guarded child completed one CPU FP64, 26-column exact-Hessian linearization at the pinned field-corrected endpoint. It applied no step and ran no optimizer, response, or reanalysis. The result is local curvature/Newton-algebra evidence; the full gradient remains nonstationary, so it establishes no full-control root, response, normal condition, or physical skill.

| Evidence | Result |
|---|---:|
| Control SHA256 / parameters SHA256 | `a78d1b8e0cd8f997572f5e2a913f7023dcd6333c124ac4b7c9414358f8484e26` / `e3a45fac86e622332c6afd3472b46d58ef97cd37db1445260c86923a3c423bf9` |
| Fixed-input identity SHA256 / objective | `838fcf77a43b209b53dfe62f6a1657d0f8eba5a41797e431451059f3f8d18158` / `0.020764957942635602` |
| Field gradient max / dynamics gradient max | `7.0329883410e-12` / `0.0061284025566` |
| Dynamics gradient (g_d) | `[3.68687997e-6, -0.00349963117, -0.00612840256, -0.000878404991, -0.00261976289, 4.11087112e-6]` |
| Full-H eigenvalue range | `[1.016770927, 4772.724338]` |
| (H_{ff}) eigenvalue range / condition | `[38.81446632, 4772.482193]` / `122.956275` |
| Schur (S) eigenvalue range / condition | `[1.016778252, 56.59251349]` / `55.658659` |
| Schur RHS (-g_d + H_{df}H_{ff}^{-1}g_f) | `[-3.68688e-6, 0.00349963117, 0.00612840256, 0.000878404991, 0.00261976289, -4.11087112e-6]` |
| Full Newton step parity, absolute / relative | `6.95679e-19` / `1.39383e-16` |

The accepted algebra includes the nonzero (g_f) correction in the right-hand side. The reconstructed direction has maximum absolute field component `2.81576e-5` and dynamics components `[-0.000472343, 0.002971956, 0.003818974, 0.000742074, -0.000847237, -0.000019091]`; it was not applied. Relative residuals were `3.29937e-17` for `Hff*y=gf`, `3.77181e-17` for `Hff*X=Hfd`, `3.57331e-18` for the Schur solve, `4.34584e-19` for `H*step+g`, and `3.88070e-19` for the block equations. All passed the declared `1e-10` gate.

The endpoint's frozen 54-stage branch stayed unchanged (minimum scaled face/slope margins `1.16627e-4` / `4.06087e-4`). The parent reports completed/exit 0, no monitor or resource termination, 22.291 seconds, and sampled peak RSS `347,176,960` bytes under the 120-second / 1-GiB guard.

Before/after pins agree for the 23 child sources, input identity, branch, J, gradient, endpoint raw (`8ad5b5e3…2d9442cf`), plan (`a54b0546…a0da26`), tests (`9d1fe1a2…7a89fe4`), and probe (`2c60602c…9684d6`). The parent preflight's 27-file source map also matches current files before and after. The focused pure suite passed 10 tests (18 existing TorchScript warnings); offline basedpyright reported 0 errors and 117 warnings.

Raw result SHA256: `0eb9beb677c4a72a5b0d8559c0a94482faef29ab582a389ecb6a7a9878b8bbd4`. Curvature at this nonstationary endpoint does not qualify minimization or resolve the separate nonsmooth active-face question.
