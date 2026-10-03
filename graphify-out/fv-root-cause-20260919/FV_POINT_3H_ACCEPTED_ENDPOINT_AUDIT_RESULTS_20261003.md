# S4 fresh accepted-endpoint Hessian audit

A new exact-Hessian audit completed at the accepted endpoint from the one-step S4 experiment. The accepted control SHA is `d74b35f3cb8c9f70d3eeb98b2fffc2efc5492e818fe6de12abece6276f648ba8`; the original 13-parameter, fixed problem, terminal truth, and 3-hour input remain unchanged. Fresh J, Phi, and all 26 gradient components match the accepted-step record with zero saved/fresh difference under the declared component-scaled budgets. The endpoint branch replays as strict for all 3,600 stages.

The endpoint is strongly nonstationary: `J=0.4254527702`, `Phi=485.7293098`, full gradient infinity norm `22.08953658` and L2 `31.16823093`. Field gradient L2 is `18.76278381`; the six dynamics coordinates have L2 `24.88808075` (flow-five L2 `11.46599045`, growth `22.08953658`).

The fresh 26-column Hessian has symmetry relative error `1.50e-15`; its independent 27th minimum-eigenvector HVP passes with relative residual `1.48e-16`. The symmetric-part full-H eigenvalues range from `-21.8270` to `3712.6168`. `Hff` is numerically SPD here, with eigenvalues `[0.492028, 1646.850]`, condition `3347.07`, and positive floor `1.64685e-5`. Both field solves pass (`Hff⁻¹gf` residual `5.92e-16`; `Hff⁻¹Hfd` residual `4.48e-16`). The six-dimensional Schur spectrum is `[-406.3280, -241.5944, -20.2159, -4.8621, -0.7137, 242.0520]`: five negative directions. No step was formed. This is local curvature evidence at a nonstationary accepted endpoint, not a minimum or root result.

The parent reports completed/exit 0, no resource termination or monitor error, `201.217` seconds and sampled peak RSS `368,033,792` bytes under the 600-second / 1-GiB guard. Parent/child source maps (40/27 entries), fixed input identity, and CPU FP64 runtime are unchanged. No score, adjoint, response, optimizer step, or physical validation was performed. The original three-hour stationary-root and response work remains open; no Hessian or curvature evidence is transferred from the prior endpoint.

Audit raw SHA256: `b20c5f85cc8921d8cbff69eddac6167c453739906b7a05f5ac1737df405147e6`.
