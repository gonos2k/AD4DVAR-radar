# RED code preflight: coupled robust-GN comparison — 2026-10-10

## Clearance

No actionable RED blocker remains for the frozen single comparison run. This is a code/plan preflight only; RED did not execute FV, seed reconstruction, gradients/HVPs, tests, or the guard.

Frozen plan SHA-256 is `18be87cdbb6d084ff7248339bf81b749edca4e55dbd9d3a3d9d692d5c37f611d`, with 137 source and 141 archive pins; I independently verified all 278 pin digests against the current files (zero mismatches). Relevant source hashes are adapter `4b2091f1c48763ed1d6b77b4dacd05942427057daded4afc2144af7aa998f85a`, shared continuation `dcfbf4e851deb017f3f54349524d4a69d5e65903044f4ab3a6da1e54faa71061`, minimum-model override helper `eb8568c028d725c22ec64f8859fedc60d418e9a12acb5be370be03fc75f06374`, and test `dd63cc1ecfb544249062b6595a586833e83b76c24ab3ad1009420ad0ce472f21`.

The loader pins the corrected 64-character predecessor raw digest `2a36ae30ec9749af1b212ef3395c8125fd46a18ca77956807ddc73ca5e9cb0ff` and accepts only the closed endpoint `c8fba1f9…`, theta `0.4818866600367756`, J `0.06123349298793274`. The scope is the fixed all-detected 12-row profile with pseudo-Huber delta 2, identity control prior, zero field-smoothness weight, no neural prior, and matching residual/projected-row parity across both face extensions.

The operation path matches the frozen budget: 24 scalar row reverse products; two direction arms; two fresh selected-face HVPs per arm; one Cholesky/solve of the `12×12` row-space matrix `S=I+BBᵀ`; zero PCG; at most 16 candidates per arm and one final commit. The code forms a small ambient `26×26` face projector for tangent products, but does not form or solve a `26×26` Hessian/system. The coupled arm's same `gn_direction` is validated as a chart tangent, used for both HVPs, passed to the minimum-mixture model override, and then used for side products, full theta-prime residual derivative, merit model, and chart candidates.

I read the frozen verification logs: 83 focused tests passed with 18 existing TorchScript deprecation warnings in 29.58 seconds; type checking reports zero errors/warnings/notes. The analytic 2-D Woodbury oracle and direction-override/JVP test cover the critical sign, SPD/tangent descent, theta-prime, and full-DF path. The synthetic test and receipts do not replace a production run.

No claim follows about exact constrained curvature, minimum/root, global smoothness, response, reanalysis, or forecast skill. The planned 600/660-second, 1-GiB guarded run remains a single same-point direction comparison, with actual J and minimum-Psi/F² Armijo and final closure deciding any one commit.
