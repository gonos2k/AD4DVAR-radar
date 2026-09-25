# R2 attempt 3: exact Newton feasibility from a locked exploratory endpoint

Attempt 2 ended at its **100-step search budget**, not at its planned
`||grad J||_inf<=1e-4` handoff. Its child report SHA256 is
`fbafbab3736abac64421356b67ab0896ea31976e31b0c43f41eea933414e7a51`.
The deterministic last accepted control has FP64-byte SHA256
`8f5110f2755217dece8ca9f6bbab23ceadb184b4f1a32b8e4654ddfde5f4b258`
and full limiter/face-sign signature digest
`0783f66a062d5144894513c65a4e221befca4d23a22d8fba0adc43ead5011327`.
Its objective is 0.001092566268397464, maximum gradient
0.01080526397720594, and scaled slope/face margins are
0.00019559368949964172 / 0.00035249320414798194. This is a **new
predeclared start policy** that asks whether exact Newton can polish that
specific endpoint; it does not relabel attempt 2 as a passed handoff.

Use the same full-valid 4×5 correlated point problem, fixed p, C, background,
boundaries and synthetic verification. Reconstruct only that final control
from the pinned raw report. Require its control hash, fixed problem/input
hashes, objective, gradient, full 54-stage signature and both `>1e-4`
margins to match before any Hessian or Newton work. The warm-to-endpoint
search crossed 22 observed signatures; no fixed-branch path from the warm
control is claimed and no alternative accepted event or seed may be chosen.

First build a fresh exact 26-column gradient-JVP Hessian at this locked
seed. Require all entries finite, relative antisymmetry
`||H-H.T||_F/max(||H||_F,tiny64)<=1e-10`, and for the symmetrized matrix
finite `lambda_max>0`, `lambda_min>0`,
`lambda_min/lambda_max>sqrt(eps64)`. If not, stop with
`seed_curvature_refused`; do not shift or substitute H.

Only after this gate, call the unchanged `refine_stationary` with that one
control and the pinned terminal signature. Keep 8 Newton steps, 16
backtracks, 104 PCG iterations and the actual Euclidean relative Newton
residual `<=1e-10`. Every candidate must pass the strict 54-stage trace,
the same signature and both scaled margins `>1e-4`. A PCG, branch, Armijo,
nonfinite or budget refusal is `newton_refused` with recorded reason.
Success requires a fresh maximum gradient `<1e-10`, final full signature
and margins, and a repeated finite/symmetric/SPD exact Hessian audit at the
polished control. Do not run an adjoint, full parameter VJP or signed
nonlinear reanalysis in this attempt. `response_validation` remains
`not_performed` even if nominal stationarity is obtained.

One serial child has a 600-second wall cap and sampled 1-GiB child RSS cap.
The outer runner checks actual elapsed time, RSS samples/peak, PID, exit
status, declared status-specific evidence and source/input/plan/preflight/
attempt-2 hashes before publishing. Sampled RSS is not an OS hard memory
limit. The new result is one local-root feasibility diagnostic, not proof
of a unique basin, finite branch path or physical forecast skill.
