# R4 original two-hole input: conditional active-face response

The original 4×5 collocated two-hole problem now has a numerically supported **selected-active-face response**. This is a new, explicitly declared response contract for the same observations, background dependency, full 26-latent prior, score, boundary and time schedule. It does not reopen the original smooth 26-control response API. No prior mean, objective scale, mask, or tolerance was changed to manufacture a root.

## Mathematical scope

The selected event is `qy[2,0]=0`, a zero normal volume flux, not clear sky or a missing observation. Four projected streamfunction modes enforce it structurally. The recovered fifth coefficient remains in the original open coefficient domain; its latent value contributes to all 26 original prior terms. The 25 tangent coordinates retain 20 field controls, four flow coordinates and growth.

For restricted objective j(t,p) and score e(t,p), the calculation uses

    H_tt^T lambda = e_t
    g_p = e_p - j_tp^T lambda

with the existing matrix-free local-response and PCG implementations. Qualification requires fresh tangent stationarity, numerical positive tangent curvature, all other strict minmod/upwind branches, and strictly negative left / positive right increasing-eta normal derivatives. The independent 80-digit interval-jet calculation encloses those normal derivatives at each captured approximate endpoint. It audits the 36 analysis stages and 48 nonselected face signs per stage; the full 54-stage native trace additionally covers forecast stages.

These measurements support a conditional local active-face calculation at the tested points. They do not prove an exact stationary root, a uniform parameter neighborhood, a global minimum or uniqueness. Ambient native 26-control gradients at the nonsmooth event remain nonzero and are diagnostic only.

## Actual execution

The baseline tangent gradient maximum is 7.947620860038196e-12. Its fresh 25-column Hessian audit has minimum eigenvalue 1.031278100528556, maximum 4772.621686532755 and numerical condition estimate 4627.870682104726. Dense columns are used only for qualification, not for solving the adjoint, predictor or endpoint corrections.

One middle-time common-bias direction includes all 20 observation slots; 19 are active. Both missing slots have zero direct, indirect and total sensitivity. The full 61-component vector is returned, but nonlinear validation covers this one direction only. Its direct term is zero and total response is -0.004418014129384397. The independently recomputed transpose-adjoint relative residual is 4.396969025416272e-11; the predictor true relative residual is 4.2538722943795115e-11. Each solve takes 25 PCG iterations.

| Per-slot step, dBZ | Actual central score difference | Relative difference from adjoint |
|---|---:|---:|
| 0.00025 | -0.0044180140920598325 | 8.448267333021634e-9 |
| 0.000125 | -0.0044180141200588115 | 2.110809329632842e-9 |

All four endpoints pass their own fresh tangent gradient, curvature, normal orientation and branch checks. Each requires one Newton correction. Absolute difference falls by 4.002382979087524, with observed order 2.0008592221148915. Actual changed p enters objective, refinement, branch checks and score; these are nonlinear reanalyses, not merely projections of the reported vector.

The guarded child exits 0 in **234.12348483409733 seconds**, with sampled peak RSS 354041856 bytes (not a continuous system-memory bound), no resource termination and no monitor error. Raw `timings.total_seconds=214.2546293749474` starts after baseline qualification and covers response/predictor/endpoints; it is not the complete producer time. Response-call time is 21.2111 seconds and predictor time 17.7485 seconds.

## Verification and provenance

- Focused source tests: 18 passed, 18 existing torch.jit warnings. These are kernel/contract regressions, not 18 FV reanalyses.
- Changed modules/tests: zero error-level basedpyright diagnostics. Lower-level diagnostics were filtered; no all-level warning-free claim.
- GREEN and RED independently audited terminal raw/resource records, source/input pins, actual endpoint parameters, score differences and qualified limits: both GO.
- The first two independent normal-evaluator attempts failed for programming/setup reasons. Their exact source snapshots, logs and resource records are preserved; neither is a scientific refusal or completed normal result.
- Cached structural graph was consulted and changed code was refreshed with bounded AST-only Graphify. The shared graph remains unchanged because of its recorded multigraph integrity issue. No semantic rebuild or whole CPU/package/deployment CI was requested.

Raw evidence: `partial_active_face_stationarity_attempt1/`, `partial_active_face_normal_attempt1/`, `partial_active_face_normal_attempt2/`, `partial_active_face_normal_attempt3/`, `partial_active_face_response_attempt1/`. Plans and the hash manifest bind the distinct attempts. Historical smooth-policy and signed-slice refusals remain unchanged.

## Checklist decision

- [x] R4-A: original two-hole statistical input, selected-active-face tangent stationary calculation, full parameter VJP and one direction / two-size signed nonlinear validation.
- [ ] R4-R: original smooth 26-control API remains unsupported at this event; a classical smooth-root claim is not made.
- [ ] Original zero-centered point input (R2-O-R) and long-horizon point stationary response (R5-R-R) remain separate unresolved research targets.
- [ ] Product-defined real radar support and independent physical/learning evaluation (R6/R8) require external data and provenance; synthetic score validation does not close them.

No new physical forecast skill, observation identifiability, finite-amplitude validity range or generic service capability is claimed.
