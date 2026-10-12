# Original-problem response/reoptimization path — prepared, gated

The current fixed-case interface is `FVPointResearchProblem`: controls26, parameters13, observation_shape3x4, theta_index12, analysis times0/600/1200s, terminal score at12000s (180min after20min analysis). The source interface separates flattened observed dBZ parameters0:12 and fixed-background-pattern coefficient at12. The coefficient enters background_dbz + p[12]*background_pattern; it is distinct from the face mixing weight. These are observation/background parameters, not new wind or growth controls.

No exact response or perturbed reanalysis was run in this session. The admitted2a4d point and the comparison's newfb0a point are only conditional Q0 accepted points; their full residual is about0.049 and not an eligible original minimum. P2 approval is not this prerequisite.

Execution order for each future parameter-response job:

1. Admit a separately certified original local minimum with resolved native active structure, full residual criterion, exact reduced curvature and matching numerical/input/runtime pins. Explicitly check whether Q=0 remains the original problem's supported active hypothesis; release the face if required.
2. Freeze a new input-transition plan at the same observation/forecast domain. Construct p+h*v and p-h*v, with v initially either one detected observed dBZ entry or the background-pattern coefficient atindex12; choose h and h/2 from that component's units/dtype/error scale, record h-convergence rather than approving a single finite difference. This job must intentionally change its parameter hash and verify all other inputs/truth remain fixed.
3. Reoptimize the original native J separately for both parameter endpoints using fresh native gradients/directions/HVPs; do not reuse unperturbed or cross-endpoint operator caches. Require the same original final stationarity/curvature/active-structure qualification at each endpoint. Search-GN or coefficient-solver derivatives are not certificates.
4. Evaluate terminal E only after each reoptimization, from the fixed verification field. Compare the centered endpoint difference against the actual original optimal-condition adjoint F_z^T lambda=E_z and E_p-lambda^T F_p. A failed endpoint qualification or unresolved h-response stops the response job without fabricating a derivative.
5. Record solver execution, nonlinear convergence, exact-response agreement and independent forecast validation as separate outcomes. Matched synthetic fixed-truth scores remain observations, not independent weather validation.

This path uses the existing original cost/score interfaces and isolated parameter-transition plans. Current frozen search/score plans intentionally require fixed p; they cannot be silently reused for perturbed reoptimization. The outstanding minimum/curvature gate prevents execution now. No new solver, tolerance relaxation or implicit-GN substitution was added.
