# PR177–195 generalization review resolution checklist

Source: user review of PR #177–#195, received 2026-09-25. This ledger
separates input support, forward execution, stationary response, independent
reanalysis, and physical skill. Prior evidence remains in
`PR175_176_REVIEW_RESOLUTION.md` and `POST_MERGE_REVIEW_CHECKLIST.md`.

| ID | Review item | Closure criterion | Current state |
|---|---|---|---|
| R1 | Preserve completed scope | Verify PR #177–#195 main integration, regular 3-hour forward result, fixed-control point likelihood and derivative tests, and process-isolated local responses without adding them as one success rate | Done at their reported scopes; no new FV run implied |
| R2 | Full-valid correlated point observations lack a stationary response | Fix one 4×5 synthetic point problem, source/input/branch/resource gates and direction; obtain a qualified stationary point, true adjoint residual and full parameter gradient, then compare signed nonlinear reanalysis at two local step sizes. Preserve numerical refusals as attempt evidence, not closure | Stage A and four bounded numerical attempts are recorded below. The locked exploratory endpoint had SPD Hessian but fixed-branch Newton refused at iteration 4; no qualified point root, adjoint or endpoints. **Open** |
| R3 | Missing/QC/one-empty-time point profiles have no stationary response | After R2, bind one declared incomplete profile at a time to its own stationary point, branch and signed endpoint checks; retain exact fixed masks and covariance subsets | Open; fixed-control checks alone do not close it |
| R4 | Two-hole collocated partial problem refuses final Newton correction | Keep the three original refusals; choose and predeclare a branch-aware nominal search policy separately from final-point local eligibility, then validate any new root and signed response | Open; no claim that a stationary point is absent |
| R5 | Point observations and 3-hour leads are in separate profiles | Define common time/trajectory and score semantics while preserving old one-lead and long-forward references; check a bounded combined forward run before any long-horizon response claim | Open |
| R6 | General observation products remain unsupported | Bind product-defined footprint/coordinate, censoring/QC provenance and covariance semantics separately; do not infer physical measurements from prepared synthetic masks | Open; needs a declared product contract |
| R7 | Service-level concurrency and result publication | Preserve process-level forward-AD isolation; verify cancellation, resource accounting, in-flight callback completion and publication ordering in a bounded service path | Open; current process result is two fixed local responses |
| R8 | Independent physical performance, finite influence and learning | Use independent verification events and a predeclared finite-amplitude range; keep learned error/prior normalization and data splitting separate from local synthetic derivative checks | Open; current truth is same-operator synthetic |

Do not relabel skipped CI jobs or focused tests as a full CPU/package
regression. Preserve the user's working-tree `AGENTS.md` change and all
historical numerical reports; new attempts must have distinct records.

## R2 sequential evidence

- [x] Source- and resource-bound correlated full-valid point preflight:
  26 controls / 13 parameters / 54 minmod stages; warm gradient maximum
  15.15045. See `FV_POINT_NOMINAL_ATTEMPT1_RESULTS.md`.
- [x] One direct warm exact-Newton feasibility run: first PCG refused
  non-SPD curvature after seven HVP calls; no candidate step.
- [x] One fixed-response-margin L-BFGS exploration: 13 accepted steps,
  objective 0.07993508→0.00644675, then all 16 next candidates refused
  by the combined stage/slope/face gate. See
  `FV_POINT_BASIN_ATTEMPT1_RESULTS.md`.
- [x] One separately planned mathematical-strictness search: 100 accepted
  steps, objective→0.001092566, 22 pointwise branch changes, gradient
  maximum 0.0108053; `step_budget`, not a stationarity handoff. See
  `FV_POINT_BASIN_ATTEMPT2_RESULTS.md`.
- [x] Locked last search endpoint: exact seed Hessian locally SPD, four PCG
  solves converged, but fixed-branch Newton iteration 4 refused all 16
  candidate steps before objective/gradient evaluation. See
  `FV_POINT_TERMINAL_NEWTON_ATTEMPT1_RESULTS.md`.
- [ ] Find a qualifying stationary point under a separately declared
  branch-aware or mathematically strict root-search policy. Keep final
  gradient `<1e-10`, exact SPD and robust branch/margin gates separate.
- [ ] Only then compute the fresh full 13-component adjoint/VJP response and
  two adjacent signed nonlinear reanalysis pairs; keep physical skill and
  finite-impact validation separate.
