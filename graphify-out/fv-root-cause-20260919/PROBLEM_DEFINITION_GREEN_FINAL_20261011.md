# Problem definition: GREEN review

Read-only source/evidence cross-check of `docs/FV_INVERSE_PROBLEM.md` and `README.md` lines 83–93. No source, plan, pinned code, or numerical execution changed.

- The definitions align with `FVPointResearchProblem`: 26 normalized controls, 13 parameters, β at parameter index 12, and vartheta reserved for the side-gradient mixture.
- The timeline is consistent: observations at 0/600/1200 s; 18 ten-minute forecast leads end at 12,000 s absolute, 10,800 s after the last observation. The profile has 360 analysis and 3,600 combined SSPRK2 stages.
- The documented observation operator, same-time whitening, robust objective, identity control prior, zero smoothness weight, terminal 4×5 dBZ MSE, and separation of J from E agree with source.
- Q and the 0.84 scale are now identified as configured face-flux quantities, with basis-dependent coefficient units and no physical-wind interpretation. The saved static checker independently obtains Qy[4,3]=0.10399999999999998 by both polynomial difference and dot product; it also confirms no Q=0 truth constraint, counts 26/13, and times 12,000/10,800 s.
- The caveats correctly separate strict-branch support from physical domain membership, finite candidate gates from minimum certification, and the current endpoint from an eligible minimum, adjoint, or forecast validation.
- The separate semantic graph parses as valid JSON with 5 nodes, 4 extracted edges, and source locations on every node and edge. The final incremental Graphify snapshot has 13 nodes and 22 edges, including the five doc concepts and their four extracted relations.
- Final wording now explicitly uses `R=‖G‖₂²`, limits the positive-Hessian sufficiency claim to supported nondegenerate points, leaves degenerate minima unresolved, and distinguishes per-step J/R policy gates from mathematical necessity. Verification records 16 resolved source links; all checked README/source targets exist.

No substantive blocker found.
