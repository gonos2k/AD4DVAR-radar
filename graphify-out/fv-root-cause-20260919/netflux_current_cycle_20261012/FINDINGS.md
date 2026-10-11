# Minmod net-flux evaluation and one current-gradient cycle — 2026-10-12

The applied minmod Euler update computes `q + dt * ((incoming - outgoing) / area)` instead of separately subtracting outgoing and adding incoming flux. The discrete mathematical operator, boundary treatment, physical units and AD connection are unchanged; the FP64 arithmetic version changes. The existing common-face tolerance is unchanged. The constant-field regression distinguishes the old one-ULP drift from exact constant preservation, and checks the amplitude derivative.

## Cause and supported scope

Eight GREEN/RED teams investigated PR297's common-face failure before application; their 85-file original manifest and records are preserved in `eight_team_diagnosis.tar.gz`. Stored-vector high-precision projection ruled out the final projector as the dominant error. Replay on/off did not remove the state/residual discrepancy. A VJP split located the dominant tangent difference in the loss-cotangent/residual term (two final analysis residual entries), rather than in the projector or the small production Q alone. Temporary net-flux regrouping removed that residual gap in the screened current-point comparison. The earliest individual rounding operation was not proved; this evidence supports an arithmetic-path remedy in the tested domain, not universal exactness of the FV gradient.

Synthetic observations and truth were saved using the OLD evaluator before modifying the kernel. Otherwise `make_flow_case()` would regenerate the target and change the problem during comparison. The new backend checks `fixed_case.pt` SHA before loading it and uses its original observations, prior, parameters, boundaries, timing and truth. The fixed observations, prior, parameters, boundaries, schedule and truth remain unchanged, as does runtime. The full input identity intentionally changes when the control advances on the optimizer commit. No target is regenerated after application.

## New qualification and actual result

`PLAN.json` was frozen before launch (SHA `7f26c76bad511e1345c2d3c27b2073908c574587fc3aefd433fb44b98a932441`). All 201 source and 275 archive pins matched. Fixed asset SHA is `1e810c23c73b93d89b2a10b46d9ddea44a8217b07dcd3ebfebb462858e3fa9b7`; transport transitions from `93bacc2ec0aef671df488025b280d46d79ed6613acf78b33072efbd8e701738c` to `b0335b7a72ecd22084d5ab8ea0e46707240be15ef0039f4b31f96adbc937ee1a`. Only the known old-to-new transport hash is permitted among inherited sources. An independent new-version base repeat and componentwise full-G reconstruction qualify the unchanged `556a25d9…ed298a8b` control; archived old values are version deltas only, never new numerical P2. Requalification is not an optimizer commit. Native release is checked before strict-interior mixture construction. Old producers/receipts retain their source contracts.

At the freshly qualified base, tangent gradient-jump norm is `7.757919228897728e-18`, below the unchanged `7.975354596700906e-15` budget. Both outward normal costs are positive (`0.19436399229299176`, `0.08624385137770964`). The current gradient direction, two fresh directional HVPs, actual J/R candidate and independent final P2/full-G evaluation complete one cycle.

| Metric | New same-control base | New committed point | Decrease |
|---|---:|---:|---:|
| J | 0.06101963540295345 | 0.06101864864417508 | 0.00161712% |
| R | 0.0027594786201677354 | 0.002743440727443838 | 0.581193% |
| norm G | 0.0525307397641394 | 0.05237786486144542 | 0.291020% |
| max mixed gradient | 0.02625619077029916 | 0.026212827660186504 | 0.165154% |

New point: `0aafdf8069fcb8d5adc4ec813646707a3d1e083804631b17facb24664098655b`. The first actual candidate passed both J/R conditions and final repeat. Displacement is `1.883912762404316e-05`; alpha is `0.00035863054108483696`. The cost drop / linear prediction is `0.9970953561648895`. There are 2 fresh HVPs, 1 candidate, 1 optimizer commit, 0 GN row/solve calls and 0 future-E evaluations. The outer guard completed with exit 0 in 36.7741 seconds, sampled maximum RSS 536395776 bytes (~0.500 GiB), with no timeout/memory termination. Source, fixed input, runtime and deadline closure checks pass. `receipts.tar.gz` preserves the original raw step, base qualification, candidate, log and guard bytes; the manifest verifies restoration.

## Limits and next point

Termination is `commit_cap_reached`, not convergence. The retained top-level current-point direction/HVP operators belong to the step's 556a base. Final point operators have not been regenerated; `postcommit_readiness_performed=false`. The candidate limiter signature differs from the base, although the side pair agrees at the candidate and its independent repeat closes. The recorded one-direction model error and J/R acceptance certify this candidate only, not a uniform branch/Taylor model. The final stored-gradient projection in METRICS is a pure vector/geometry diagnostic, not a new FV direction/HVP evaluation.

The unchanged-control evaluator delta in J is about -1.21e-14 and in full-G norm about 3.55e-12. These are numerical-version differences, excluded from the optimization decreases above. The legacy terminal E remains an archived observation in the old version; it is neither a new baseline nor a selection criterion. No future score, eligible original minimum, exact curvature/adjoint, parameter reoptimization or independent forecast validation is claimed. Q=0 remains a conditional search-face hypothesis. Historical weighted progress remains 70%; it is not a solved fraction.

## Verification

Root affected tests: 91 passed in 7.34 seconds. GREEN independently ran an extended affected/numerical-range set: 115 passed in 4.30 seconds. Those totals overlap and are not summed. Both runs have 18 existing torch.jit.script deprecation warnings. The five new modules/tests have 0 basedpyright errors/warnings. Existing Graphify evidence was consulted and a scoped six-file AST update generated 87 nodes / 164 edges, without a semantic rebuild. GREEN and RED prelaunch and final reviews have no blocker; their receipts describe the scope. The final reviews independently inspect hashes/arithmetic/provenance and do not represent additional production FV runs.

To continue, first derive fresh direction, normal/branch support and required HVPs at the new point under the same fixed-input/new-version contract. One accepted cycle does not complete the higher qualification or independent forecast work.
