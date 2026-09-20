# PR169 follow-up: checkout portability and full parameter VJP

Baseline: merged PR169 `c4ba18b7`. Historical JSON and measured sources remain
unchanged. New measurement: `minmod_parameter_vjp.json`.

## Change and mathematical contract

The comparison probe maps each archived absolute source path relative to the
explicit producer root `/Users/yhlee/ADVAR`, then reads the full relative path
under current ROOT. SHA256 is unchanged. Relocated-root tests preserve duplicate
basenames in different directories, reject changed content and outside-root paths.
This closes the reviewed checkout-path P2, not a certification of arbitrary caches.

`compute_local_response` now returns `direct_gradient`, `indirect_gradient`,
`total_gradient` (all parameter-shaped). One parameter VJP of the control gradient
computes `indirect_gradient = -J_cp.T @ adjoint`; total is `E_p + indirect`.
Existing directional JVP/mixed diagnostics remain for a small supplied direction
set. Callers should not construct one direction per pixel: use the full gradient.
The retained mixed diagnostics still cost O(K*N_control).

No governing equation, limiter, boundary, observation semantics, PCG tolerance,
stationarity gate or public donorcell restriction changed. CPU FP64 and the serial,
conditional full-support 4x5 adapter remain the supported experiment scope.

## Actual measured projections

| Direction | Relative difference: full VJP projection vs directional response |
|---|---:|
| sine | 8.4063081e-14 |
| theta | 4.0594734e-11 |
| middle-time bias | 2.8547858e-16 |

All satisfy the predeclared 1e-6 relative comparison gate. The 61-entry vectors
are stored, with direct and indirect terms separate. Theta remains strongly
cancellation-sensitive; response-specific comparisons are retained.
The stationary point, 26-column HVP/dense comparison and old three directions
are reused. No optimizer or nonlinear reanalysis is rerun. The complete report
includes actual residual, fresh/mixed gradient comparisons and source hashes.

## Actual analysis-result boundary

The regression loads `minmod_spatial_inverse.json["product_gn"]["control"]`, not
the polished top-level control. It evaluates the current real FV objective and
54-stage branch check, then the unchanged `<1e-10` gradient gate rejects the
unqualified GN state before PCG. Mocking PCG to fail if called confirms order.
The archived GN maximum gradient is 1.334876e-4. Its tiny objective gap does not
qualify it for implicit response. Conversely, the actual bounded measurement
accepts the archived polished control (gradient 6.9620403e-12).
This tests ingestion of saved real solver output, not a fresh end-to-end GN run
or a new automatic refinement path. Product GN stationarity remains open.

## Costs and verification

Actual internal wall times for this bounded run:

- Preparation / identity checks / fixed verification: 0.099 seconds.
- Response call (branch trace, gradients, PCG, VJPs, 3 direction diagnostics): 23.639 seconds.
- Post-call dense/HVP/projection comparison: 20.058 seconds.
- Internal total: 43.795 seconds.

No peak-memory measurement was made for this run; previous RSS is not reused as
new evidence. These timings do not certify whole-chain D7 or grid scalability.

65 distinct focused tests passed (18 existing torch.jit deprecation warnings).
Pinned basedpyright 1.39.9: 0 errors, 0 warnings, 0 notes. Full CPU/package CI not
requested. Graphify code-only refresh: 6,651 nodes, 65,018 edges, 238 communities;
no semantic LLM extraction. Original animation data is unchanged.

General minmod FSOI, finite-path/finite-impact validity, product GN stationarity,
typed mean/precision/support learning, and whole-chain D7 remain open.

Final GREEN and RED reviews found no blocker. Aside desktop snapshot and screenshot
confirmed the new table in the original demo (`parameter_vjp_desktop.png`).
No new mobile verification is claimed.
