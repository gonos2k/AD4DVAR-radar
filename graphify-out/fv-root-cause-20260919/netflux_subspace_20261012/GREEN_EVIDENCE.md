# GREEN evidence — PR300 endpoint and stored finite-error diagnosis

## Stored vectors and candidate payloads

The comparison run stores enough material for the nearest rejected/accepted GN finite-error comparison without another FV or HVP evaluation. The relevant artifacts are `netflux_direction_comparison_20261012/attempt1/step.json`, `attempt1/operators/GN.json`, and `attempt1/robust_gn_coupled/candidate_09.json` plus `candidate_10.json`. The step record contains each arm's 26-control direction, two 26-entry side HVPs, 27-entry base residual and residual direction, candidate alphas, and repeated-P2/full-G approvals. Candidate receipts add actual 27-entry `G`, one-sided gradients, and full minmod selector traces. The GN operator receipt contains all 24 row gradients and the Woodbury solve vectors.

The finite error is reproducible directly as

`e(alpha) = G(candidate) - (G_base + alpha * DGd)`.

For the nearest rejected GN candidate (#9, alpha `0.00303690765`) and accepted candidate (#10, alpha `0.00151845382`), the stored norm matches this vector reconstruction exactly. The supported control layout is 20 echo controls, five flow latents at `[20:25]`, one growth latent at `[25]`, plus the normalized face residual at index 26. The earlier proposed `[0:12]/[12:17]/[17:26]` grouping is inconsistent with the implementation: flow latents are explicitly read from `control[20:25]` in `examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py:311-313` and the face chart uses those same indices in `examples/weather_scenarios/fv_point_3h_face_transition_diagnostic.py:84-102`.

| Candidate | Status | Echo error L2 | Flow error L2 | Growth error | Face error |
|---|---|---:|---:|---:|---:|
| GN #9 | rejected by R Armijo | 0.00813264 | 0.00012900 | 0.00002869 | 2.07e-18 |
| GN #10 | accepted | 0.00377652 | 0.00005894 | 0.00001595 | 2.07e-18 |

The largest stored echo error is component 13. The side-gradient jump is zero across echo/growth coordinates and nonzero in the flow block. From b647 theta `0.6931095102` to d19e theta `0.6946385051`, the `delta_theta * (gplus - gminus)` correction has flow L2 `4.2610e-4` and zero echo contribution. This theta correction therefore does not account for the dominant echo finite error.

Candidate #9 and #10 each report a changed selector signature relative to their base. Comparing their stored traces shows nine `choose_left` changes on each side and no left/right/slope sign changes. The base selector arrays are not stored with the comparison step, so this is only an adjacent-candidate count; it cannot identify which base-to-candidate branch changes caused the error.

The reproducible, receipt-only report is `candidate_model_error_blocks.json`; its source is `diagnostic.py`. Both are copied here and retained beside the PR300 source evidence. The diagnostic reads candidate bytes from the pinned receipt tar, not loose candidate files.

## d19e receipt lineage and fresh-preparation boundary

The d19e endpoint is the one committed gradient candidate from the PR300 comparison, with control SHA256 `d19e882d31b634ef0be755e6419b57ecf3691734efc736d5f40ea18c3ac4145e`. The pinned plan hash is `36686ccf05c55ab9dff1d6c0c4658e14049a108eba402b76c86fc0b9ac904b35`; the source-bundle hash is `556fec30ca5b0a06c0b00601c8652b616b02705a09d7ac82fcea3203fd0c46bc`; the receipt-manifest hash is `2d1467b35d1bd99bacbf4f0cf16a9596c57e31ae963f31b29481d2ccffa233a7`; the receipt-bundle hash is `55f220c1f5da91a2a76175984e419049470acb324aef8b34b1eacd36e77a4040`; and the raw child hash is `1c79690de1d8e42ff4accf5e6b3bd62dac5c55c0607d3cc8b771fb6b6a8b4703`.

`fv_point_3h_netflux_direction_comparison.py:43-61` admits the plan and preserves the b647 parent source/archive pins through `continuous_endpoint.load_endpoint()`. The new `fv_point_3h_netflux_comparison_endpoint.py` rechecks source bundle members, receipt-manifest member hashes and byte lengths, the run-to-resource link, the raw child hash, both independent eligible arms, repeated-arm selection, and the selected candidate-receipt linkage. It then reuses `current_gradient_endpoint._close_step` to validate the closed step's proposal, repeat, previous full-G, final full-G, and control digest. It reads packed receipts and does not invoke `CurrentBackend`, `_observe`, a live loose receipt, or a production operator.

The d19e receipt explicitly has `postcommit_readiness_performed: false`, and its HVPs belong to the earlier b647 control. A fresh two-direction attempt still needs a new native side-pair/current-P2 check at d19e, carry-forward theta and minimum-mixture P2 matches, the native face-release gate, the fresh gradient and curvature directions, four fresh HVPs, candidate search on both arms, and independent repeated P2/full-G approval before commit. The b647 HVPs cannot qualify d19e.

## Verification and limits

- `tests/test_fv_point_3h_netflux_comparison_endpoint.py`: 2 passed; the test blocks production backend/observation calls and rejects a changed manifest.
- `tests/test_fv_point_3h_netflux_subspace_trial.py`: 3 passed; synthetic checks cover HVP4/row24/two repeats and composed-DG linearity, no baseline fallback after a corrupted second repeat, and no commit on GN timeout.
- Basic Pyright for the endpoint: 0 errors and 0 warnings.
- The comparison's broader stored evidence reports 43 focused tests and an independent GREEN/RED audit in `netflux_direction_comparison_20261012/FINDINGS.md`.

These records establish vector arithmetic and receipt lineage. The branch-trace difference is correlated evidence, not a sole-cause proof; no d19e FV evaluation, d19e HVP, exact constrained-curvature claim, or forecast validation is included.
