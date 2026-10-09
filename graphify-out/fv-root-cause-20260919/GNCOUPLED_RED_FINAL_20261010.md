# RED final audit: same-point coupled robust-GN comparison — 2026-10-10

## Disposition

The saved one-guard comparison completed one actual control commit, selecting the coupled robust-GN direction under the predeclared actual minimum-F²/J rule. Both arms started at the same saved c8fba1 endpoint and produced a candidate passing actual native-J and fresh-minimum F² Armijo, face/current paired-branch, strict interior theta, nonzero movement, and final per-side closure. The selected GN endpoint is a finite local improvement at this one base; it does not establish general superiority, stationarity, curvature, or convergence.

RED analyzed saved raw/gzip/run/resource receipts and saved arrays only. No FV, seed, production gradient/HVP, test, or guard rerun was performed.

## Same-base comparison and selection

The fixed base is control SHA `c8fba1f93ee0b6c83edc5b56792de158921c0a64bab41eef5403c2a514b1fc33`, theta-star `0.4818866600367756`, native `J=0.06123349298793274`, and fresh-minimum `F²=0.004605971728402105`.

| Arm | First accepted alpha | Actual path norm | Endpoint J | Fresh-minimum F² | F² decrease from base | J decrease from base |
|---|---:|---:|---:|---:|---:|---:|
| Baseline tangent | 0.000816437838 | 5.54094e-5 | 0.061229742391 | 0.004586433310 | 0.424198% | 0.006125% |
| Coupled robust GN | 0.011531772605 | 5.78088e-4 | 0.061202333948 | 0.004490021942 | 2.517379% | 0.050886% |

The GN arm's accepted alpha is its initial cap divided by 64. Its six larger actual candidates passed native-J, current paired branch, face, and interior theta checks; each failed only the actual minimum-F² Armijo test. Candidate seven passed both actual merit gates and final closure. The baseline's first candidate also passed, but had a larger endpoint F², so the predeclared selector chose GN. The accepted candidate theta-star `0.4829044955080727` recomputes from its saved side gradients and matches the final repeat exactly. Final face Q is zero; the base and endpoint branch signatures differ, so paired endpoint acceptance does not establish smoothness of the path.

The local quadratic F² model predicted decreases of `0.536675%` for the baseline and `2.692548%` for GN; actual decreases were `0.424198%` and `2.517379%`. Both local models were somewhat optimistic here. They did not add acceptance criteria; the actual candidate function values and final repeat support the selected endpoint.

## Independent saved-array checks

At the selected base, the raw predecessor closure independently gives the same theta-star as the current-point working theta. The four HVP rows are complete and split two per arm; within each arm, both sides use the same direction hash, base SHA, carried theta, and working theta. Recomputing

`theta' = -[j·h_mix + g_star·(h_+−h_-)]/(j·j)`

from the saved base side gradients and each arm's saved HVPs matches the arm's stored value. The reconstructed full residual derivative agrees componentwise with the saved direction derivative to roundoff, and `F·DF` matches the envelope slope for each arm. This supports that the HVPs, direction override, theta-prime, and local alpha model refer to the same direction.

The coupled audit's saved `B` is `12×26`; recomputing `S=I+BBᵀ`, its Cholesky solve and Woodbury direction matches the saved arrays (direction max error `3.5e-16`, solve-vector error `5.5e-16`, solve residual `2.0e-15`, `lambda_min(S)=1.00163`). The saved D values recompute from the twelve residuals and delta 2 with zero array difference. Projected side row parity max error is `2.45e-15` against max budget `1.07e-12`; residual-value parity error is zero against budget `1.05e-11`. Reconstructed native J matches the observed base J exactly. The coupled metric's tangent descent product is `−0.00277225`, and direction normal residual is `−2.17e-19`. This verifies the implemented row-space solve and saved direction locally; it does not verify general GN accuracy or positive exact constrained curvature.

## Provenance, resource and limits

Raw SHA-256 is `362e8e0aa7cf12202618c11873633201d23ddc2af835ac67a5d499612edb4599`; gzip SHA is `bdf479a14b0ba0e3342a8dc630e9bce2c89ac83fd4fea4625d17b2a01e507b84`, and decompression reproduces the 92,254,181-byte raw file exactly. Run and resource SHAs are `627505212d26ca787e4979bcc25706c9ac68ff8fd4ed3fceb262a10611a2b653` and `7585c4dbc8a2fb9bfce1af96d691fb00f92ccefb17de8f4b412e35b8a4027f4a`, matching the archive manifest and parent receipt. Execution closed in 92.535 seconds wall time (89.509 seconds child elapsed), with 24/24 row reverse products, 4/4 HVPs, one `12×12` Cholesky solve, zero PCG/full-control Hessian, and sampled peak RSS `984,743,936` bytes under the 660-second / 1-GiB sampled limits. The sampled RSS was close to the limit and is not an OS hard cap.

Raw claims mark full smooth root, minimum, and response false. This one-point comparison does not support a forecast-skill, response, adjoint, reanalysis, or curvature claim. `S` is a positive search metric; the robust GN model omits residual Hessian terms and is not an exact constrained Hessian.

## Final findings/KG cross-check

The Korean findings and KG record agree with the saved raw candidates and archive: baseline accepted candidate `F²=0.004586433310`, coupled-GN selected endpoint `F²=0.004490021942`, predicted-vs-actual scalar F² decreases `2.692548%` vs `2.517379%`, six larger GN trials rejected only by actual F² Armijo, and the eight total candidate evaluations, resource limits, selected control hash, and one-commit accounting all match. I independently recomputed the reported full-vector relative errors: baseline `0.00447826083` and GN `0.15350389629`, each normalized by the actual endpoint residual norm. The reported percentages (`0.447826%`, `15.350390%`) and denominator description are correct and distinct from the scalar F² decrease error.

The docs preserve that this is a single base/candidate-grid comparison, the GN model omits residual-curvature terms, and there is no root/minimum/response/forecast claim. The task checklist final stage remains open while the GREEN collector/main result receipt is completed after a stream disconnect. This RED cross-audit does not close that remaining stage; no issue in the checklist state changes the saved numerical result.

## Final receipt closure

The saved-array collector/CLI and GNCOUPLED_RESULT are now complete. The fresh GREEN final review reproduced the result byte-for-byte and verified archive/selected closure; the formerly pending derived-result stage is resolved. PR integration is recorded separately.
