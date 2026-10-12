# RED final audit — same-point gradient versus robust GN (2026-10-12)

## Result

The comparison completed both arms under the fixed `minmod_net_flux_fp64_v1` evaluator at the admitted `b6470069…ef88660` point. Both arms used fresh HVP pairs, independent actual J/R candidate searches, and independent repeated P2/full-G approval. The selected arm was the baseline tangent-gradient direction because its repeated R was lower. Robust GN had lower repeated J but higher repeated R. This is one bounded comparison and one conditional commit; it does not show a general winner, converge the optimization, or establish a minimum.

## Independent receipt checks

- Raw SHA `1c79690d…8b4703` matches the child SHA in `step.run.json`. The run records 131.7895 s internal / 133.6396 s external, exit 0, no resource termination, monitor error or SIGTERM, and sampled RSS 1,119,698,944 bytes under the 2 GiB limit.
- The frozen comparison plan and source bundle remain `36686ccf…904b35` and `556fec30…0c46bc`. Earlier RED verification found all 209 sources and 283 archives pinned with zero inherited source drift. Raw source/input/runtime and deadline closures pass.
- At base `b647…ef88660`, current fresh previous-P2 and minimum-mixture-P2 matches pass with zero recorded G, objective and theta error. Native release is not required. Base J/R are `0.061003289930848054 / 0.0025752047662763056`.
- The baseline arm completed 2/2 fresh HVPs and one actual candidate. Its repeated candidate is `J=0.061002341038211755`, `R=0.002559866170789153`; both Armijo bounds pass, repeat P2/trace/source/input/runtime/deadline closure passes, and full-G component/merit checks pass.
- Robust GN completed 24/24 row VJPs (12 on each side), one 12x12 Cholesky solve, and its own fresh 2/2 HVP pair. It evaluated 10 candidates: all 10 pass J Armijo; the first nine fail R Armijo and the tenth passes. Its independently repeated eligible candidate is `J=0.06100086445604125`, `R=0.002567751483553327`, with repeat P2/trace/source/input/runtime/deadline and full-G closure passing.
- Repeated GN R exceeds repeated baseline R by `7.885312764174379e-6`; GN J is lower by `1.4765821705065019e-6`. The selector correctly chooses baseline by its declared actual repeated-R priority, then J tie rule. The completed one-step endpoint is `d19e882d…c4145e`; raw counts are 4/4 HVPs, 24 GN row VJPs, one solve, 11 candidates total, and one commit. Future-E recomputation and postcommit readiness are false.
- HVP-stage receipts are explicit, and both arm HVP vectors match their recorded direction hashes. Candidate receipt hashes are checked before their repeats. No previous-point HVP is reused.

## Interpretation limits

The reported `chi` is local model directional efficiency, not realized R reduction. It is `0.00485297` for the baseline and `0.87468153` for GN; despite GN's higher modeled efficiency and lower J, its repeated R is worse, and its search backtracks to `alpha=0.00151845` from the model's unbounded alpha `0.77744836` because R Armijo fails at the first nine tested alphas.

The candidate model residual errors, measured against actual candidate `ΔG`, are 41.7921% for baseline and 100.0734% for GN. Both accepted candidate signatures differ from the base; every evaluated candidate records a changed base signature, while each candidate's own two-sided pair and independent repeat traces close. This comparison does not isolate those errors to selector changes: finite-step nonlinear terms also contribute. These are model-prediction discrepancies, not independent HVP/true-gradient error measurements. The exact candidate J/R/P2 repeat—not the local prediction—is the approval evidence.

No causal statement that GN is inferior in general follows from one base and one candidate per arm. The 9 failed GN R checks and selection of the gradient direction are findings for this fixed point, radius, dyadic grid and budget. The result does not establish a normality root, convergence, a minimum, exact curvature/adjoint, response, new-version future score, or forecast skill. The separate score observer is observation-only and was not run as part of this comparison.

No source, plan, input, test or archive was edited for this RED audit. Checks used saved JSON, source pins and arithmetic only; no FV, HVP, optimizer, score or tests were run by RED.

## Packaging/prose RED check

- `receipts.tar.gz` matches `RECEIPT_MANIFEST.json` SHA-256 `55f220c1…7a4040`; all 16 tar members have the manifest's exact paths, byte lengths and content hashes. The bundle preserves the 11 candidate receipts, GN operator record, step/run/resource receipts and empty log without rewriting the raw run.
- `FINDINGS.md` and `METRICS.json` agree with the audited raw: both arms are independently eligible; baseline is selected on repeated R, despite GN's lower J and much higher modeled chi; GN has nine R-only candidate refusals; both accepted candidates changed base signature; final readiness/E remain absent. Model-error ratios are labeled as finite vector-prediction mismatch, not HVP accuracy, and no branch-only causal attribution is made.
- Verification is explicitly same-point/one-commit, not an eligible minimum or full-suite result. No test-count or remote-regression overclaim found in the package.
