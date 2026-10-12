# PR302 follow-up: augmented QR and original J/R continuation

The active-radius P2 is repaired and the original objective/residual search made nine confirmed steps in one frozen plan. GN technique selection no longer gates this original-progress runner. Historical weighted completion remains about70%; these results do not certify convergence, a minimum, an adjoint or forecast improvement.

## Stable active-radius algebra

Active multiplier solves now use augmented QR of [V;sqrt(lambda)D], and triangular factors apply the inverse needed by the implicit multiplier derivative. No V^TV+lambda D^TD linear system is formed. The active movement norm is closed differentiably after a QR-condition-scaled root precision check. This closes numerical radius underfill and its false scale-G derivative without adding a ridge or relaxing candidate J/R approval tolerances.

Under local PyTorch2.13 the review's full-rank SPD family returned inaccurate radius/derivatives rather than throwing the PyTorch2.10 exception. Tests use represented binary64 inputs, an80-digit Decimal spectral reference, both unit-eigenvector and G=(-1e-8,-1e-8) inputs, padded26/27 shapes, near-machine radius closure and fixed-rank active/inactive derivative checks. The active-QR/rank suites pass27 tests; no exception reproduction is claimed for the local version.

## Cached current model and original progress

The exact PR302 source/receipt bundles,7839 control/context, two cached side-HVP vectors and DGd were admitted independently of the obsolete live ancestry loader. All228 frozen source hashes and296 archive pins were checked; only the search solver and continuation dispatcher changed. Forward transport, observations, prior, parameters, verification truth, timing and boundary sources remain pinned. The closed endpoint snapshot is explicitly hashed.

Gradient-ray model predictions at the original base radius1.8757177916958825e-5:

| Radius multiple | Predicted R |
|---|---:|
| 1 | 0.0025390637671354006 |
| 1.25 | 0.0025371026109095723 |
| 1.5 | 0.002535322133116011 |
| 2 | 0.0025323032128256864 |

The same-radius two-direction model adds zero GN gain throughout this range. This does not remove the positive decrease predicted for the gradient ray itself: the2x model predicts about0.644% R decrease. Only the actual original J/R and repeated qualification decide acceptance.

One frozen plan capped physical movement at the existing2x limit3.751435583391765e-5,19 commits,24 total FV candidate calls/6 slots per point,36 new HVP requests plus4 historical calls at most40, and2GiB. Previous preparation79.7018435s (four HVPs,24 row VJPs,one GN solve) was charged into the600s internal/660s outer time envelope. There are no new GN row or solve calls. First-point native/P2/control/direction and cachedDGd closures requalify the cached gradient HVP pair; every later point prepares a fresh pair. Source/gates, actual chart displacement, actual alpha, theta, native branch-pair/individual gradients, independent repeat and full-G validation remain enforced. Rolling comparison diagnostics are observational and do not start another technique comparison.

## Actual accepted prefix

Start7839625d… -> final **2a4dba130d28273a63e97d0de39a11f64fdee09b0ad754559bc8902ba6cddbca**:

| Metric | Start | Final |
|---|---:|---:|
| J | 0.061001393469310765 | 0.06099005811764152 |
| R=||G||² | 0.002548715176361377 | 0.002441269566693863 |
| ||G|| | 0.05048480143926 | 0.04940920528296 |
| max mixed gradient | 0.02539963944426 | 0.02487654677420 |

J decreased0.0185821192%; R decreased4.2156773995%. Nine candidates and nine independent repeats became nine confirmed points, each with strict originalJ/R Armijo, previous/minimum-mixture P2 and full-G closure. Cached HVP reuse was exactly two calls at point0. Sixteen new HVPs completed at changed points1–8; one additional request at point9 remained at precheck before any JVP. Thus completed historical+new HVP count is4+16=20, not21. E/score calls, root/minimum/response/forecast claims are zero/false. The final point is not scored.

## Timing and verification limits

The run stopped with budget_exhausted at the next HVP precheck and preserved the complete nine-point prefix. Exit0/no external guard kill. Guarded wall531.9347970s plus charged cache79.7018435s =611.6366405s, below660s. Child elapsed527.4582353s plus charge =607.1600788s, **7.1600788s above the nominal600s internal deadline**. The recorded deadline_passed flag is false. The uninstrumented difference may include identity/write/finalization costs; that cause is not established. No final JVP, candidate or commit began. Do not report a strict600s total pass. Sampled peak RSS497,696,768bytes (about0.464GiB); this is sampled RSS, not an OS hard allocation limit.

The protocol test run reports54 passed and2 failed in6.76s, with18 existing TorchScript deprecation warnings. Both failures are legacy d19e live-endpoint ancestry tests requiring the old immutable continuation source bytes; that loader correctly refuses the intentionally changed dispatcher before reaching their intended receipt checks. This is a recorded compatibility limit of the historical live-loader path, not a clean full suite. New pinned-snapshot/cache, cache-only-at-point0/fresh-later, actual-alpha receipt and candidate-forwarding tests pass. Basic Pyright on five changed/new core/script files reports0 errors/warnings. No full repository or remote CPU/package baseline was repeated.

GREEN/RED independently validated source/archive seals, each accepted receipt/control chain, actual repeated J/R/G/theta, no phantom JVP and resource outcome. `VERIFICATION.json` passes every stored-prefix audit check. These audits do not independently regenerate the production FV/HVP computations. The exact source and raw progress archives are supplied; restore loose progress1 files from receipts.tar.gz without rerunning them.

Remaining: residual norm0.0494092 is far above the unchanged1e-8 candidate threshold. Exact stationarity/curvature/native active-structure qualification, original assimilation adjoint/VJP, p±h reoptimization and independent future validation remain incomplete. A future continuation must freshly prepare the final2a4d… point: its next HVP request never passed precheck; products from earlier accepted bases are not operators at this final point.

Restoration: extract `closed_endpoint.tar.gz` here for the exact pinned closed-endpoint dependency, then `receipts.tar.gz` here for `progress1`. The snapshot is compressed for review size; its original SHA-256 and bytes are unchanged.
