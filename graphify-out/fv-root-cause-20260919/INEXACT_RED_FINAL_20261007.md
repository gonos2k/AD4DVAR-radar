# d31 strict/inexact paired run — RED final review

Date: 2026-10-07
Scope: post-run review of the saved d31 pair receipts and arithmetic summary. I did not launch FV/HVP/PCG or rerun tests.

## Disposition

Both cold-start arms completed one accepted correction from the same d31 endpoint under the strict-first shared budget. The requested inexact endpoint was selected and passed the same actual J/Phi Armijo, pointwise branch/margin, source, and input-closure gates as the strict endpoint. No receipt, counter, hash, or classification mismatch was found in the inspected run.

This pair supports a real **per-direction HVP reduction** for this one start: 31→22 HVPs and 29→20 PCG iterations. The saved arm wall times were 389.556 s and 304.147 s, but their order was fixed (strict first, inexact second) and the second arm may benefit from warmed caches. Treat the time difference as an observed single-run result; it is not a general speedup estimate.

## Recorded comparison

Both arms start from control hash `d31ecd393002a1e098c939838ce22e9a7526c1b7f0f426effbb3f3a408ccdeeb`, with base \(J=0.06211898776588735\), \(\Phi=0.3252767184999386\), and \(\|g\|_\infty=0.4267256181\). Each accepted one step of norm 0.00625 after four candidate evaluations.

| Measure | Strict | Inexact |
|---|---:|---:|
| Relative true-residual target | \(10^{-10}\) | \(10^{-3}\) |
| Fresh relative true residual | \(8.10355\times10^{-11}\) | \(3.77048\times10^{-4}\) |
| PCG iterations | 29 | 20 |
| HVPs, including independent true-residual product | 31 | 22 |
| Arm elapsed time | 389.556 s | 304.147 s |
| Accepted \(\alpha\) | 0.0667798502 | 0.0667789032 |
| Final \(J\) | 0.06177956399461902 | 0.06177954756643603 |
| Final \(\Phi\) | 0.2837064060538132 | 0.28369125567712217 |
| Final \(\|g\|_\infty\) | 0.4166193341 | 0.4166189970 |
| Accepted control hash | `e9865799…` | `f462a496…` |

The forcing value was \(\eta=10^{-3}\), and the independently recomputed inexact residual is below it. Both saved directions have negative `g·s` and `g·Hs`; both accepted trials pass both Armijo tests and the complete 3,600-stage endpoint branch/margin gate. Relative to d31, J fell by about 0.5464% in either arm and Phi fell by 12.78% in either arm. The two resulting controls differ, so the strict point is a comparison endpoint, not an intermediate step on the selected trajectory.

## Resource and record audit

The parent and resource receipts agree with the comparison JSON: raw child SHA `553f26dd…`, normal execution exit 0, `paired_comparison_completed`, 53 total HVPs (31+22), no HVP outside completed arm deltas, and one selected inexact step. Guarded elapsed time was 695.530 s under the 780 s outer limit. Sampled peak RSS was 373,604,352 bytes under the 1 GiB sampled threshold; this is periodic process sampling, not an OS hard memory limit. The top-level record selects the inexact control hash, and both arms retain the shared d31 base hash.

The frozen plan hash is `d69ce300…`. The stored run reports unchanged source and fixed-input closure, and the independent saved-array result record agrees on the arm values, residuals, counts, selection, and parent hash. The result record explicitly limits itself to saved-array/receipt arithmetic. It does not claim an independent FV or PCG rerun.

## Remaining numerical and statistical limits

- **Branch path:** each candidate endpoint passed its complete strict pointwise gate, but the accepted candidate's branch signature changed from the base in both arms. The run does not certify one smooth branch along the intervening segment. Actual endpoint J and Phi acceptance remains the basis for committing the step.
- **Timing:** strict always ran first and inexact second under the same remaining budget. The 85.408 s lower inexact wall time is order- and cache-confounded. The 29.0% HVP reduction is the cleaner discrete work comparison, but it is still one pair at one control point and does not establish expected savings across later iterations.
- **Convergence:** both arm statuses are `iteration_limit` because each was capped at one accepted iteration. The selected endpoint has \(\|g\|_\infty=0.416619\), about \(4.17\times10^9\) times the unchanged \(10^{-10}\) root threshold. No eligible root or final curvature was obtained.
- **Scientific validation:** no adjoint, reanalysis, independent forecast score, or forecast-skill result was computed. Improvements in J/Phi are objective-space changes only.

## RED conclusion

The bounded experiment passed its declared numerical acceptance and integrity gates. It provides evidence that the inexact forcing rule reduced HVP/PCG work while preserving nearly the same one-step J/Phi progress at d31. It does not show a general runtime speedup, robust convergence advantage, smooth branch path, stationary point, or forecast improvement. Keep the strict `1e-10` final-root criterion and require later accepted points to rebuild current-point HVPs under the same true-residual and dual-merit gates.
