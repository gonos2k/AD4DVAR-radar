# A35 dual-merit guarded result — final GREEN review, 2026-10-06

**Disposition: GO for the scoped one-step dual-merit result.** The single guarded run completed and accepted its second dyadic candidate under both actual Armijo conditions and the candidate's own strict endpoint branch gate. This review inspects saved records only; I did not rerun FV, HVP, or PCG work.

## Lineage and integrity

The dual plan SHA-256 is `62e291b8b683ef1480f60ae302e562c7313c2a545a3cb6517fd53f3b294220c0`. All 94 source and 8 archive pins match current bytes. The child result SHA-256 is `ace5c514f661c6257a635c59282c5fc7fdb31c6128e666333bdaebb6d5a8ebe1`; the parent receipt names that digest, reports `execution_status=completed`, exit 0, no child read error, and no monitor or resource termination.

The direction source is the separately saved a35 step with SHA-256 `64bef5e1a16af8fe4a480601c94d4357ce192fec0ab4c1e41ae510c8177b0c52`. Its own producer receipt also matches the raw step and records the original-J live Hessian action at a35. The dual run starts from that base control, SHA-256 `a35e2f6ef000cca4b593a3e5ed49f1c3e94b951450f2f235902a50e1dde6647c`, with unchanged parameter SHA-256 `8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed`. The historical accepted endpoint `38901e3a4c2ee97c8083b43e01e335f6328c3410982118b4ba891023468c471d` is not the start point; it reappears only as trial 1.

The before/after source maps have 98 entries and are identical. The fixed problem, terminal truth, archived observations, prior parameters, and runtime stay the same; the control hash changes to the accepted candidate as intended and passes the fixed-input closure check. The new search reuses the stored `s` and `Hs`; the result reports zero new HVPs and zero PCG iterations. It does not recompute curvature at the accepted point.

## Dual-merit decisions

At a35, `J=0.07161498269916244` and `Phi=0.5||g||²=1.1129736170918725`. The cached solve has true relative residual `8.2141051e-11`, `gᵀs=-0.04171826501708994`, and `gᵀHs=-2.225947234104554`. These are the first-order slopes for `J` and `Phi`, respectively, for the fixed smooth-branch direction.

The frozen policy uses `cJ=cPhi=1e-4`, radius `0.05`, and at most 16 dyadic candidates. Trial 1 used `alpha=0.17287763958236457` and reproduces the historical 389 endpoint exactly. Its actual `J=0.06681574666444642` passes the J bound `0.07161426148364408`; actual `Phi=1.8573840876146208` fails the Phi bound `1.1129351354415058`. Its strict endpoint check passed. The dual policy therefore rejected it only for Phi, while its historical J-only acceptance remains intact under the old rule.

Trial 2 halves alpha to `0.08643881979118229`, giving a displacement norm `0.025`. Actual `J=0.06835683125676147` passes its bound `0.07161462209140326`; actual `Phi=0.8756777444539787` passes its bound `1.1129543762666891`; and the candidate passes its complete strict endpoint branch/margin gate. It is accepted with control SHA-256 `e0b04a4dc019cea2664af811e7cd956453dd6b34c9d2f7a35bff171fdc57132a`. The accepted full-gradient infinity norm is `0.7881555762`, down from `0.7980151516` at a35.

The accepted step reduces `J` by `0.00325815144` and `Phi` by `0.237295873`. Its saved local J-model reduction ratio is `0.94433`; the gradient linearization relative error is `0.34466`. These are finite-step diagnostics, not a stationarity certificate. The branch signature changes from a35, and the record certifies the endpoint only, not the path between points.

## Runtime and limits

The one guarded child completed in `24.3433` seconds under the 300-second outer wall limit, with sampled peak RSS `353,665,024` bytes under the 1-GiB sampled limit. This is sampled process RSS, not an OS hard allocation cap. No HVP or PCG work occurred in the dual search.

The result records no global SPD claim, no accepted-point curvature computation, no full-root claim, and no response, score, forecast-skill, or physical-validation result. The supported conclusion is one bounded same-direction step accepted by both actual merit tests at a strict endpoint, starting from the original a35 point. It does not establish a stationary point or root, global curvature, a branch-path certificate, or weather impact.
