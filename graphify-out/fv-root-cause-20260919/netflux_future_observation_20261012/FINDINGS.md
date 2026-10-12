# Paired new-version terminal score observation — 2026-10-12

Exactly two production terminal scores were computed with the identical minmod_net_flux_fp64_v1 transport, SHA-verified old fixed-case observations/prior/parameters/boundaries/timing/truth, and admitted control receipts. The point pair is the start/end of PR299's 11-step gradient trial: 0aafdf80… to b6470069… . This is separate from the new direction comparison, whose selected control is d19e882d…; the new d19e point was not scored here.

| Point | New-version terminal E (dBZ squared MSE) |
|---|---:|
| 0aaf start | 7.388754298552231 |
| b647 final of 11-step trial | 7.386491165784028 |

Final minus start is -0.0022631327682037 dBZ squared, a **0.03062942% decrease**. Both values are recomputed under the same new kernel and same old synthetic verification field. A cached old-version E is not the baseline. The score is the last +180-minute frame after the 20-minute analysis window (absolute12000s), not an18-lead average.

The source/plan/asset/runtime and both point input identities close. Exactly2 calls start/complete; optimizer/candidate/selection counts are0. The plan 7c2b7f317e90e49a0c1bd9a289529816b8cef91e9d33048c8017b9b884279ae5 freezes209sources283archives, max2scores,600internal660outer2GiB. The guard exits0 in 12.638017s with sampled peak 944701440bytes (0.879822GiB), no termination/monitor error/SIGTERM. The parent publisher checks full guard status and closed child receipt instead of only exitcode. RED found that gap before any score execution; the unlaunched v1 plan/bundle are preserved in preflight_v1.tar.gz.

This metric is observational and is never read by the candidate selector. The fixed synthetic truth comes from the original matched-model setup; the score pair is not independent forecast skill, exact response, actual signed reoptimization or minimum qualification. Its small observed decrease must not be equated to the6.1323% stationarity-R improvement, or used to forecast convergence. The changed numerical evaluator is separated from control changes by computing both values anew.

GREEN/RED audit the saved pair, identity and guard without another forecast run. The final focused test set has43passes29.91s (includes both new experiments and unchanged GN/search checks; subset counts are not summed),18existing JIT deprecation warnings; six new files have0 basic-mode type errors/warnings. This is not a full repository or remote FV regression claim.
