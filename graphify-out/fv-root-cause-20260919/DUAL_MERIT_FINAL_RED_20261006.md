# A35 dual-merit final RED audit — 2026-10-06

**Disposition: GO for the recorded bounded step.** This review independently checked the frozen plan, saved child/parent/resource records, and the trial arithmetic. I performed no FV, HVP, or test execution and changed no source, graph, or raw result data.

## Identity and execution record

The execution used frozen plan SHA-256 `62e291b8b683ef1480f60ae302e562c7313c2a545a3cb6517fd53f3b294220c0`. The plan’s 94 source and 8 archive pins all exist and match current bytes. Its control and parameter identities are the original a35 base (`a35e2f6ef000cca4b593a3e5ed49f1c3e94b951450f2f235902a50e1dde6647c`) and fixed parameters (`8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed`). The cached direction receipt is the pinned `a35_live_hvp_newton_20261006_attempt1/step.json`; the run reports the same base identity and `new_hvp=0`, `pcg_iterations=0`.

The saved result SHA-256 is `ace5c514f661c6257a635c59282c5fc7fdb31c6128e666333bdaebb6d5a8ebe1`; it matches `step.run.json`’s child digest. Child and parent status agree: `phase=finished`, `execution_status=completed`, and `numerical_status=one_dual_merit_step_accepted`, with no child-read error. The run’s 98-entry source snapshot is unchanged, `source_unchanged=true`, and `fixed_input_unchanged=true`. The input identity records the intended control update from a35 to the accepted endpoint while retaining the fixed observations, problem, parameters, and terminal truth.

The one guarded child exited 0 in `24.343269` seconds under the 300-second wall limit. It recorded 93 RSS samples and a peak of `353,665,024` bytes under the 1-GiB sampled limit, with no monitor error, resource termination, or SIGTERM. The separate run/resource receipts agree. This is sampled RSS evidence, not a hard OS memory limit.

## Candidate and policy audit

The cached direction has `gᵀs=-0.04171826501708994` and `gᵀHs=-2.225947234104554`. For `Phi=0.5||g||²`, these are the correct base directional slopes for J and Phi. The plan applies both Armijo constants at `1e-4`, radius `0.05`, and at most 16 dyadic trials.

The first trial used `alpha=0.17287763958236457` and has the historical 389 control SHA `38901e3a4c2ee97c8083b43e01e335f6328c3410982118b4ba891023468c471d`. It passed J Armijo (`J=0.06681574666444642`, threshold `0.07161426148364408`) and its strict endpoint branch, but failed Phi Armijo (`Phi=1.8573840876146208`, threshold `1.1129351354415058`); its saved rejection reason is the Phi inequality.

The second trial used half alpha, `0.08643881979118229`, with displacement norm `0.025`. It passed both gates: `J=0.06835683125676147` against threshold `0.07161462209140326`, and `Phi=0.8756777444539787` against threshold `1.1129543762666891`. I independently recomputed both thresholds from the saved base values, slopes, alpha, and policy constants. The full candidate gradient is finite with 26 components, and its squared norm divided by two reproduces the saved Phi to floating-point precision. Both endpoint branch records report 3,600 Euler, choice, and face-sign stages with strict margins. The accepted control SHA and vector match the second trial, and its full signature matches the saved accepted partition.

Relative to the same a35 base, accepted J decreases by `0.0032581514424009778` (4.55%) and Phi decreases by `0.23729587263789376` (21.32%). The accepted candidate has higher J than the historical 389 endpoint (`0.0683568` versus `0.0668157`); this audit makes no claim that the new result has lower cost than 389. It records two separate policies from the same a35 base.

## Supported limits

This record supports one accepted, same-direction dual-merit step at the pinned a35 point. The branch signature changes from the base; the strict check certifies the endpoints only, not a smooth path between them. The result does not establish stationarity, a root, accepted-point curvature, global SPD, adjoint/reanalysis, response, forecast score, or physical/weather validity. Those fields remain false or uncomputed in the saved result, and the accepted step’s gradient-linearization relative error is `0.34466`.
