# Bounded three-hour point merit continuation: two steps, then policy refusal

The single predeclared continuation from the PR #221 accepted endpoint
completed **two strict, branch-changing J/Phi descent steps** and
then refused all eight candidate sizes in its third epoch. The child
and parent both finished normally with numerical status
`epoch_line_search_refused`. The last accepted endpoint is a
**nonstationary search candidate**, not a root, smooth connecting
path, minimum, local response or physical forecast result.

The starting 26-control tensor and 13 parameters match the PR #221
source-bound endpoint, with control SHA256
`9f8d3ae0d77d8b24f1565ab622cf8fe7bbfd83766fbd67964a6a5bd52920a078`.
The problem, background, verification target, 0/10/20-minute
observations, 18 future ten-minute leads and boundary schedule were
unchanged. At every visited current point the child ran a separate
complete margin pass and the original 3,600-stage strict branch
checker, then freshly computed J, g, Phi, `Hg` and `H^Tg`. The three
one-vector transpose relative differences were approximately
`7.85e-16`, `1.04e-15` and `9.41e-16`; these confirm the particular
directional products, not a full Hessian spectrum or path-wide
smoothness.

| Current/accepted point | J | Phi=`||g||²/2` | `||g||_inf` |
|---|---:|---:|---:|
| PR #221 starting endpoint | `0.9319504904669574` | `2.6187151084118883` | `1.2605349625093007` |
| First accepted endpoint | `0.9314780544833493` | `1.0156964286855181` | `0.8564443672589114` |
| Second and final accepted endpoint | `0.9314590521034872` | `0.9755199172601001` | `0.8727746789931015` |

The first step accepted `alpha=0.000625` after five larger strict
endpoints increased both J and Phi. The second accepted
`alpha=0.00015625` after seven larger strict endpoints failed the
two-scalar decrease gate. Both accepted endpoint signatures differed
from their respective current signatures. On a changed signature,
only **freshly measured** J and Phi decreases were used; the old
branch's Armijo slope was not treated as a crossing-path proof.
Although J and Phi decreased at both accepted steps, the maximum
gradient increased slightly on the second (`0.85644→0.87277`).
Overall from PR #221, J fell by `0.0004914383634702091` and Phi by
`1.6431951911517881`; the final gradient maximum remains far above
the final `<1e-10` gate.

At the third current point, all eight fixed-alpha endpoints again
passed strict branch checks on changed signatures. The seven larger
steps increased both J and Phi. The smallest `alpha=0.00015625`
reduced J slightly but **increased Phi** from
`0.9755199172601001` to `0.9978258546287937`, so it too was
refused. No smaller step, fallback direction or retry was introduced.
The last accepted control SHA256 is
`baed75b17f6552cecf999be2a23db75725fef8d713a1dd360326ee82680159ba`;
its own strict branch signature is
`2925e02c77e293436b8eccfeda1db44390bb4ca6f106f5a5b581818d9ab57dc5`.
Its 3,600-stage current-point margin record is complete. The minimum
active x-limiter gap normalized by the field scale was
`1.6824459398602114e-7`; the minimum absolute face flux was
`0.0007679630698590456` at `q_y[3,1]`. These remain pointwise
diagnostics, not a finite perturbation certificate.

The run made **22 endpoint trials**, three fresh HVPs and three
fresh parameter-free control VJPs, with 25 full strict branch-oracle
calls including current points. It accepted two of three attempted
epochs. One guarded child exited 0 with no resource termination or
monitor error. The guard measured **122.320 seconds** elapsed and
457 sampled child-RSS readings with peak **381,059,072 bytes** under
the 600-second wall trigger and sampled 1 GiB limit; child internal
elapsed was **121.033 seconds**. Measured phase times sum to
120.983 seconds, leaving about 0.050 seconds of unassigned overhead.
The parent did no FV work and validated source, PR #221 raw/aggregate
evidence, archived input, fixed parameters, plan, epoch chain and
resource record before accepting the completed refusal.

The affected suite passed **108 tests** with 18 existing TorchScript
deprecation warnings; nine focused tests passed. Error-level
basedpyright 1.39.9 reported 0 errors/warnings/notes. GREEN/RED
reviewed the plan and code, including two-epoch relinearization,
incomplete budget records, branch-call accounting and forged
handoff/control reports. Exact file hashes and commands are in
`FV_POINT_3H_MERIT_CONTINUATION_EVIDENCE.json`.

This closes only the **declared 8-step/64-trial continuation attempt**
with two measured improvements and a subsequent line-search refusal.
R5-R-R remains open. The current endpoint's own Hessian has not been
fully audited for a final root/adjoint; no step after its refusal,
stationary result, 13-parameter VJP, signed reanalysis or physical
skill is claimed. Any new step-size range, direction or globalization
policy needs a separate plan and must preserve the final stationarity,
curvature, original-H adjoint-residual and reanalysis gates.
