# Smaller-alpha tail from the refused three-hour point epoch

One predeclared tail run from PR #223's last accepted control completed
**two branch-changing exploratory steps**. Both used the first new
alpha `0.000078125`, strictly below the prior run's smallest refused
`0.00015625`. Each endpoint passed the original 3,600-stage strict
branch and reduced both the unchanged analysis objective J and
stationarity merit Phi=`||grad_c J||²/2`. This is measured candidate
progress, **not** a stationary analysis, smooth connecting segment,
adjoint response or physical forecast result.

The starting control SHA256 was
`baed75b17f6552cecf999be2a23db75725fef8d713a1dd360326ee82680159ba`,
with fixed PR #204 4×5 point observations, 13 parameters, external
background, target, 0/10/20-minute observation times, 18 future
ten-minute leads and boundary schedule. The PR #223 raw/evidence
hashes, tail plan and current refactored source were checked before
and after the run. PR #223's historical execution used Git commit
`32ee38d72084735555e11ab4fbbad34afc321f45` (source SHA256
`00bed148ffc7549eca3cd8446ac1fd1d5c4042f1278867808aafe78f51e848dc`);
this tail run used a **different** source hash for the immutable-policy
refactor. The old numerical records were not overwritten.

| Point | J | Phi | `||grad_c J||_inf` |
|---|---:|---:|---:|
| PR #223 start | `0.9314590521034872` | `0.9755199172601001` | `0.8727746789931015` |
| First accepted | `0.9314473935070856` | `0.9533070968486816` | `0.865280110188193` |
| Second/final accepted | `0.9314353874766477` | `0.9522514198858343` | `0.8719793800953134` |

The net J reduction is `2.3664626839559055e-5`, and the net Phi
reduction is `0.02326849737426584`. The gradient maximum decreased
only `0.000795298897788066` overall and **rose on the second step**.
Each accepted endpoint had a new strict branch signature; its old
branch's Armijo slope was not used as a crossing-path certificate.
Both current points were relinearized with fresh exact Hg and H^Tg,
and the final accepted endpoint received a separate full strict
branch/margin recheck. The final control SHA256 is
`602b08828b92508dfaed3605f4626882a1cf70646730da1cb866416dd15cf0b9`;
its strict signature SHA256 is
`18a36b01bc9a8b7c21ea1970f79f8bd379a52ceb67b7fae72b2e79c1c5d0eed8`.
The result status `accepted_epoch_limit` means the **planned two-step
limit** was reached. Its gradient remains far above the `1e-4`
handoff and `<1e-10` final stationarity gates.

The guarded child exited 0 with no wall/RSS termination or monitor
error. The guard measured **45.922 seconds** elapsed and 172 sampled
child-RSS readings with peak **379,846,656 bytes** under the
300-second wall trigger and sampled 1 GiB limit; child internal time
was **44.656 seconds**. Phase times sum to about 44.651 seconds,
leaving about 0.005 seconds unallocated. There were two endpoint
trials, two fresh HVPs, two fresh control VJPs and five strict
branch-oracle calls, including the final endpoint check. The parent
performed no FV reconstruction and validated the profile, source,
archived input, plan, seed records, resource record and child status.

The final affected suite passed **110 tests** with 18 existing
TorchScript deprecation warnings; 11 focused policy/continuation
tests passed. Error-level basedpyright 1.39.9 reported 0
errors/warnings/notes. GREEN/RED reviewed the immutable policy
refactor, the old default command, the tail-specific guard and the
actual raw run. Graphify's incremental code refresh produced a valid
graph with 8,240 nodes and 212,454 links. Exact artifact hashes are
in `FV_POINT_3H_MERIT_TAIL_EVIDENCE.json`.

This closes only the **declared smaller-alpha two-step tail attempt**.
R5-R-R remains open. The current endpoint still needs an appropriate
search or separate terminal stationary handoff; no exact Hessian
suitability, original-H adjoint residual, full 13-parameter VJP,
signed nonlinear reanalysis, finite observation impact or physical
skill was established here. The tail's success does not imply that
additional steps or a larger finite perturbation will succeed.
