# RED final review: released GN resume at 15516 — 2026-10-10

## Disposition

The frozen plan and saved run close one fresh current-point GN build, one bounded 24-candidate search, one independent P2 repeat, and one commit. The source pins and run-state receipts agree. The run uses no event gradient/JVP or event projection, and it does not carry the older negative/right-selector requirement forward as a constraint.

## Plan, source, and closed parent

Plan SHA: `8180ad76168653ec87de1865d87fab48f207a1fab76448abc9f7d655b66c1ded`. All 160 source and 218 archive pins exist and match. The loader closes the prior `15516fbae36637547a151066dc4af5eaa0ce74af9dbc5e93fe24694ff96c4cc0` point through its pinned archive, run/resource receipts, source scope, and P2 closure.

At 15516 the runner freshly checks the fixed input/runtime and full mixed residual/J/θ/paired traces, then builds 24 current-point residual rows and one dense solve. It computes two fresh HVPs bound to the new GN direction, base control, theta, and selected-face sides. Counts close at 24/24 rows, 1/1 solve, and 2/2 HVPs. The model residual and theta match the freshly measured full G and θ.

The stage-124 x-cell snapshot records left and right increments both negative, with the left selector active on both sides. The old negative/right-selector condition is explicitly marked false and retained as a diagnostic snapshot only. No event gradient, event JVP, or event projection was performed.

## Search, commit, and limits

The fresh prepared-GN direction first passed the original actual-J and full-R Armijo checks at candidate 6/24: α=`0.026139762146698584`, actual displacement `0.001159532966076956`, J=`0.06104894133302345`, and R=`0.0032814674811390797`. The six evaluated candidates stayed within the 24-slot plan cap and actual radius `0.05`.

The selected endpoint received the only independent P2 repeat. It matched objective, full R, θ, side gradients, and branch trace, and passed paired-branch, face, source, fixed-input, runtime, and deadline closure. The committed control is `29b52a1377caa7fb52431bdbea25124fa827ac548eeefb9b39ce0e5f240d97c4`, with θ=`0.6865600659804322`. Relative to 15516, J decreased by `4.5234e-5` (`0.0740%`) and R decreased by `0.00101655` (`23.65%`). The full mixed-residual L2 and L∞ norms both decreased, from `0.0655593`/`0.0322529` to `0.0572841`/`0.0268769`.

At the committed endpoint, both negative increments have `choose_left=false`; the selector returned to the right side. This is endpoint evidence only. It does not impose the previous left-selector state as a constraint or establish the selector path between endpoints.

The run applied one optimizer step and performed no postcommit readiness or further candidate. It supports a local cost/merit decrease only; no convergence, minimum, response, or forecast claim follows. Any next point needs fresh current-point operators and a newly observed event snapshot.

## Run and verification receipts

The child exited 0 in `57.8854 s`, with no monitor/resource termination and sampled peak RSS `428,621,824` bytes under the 2-GiB sampled limit. The child SHA `fc1f7f8cbad2f52b6b5b1dc8eed0b1a42512e42c94020cc0d4c2affabcfd7b6e` matches the run receipt. The archive manifest confirms lossless gzip round-trip (raw SHA `fc1f7f8cbad2f52b6b5b1dc8eed0b1a42512e42c94020cc0d4c2affabcfd7b6e`, gzip SHA `f8ce74836d7a59f7da5970499b03d1d6f4517176ddf7c224b1c00ad5c16e944f`). Source, fixed-input, runtime, and deadline flags are true. The prelaunch suite reports 13 tests passed and zero type errors/warnings; it is separate from the actual guarded run.

- Producer and frozen plan: `examples/weather_scenarios/fv_point_3h_gn_released_resume.py`, `GN_RELEASED_RESUME_PLAN_20261010.json`
- Run record, compressed record, run receipt, resource receipt, and archive manifest: `gn_released_resume_20261010_attempt1/step.json`, `step.json.gz`, `step.run.json`, `step.resource.json`, and `GN_RELEASED_ARCHIVE_20261010.json`
- Prelaunch verification: `GN_RELEASED_VERIFICATION_20261010.json`

Root, minimum, score, and response claims remain false.
