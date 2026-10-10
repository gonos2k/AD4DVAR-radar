# RED final review: event-direction resume at 48b0 — 2026-10-10

## Disposition

The saved run closes one new same-point GN build, event requalification, two direction arms, a one-arm fallback, one selected P2 repeat, and one commit. The event-orthogonal arm was unsupported by its fresh merit-descent gate, so the result is not a direction comparison or a win for either arm. The prepared-GN fallback was supported and accepted. No future-point readiness or second candidate was run.

## Frozen plan and operator receipts

Plan SHA: `2ba1d88c83e6abdc054647af4d5d5a074a4b2569eee3e8ffc7eafcbe5e20591c`. All 158 source and 213 archive pins exist and match. At base control `48b0f95c5aa485a6a17046d4497a2267a389b1cfb2eeedb75d79e7ba038d5729`, the run freshly observed the full mixed residual/J/θ, generated 24 current-point residual-row VJPs and one Cholesky-positive dense solve, and computed two current event gradients plus two JVPs. The event values matched the original objective and replay-disabled trajectory exactly, and each gradient dot direction matched its JVP.

The two HVPs for each arm were fresh and bound to the 48b0 control, arm direction hash, selected-face operator, and sides −1/+1. Counts closed at 24/24 rows, 1/1 solve, 2/2 event gradients, 2/2 event JVPs, and 4/4 HVPs.

At 48b0, event stage 124, x, interior cell `(1,2)` remained on the supported same-negative/right-selected side on both extensions: left `−0.3311500547172628`, right `−0.33114990825412605`, ζ=`−1.4646313672983524e-7`, both signs −1, and `choose_left=false`. Thus the current event was requalified before the correction arm was formed.

The prepared GN direction passed its model gates and first passed both original actual-J and actual-R Armijo checks at candidate 5/24: α=`0.04574449898190079`, actual movement `0.00211619`, J=`0.06109417561387711`, and R=`0.0042980213317858635`. The event-orthogonal direction’s fresh HVPs completed, but its model failed `merit_descends`; it had no candidates. Selection therefore records `comparison_complete=false`, `selection_basis=single_supported_arm_fallback`, and `supported_arms=[prepared_gn]`.

## P2 and current-point state

The prepared GN candidate received the only independent P2 repeat. It matched J, R, θ, side gradients, and branch trace, and passed the paired-branch, face, source, input, runtime, and deadline checks. The committed control is `15516fbae36637547a151066dc4af5eaa0ce74af9dbc5e93fe24694ff96c4cc0`, with θ=`0.6882137924515473`. Relative to 48b0, J decreased by `8.9941e-5` (`0.1470%`) and R decreased by `7.5198e-5` (`1.7195%`).

The full mixed-residual L2 norm decreased from `0.0661303` to `0.0655593`, while its L∞ norm increased from `0.0285886` to `0.0322529` (about `12.82%`). This is a local merit/cost decrease, not a stationarity or minimum claim.

At the committed point, the same trace-124 event changed to ζ=`+0.01291894449740738`; both increments remained negative, but `choose_left` became true on both sides. That endpoint is evidence for requalifying event support at any later point, not for carrying the old event orientation forward or adding a permanent selector constraint. The run applied one optimizer commit, reports `readiness_complete=false`, performed no postcommit readiness, and evaluated no further candidate.

## Receipts

The child exited 0 in `79.5737 s`, with no resource termination or monitor error and sampled peak RSS `612,417,536` bytes under the 2-GiB sampled limit. Raw SHA is `7d568ad20318e0c86bb5814e8d51a444ca4bf9bbfe420994a8f14da8f5ffabf5`; gzip SHA is `53e90977b628a2e9e5f785779899df688bcbc46a92a498a8dd1c1c9f57b1c2b0`; decompression is byte-identical to the raw record. Run and resource receipt hashes are `e448b854d174f8342c87f6e29d7f1c6a4dbe14c183a962ab86fa2d297b9616aa` and `a132a3a292ee142d5ac3efa87b64ac15f7565faedff08a19b4add2174b7c1746`.

- Producer and plan: `examples/weather_scenarios/fv_point_3h_event_direction_resume.py`, `EVENT_DIRECTION_RESUME_PLAN_20261010.json`
- Raw, compressed record, run receipt, and resource receipt: `event_direction_resume_20261010_attempt1/step.json`, `step.json.gz`, `step.run.json`, and `step.resource.json`

Root, minimum, score, and response claims are false.
