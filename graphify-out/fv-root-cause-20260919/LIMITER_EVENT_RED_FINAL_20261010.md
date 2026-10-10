# RED final review: limiter event at the PR279 endpoint — 2026-10-10

## Disposition

Attempt 2 closes as a no-commit event diagnostic. Its frozen plan, saved raw record, run receipt, resource receipt, and gzip archive agree. The raw arrays are finite, and I found no derivative or state-count inconsistency. The evidence supports one local same-sign x-limiter event and endpoint samples around its selector change; it does not identify a certified path crossing or justify a direction correction.

## Plan and execution state

Attempt 2 plan SHA: `3579c2bbf9f3213cec5669948e9de21e47da258720572b291b083c5355e3f79d`. All 154 source and 200 archive pins exist and match. The plan freezes event stage 124, x orientation, interior choice cell `(1, 2)`, 360 stages, three chart samples, two event gradients, two JVPs, zero HVPs/row VJPs/solves, and zero optimizer steps.

The prior base is control `027e83cf77f7f9cf97c0c72da9f737b5627d312d6375b698bb5b7f7ae94c6769`, with its archived GN direction reused. No archived HVP vector was used. Attempt 2 completed with 2/2 event gradients and 2/2 JVPs, three samples, and zero optimizer commits. Source, fixed input, runtime, and deadline checks passed.

Attempt 1 is a failed launch, not diagnostic evidence: its saved log ends at source/input/runtime closure after the source files changed under its earlier plan. Attempt 2 uses a new plan and matching source pins.

## Event values and derivatives

The event maps to `q[2, 3]`: at stage 124, the x-direction left and right increments are `q[2,3]-q[2,2]` and `q[2,4]-q[2,3]`. The base values are identical on both selected-face sides: left `-0.3318221107073498`, right `-0.3318219531182294`, and `max(abs(q))=39.026903783277`. Thus raw `ζ=left-right=-1.5758912041974327e-7` with both increments negative and the right increment selected.

Both sides give `dζ[d]=0.2821400467008237`; the event-gradient dot direction agrees within about `1e-15`, and the projected-gradient check also passes. The direction is tangent to the face (`unit_normal·d=8.67e-19`). The first-order estimate is `α≈5.585492816865093e-7`. The ambient side gradients differ (maximum component difference `0.0131`), while their face-tangent projections agree to `2.3e-16`; the shared directional derivative is therefore the relevant comparison for this tangent direction.

At the base and each side, direct-tape event values match the original replayed objective values exactly, and the analysis `frames_linear` arrays match exactly. The record states original replay enabled and diagnostic replay disabled; the original problem contract remains unchanged. Every floating-point value in the saved raw record is finite, including the full three-component JVP vectors and sample event values.

The helper contract is intentionally narrower: it fixes the event, side, CPU control shape/dtype, and exact stage count, and returns graph-connected values. It does not itself promise finite trajectory outputs. The diagnostic runner checks finiteness of base values, both complete JVP arrays, and both sides' sample values before saving them.

| α | ζ at sampled endpoint | selector | native J | R (squared mixed-gradient merit) |
|---:|---:|:---:|---:|---:|
| `1.742031084966799e-7` | `−1.0843947251260033e-7` | right | `0.061192476880111604` | `0.004478420533101891` |
| `6.968124339867196e-7` | `+3.9009734109640704e-8` | left | `0.061192475615450645` | `0.004485870697560415` |
| `1.3936248679734393e-6` | `+2.3560838968705866e-7` | left | `0.06119247393023507` | `0.004485863289941565` |

The two samples after the selector change have higher R than the first sample. Their paired-branch, face, and side-objective checks pass; the first sample's trace matches the base, while the later traces differ. These are sample outcomes, not evidence that this selector alone caused the R change. The first-order α is local to the chosen branch derivatives, not an exact event location, path certificate, or safe interval.

## Receipt and evidence locations

Attempt 2 raw SHA is `47b6b78d824a6ee817d2d014ea74ce14f68076d68c8c75a38ae6a0a7657297ff`; run child SHA matches. The archive gzip round-trip is lossless. The guarded child exited 0 in `26.0325 s`, with no monitor/resource termination and sampled peak RSS `596,967,424` bytes under the 2-GiB sampled limit.

- Plan and preflight: `LIMITER_EVENT_PLAN_ATTEMPT2_20261010.json`, `LIMITER_EVENT_PREFLIGHT_ATTEMPT2_20261010.json`
- Run record, run receipt, resource receipt, and compressed record: `limiter_event_20261010_attempt2/diagnostic.json`, `diagnostic.run.json`, `diagnostic.resource.json`, and `diagnostic.json.gz`
- Archive manifest: `LIMITER_EVENT_ARCHIVE_ATTEMPT2_20261010.json`
- Event capture mapping and replay note: `examples/weather_scenarios/fv_point_3h_limiter_event_probe.py:67-109`
- Side derivatives, parity, and samples: `examples/weather_scenarios/fv_point_3h_limiter_event_diagnostic.py:143-224`
- Failed attempt 1 source-closure log: `limiter_event_20261010_attempt1/diagnostic.log`

No commit, constraint, root, minimum, response, or forecast-value claim is supported by this diagnostic.
