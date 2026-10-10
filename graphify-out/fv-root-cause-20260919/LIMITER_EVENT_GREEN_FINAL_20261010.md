# GREEN saved-data review: 027e limiter event

Attempt 2 plan SHA-256: `3579c2bbf9f3213cec5669948e9de21e47da258720572b291b083c5355e3f79d`. All checks passed: **True**.
This review uses saved JSON, frozen hashes, and standard-library arithmetic only; it performs no FV, AD, or HVP evaluation.

## Event and derivative

At analysis Euler stage 124, x-interior cell [1,2], the direct-tape capture and replay-enabled reference match exactly on both active-face extensions. The event is `ζ = left - right`; the observed values are negative and same-sign.
`ζ = -1.5758912042e-07`, `q_abs_max = 39.0269037833`, and `|ζ|/q_abs_max = 4.03796112791e-09`.
The saved first-order event alpha is `5.58549281687e-07`. Side JVPs are 0.282140046701 and 0.282140046701; the JVP/gradient-dot checks and projected-gradient-dot checks pass on both sides.
Projected gradient norm is 77.5440358363; direction norm is 0.046447464754; cosine(projected gradient, GN direction) is 0.0783347243641.
The c[12:15] squared-gradient fraction is 99.96237518% after projection.
The side-gradient difference has projected tangent norm 3.33066907388e-16; its dot with direction matches the pure-normal prediction.

## Local chart samples

| α | actual move | ζ (− / +) | linear ζ (− / +) | J change | R change | trace matches base |
|---:|---:|---:|---:|---:|---:|:---:|
| 1.74203108497e-07 | 8.09129274142e-09 | -1.08439472513e-07 / -1.08439472513e-07 | -1.08439447253e-07 / -1.08439447253e-07 | -4.21643331361e-10 | -1.85667475578e-09 | True |
| 6.96812433987e-07 | 3.23651709716e-08 | 3.90097341096e-08 / 3.90097341096e-08 | 3.9009572247e-08 / 3.9009572247e-08 | -1.68630429065e-09 | 7.44830778377e-06 | False |
| 1.39362486797e-06 | 6.47303419347e-08 | 2.35608389687e-07 / 2.35608389687e-07 | 2.35608264914e-07 / 2.35608264914e-07 | -3.37151986884e-09 | 7.44090016492e-06 | False |

The smallest sample remains on the saved limiter selector and lowers R slightly. The middle and largest samples cross ζ=0, change the selector, and raise R; J falls slightly at all three. The linear alpha is a local first-order estimate, not a safe interval or a path certificate.

Attempt 1 remains a failed source-closure attempt after a late helper/test change; its manifest has `success_claim=false`. Its values are not included as successful evidence, and full reproduction of that attempt is not claimed because the exact helper/test source bytes were not preserved.

Attempt 2 used 26.032461374998093 s and sampled 596967424 bytes RSS; it completed with zero optimizer steps, zero HVPs, zero Jacobian row calls, and zero dense solves.

## Receipt checks

- `frozen_plan_and_preflight_pins`: **True**
- `all_154_sources_and_200_archives_match`: **True**
- `attempt2_archive_gzip_raw_run_and_resource_close`: **True**
- `attempt2_guard_completed_zero_exit`: **True**
- `source_input_runtime_closed`: **True**
- `event_side_parity_and_jvp_gradient_checks`: **True**
- `side_gradient_difference_is_pure_normal`: **True**
- `all_three_samples_match_linear_zeta_and_event_values`: **True**
- `no_optimizer_hvp_row_or_solve`: **True**
- `failed_attempt1_is_marked_without_success_claim`: **True**
