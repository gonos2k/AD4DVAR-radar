# Two-face response readiness

Read-only handoff for the existing `active_two_face_response.py` path. This note does not authorize a response run and does not alter the running attempt or any pinned source/plan.

## Current base is not eligible

The current event-progress receipt is not a qualified response base. Its status is `no_native_Armijo_progress_at_current_event`; the terminal point has J `0.0601191807868374`, control SHA `30bed163abb770fb4cb708cabd168852d895892da05d0655bc9ec604430044f2`, and an active x-slope tie at stage 48, cell `[1,1]`. The 360-stage branch reports the two expected near-zero flow faces, but it also reports the near-tie slope entity `x:1:1`. The terminal record explicitly has `first_order_complete=false`, `P2_complete=false`, and `minimum_claim=false`.

The two event normals are independent of the two flow-face normals, and the normalized four-row matrix has rank 4 with smallest singular value about `1.647e-6`. That geometry check does not establish stationarity or curvature at the active limiter. The last accepted progress point decreased J, but Armijo progress then stopped after an intervening stage-124 limiter transition. The exact event evidence and current receipt are in `attempt1/RESULT.json`.

## Existing base and launch gates

The dispatcher in `active_face_response.py` routes two-face qualification receipts to `active_two_face_response.py`. Its base validator accepts only a RED receipt with status `qualified_original_native_J_two_face_minimum` and `minimum_claim=true`, then checks the following at the same final point:

- A finite CPU FP64 26-control vector and matching control hash, plus the fixed 13-parameter digest.
- The current two-face source transition and its qualification-plan sidecar, source/archive pins, and retained launch snapshot.
- A 360-stage `multiplefaces` branch with exactly `y:2:1` and `y:4:3` near-zero faces and no near-tie limiter slopes.
- Exactly-two-face support, a passing first-order gate, full and 24D tangent Clarke-hull infinity norms below `1e-10`, four nonnegative reference mixture weights summing to one, and strict signed normal margins in every sector.
- Four positive-definite 24×24 sector Hessians, each finite, symmetric, hash-matched, and attached to that exact final control. The reference-weighted Hessian must also admit Cholesky factorization.
- The fixed-problem identity and a separately sealed response phase plan with `root_launch_clearance=true`, matching qualification/control/parameter/chart digests, runtime, source/archive hashes, and cumulative budget.

The current event-progress endpoint fails the no-limiter-tie, first-order, and P2 requirements. Its rank-four normal audit cannot substitute for these checks.

## Source-hash requalification

Current `src/advar/transport.py` SHA256 is `51721e9196a782568b7c33840f10c9fea0f678da6b2a632e949981731d83dca6`. The existing response validator hard-pins `SUPPORT_AFTER_SHA` to `f6c680da7ca130fcbc505a03b73741591dc023f57a7745e47cfd49dd72622fab`; the final A qualification and response phase plan also require that hash. The transport change is additive observer support and leaves the default physical path unchanged, but these gates compare bytes. When an eligible two-face base exists, qualify it against the then-current source hash and create a matching response plan/validator pin. Do not loosen or bypass the current hash gate.

## Existing 13-parameter response cases

The response runs two parameter families in the 24D two-flow-face chart while keeping the four-sector reference weights fixed at the qualified base:

| Parameter | Index | Step `h` | Reoptimized cases |
|---|---:|---:|---|
| `observation_dbz_0` | 0 | `observation_std_dbz[0] / 100` | `p0 ± h`, then `p0 ± h/2` |
| `background_beta` | 12 | `1e-3` | `p12 ± h`, then `p12 ± h/2` |

For each perturbation, the existing route starts from the base 24D coordinates and uses the base reference-weighted four-sector gradient and Hessian for Armijo reoptimization. It requires tangent-gradient infinity norm below `1e-10`, then re-audits the perturbed point's actual 360-stage selector signature, two-face first-order stationarity, and strict normal signs before computing the score. A perturbed-point P2/minimum claim is not made.

At the qualified base, the route computes the exact response `E_p - λᵀF_zp`, with `F_zᵀλ = E_z` for the same fixed-weight two-face tangent model. It also reports centered score differences at `h` and `h/2` and their refinement change.

## Call route after the base is eligible

Use a fresh frozen response plan and a fresh output path. The existing CLI reloads the fixed case and dispatches based on the qualification receipt:

```sh
.venv/bin/python graphify-out/fv-root-cause-20260919/native_reference_20261012/active_face_response.py \
  --qualification <qualified-two-face-receipt.json> \
  --phase-plan <frozen-active-two-face-response-plan.json> \
  --phase-plan-sha256 <plan-sha256> \
  --output <fresh-response-result.json>
```

The response receipt is a sensitivity result only: it reports `response_claim=true`, `selection_use=false`, and `minimum_claim=false`; perturbed curvature remains uncertified.

## Source locations

- `active_two_face_response.py:89-122,139-220`: A-plan/source seal and qualified base, first-order, and four-Hessian gates.
- `active_two_face_response.py:560-694,699-735`: perturbed two-face requalification and launch gates.
- `active_two_face_response.py:811-953`: exact base adjoint, four perturbation cases per parameter, finite-difference comparison, and response-only receipt.
- `active_face_response.py:420-442,764-779`: two-face dispatch and CLI call contract.
- `event_progress_20261015/attempt1/RESULT.json`: current event-progress status and terminal evidence.
