# Event-conditioned response readiness

This note defines a future parameter-response gate for a qualified minmod-event endpoint. It is a design contract only; it does not launch a response run or authorize the existing smooth two-face response path.

## Current base status

The latest stored progress endpoint is `30bed163…` with an active stage-48 x-cell `[1,1]` limiter tie. Its status is `no_native_Armijo_progress_at_current_event`, projected-gradient infinity norm is `1.1416882854509208e-3`, and its terminal receipt has `first_order_complete=false`, `P2_complete=false`, and `minimum_claim=false`. The new maintain/release policy source has no attempt result yet. Neither point is an eligible response base.

## Local KKT response model

Use `z ∈ R^24` for the two-flow-face chart `Γ₂₄(z)`, where `Qy[4,3]=Qy[2,1]=0`. Let `h(z,p) ∈ R^m` be the active, dynamically discovered minmod equalities: `right-left` for same-sign slope ties or the selected operand for a zero-operand event. The policy caps `m` at two and rediscovers the set from each actual 360-stage trace.

For each limiter-side vector `σ ∈ {−1,+1}^m` and two-flow-face sector `τ ∈ {−1,+1}²`, construct the local smooth objective extension `Φσ,τ(z,p)` with `selected_minmod_extension` at those exact events and `selected_face_extensions` on the two flow faces. If the qualified first-order base has a reproducible product mixture `wσ,τ`, freeze those weights locally and write `Φw = Σσ,τ wσ,τ Φσ,τ`. The event-constrained stationarity system is

```text
F(z, λ, p) = [ ∇z Φw(z,p) + Nh(z,p)ᵀ λ ; h(z,p) ] = 0,
Nh = ∂h/∂z.
```

For a strongly regular local solution, differentiate the system as

```text
[ HL  Nhᵀ ] [ dz/dp ] = -[ ∂p∇z Φw + (∂p Nh)ᵀ λ ]
[ Nh   0  ] [ dλ/dp ]   [ ∂p h                         ]

HL = ∇²z Φw + Σᵢ λᵢ ∇²z hᵢ.
```

The score response is `dE/dp = ∂pE + ∇zE·dz/dp`. This derivative is conditional on the fixed event identities, product-sector mixture, and two-face chart. A release, a new event, a side-sign change, nonunique mixture, or singular bordered Jacobian invalidates this local response; rediscover and requalify the changed active set instead of continuing the old system.

## Base eligibility gate

Run a response only when all of these hold at one exact base control and one fixed CPU FP64 13-parameter vector:

- The original native objective has a complete same-point first-order qualification and P2 qualification for the actual active family. The active family contains exactly the two flow faces and zero, one, or two resolved minmod events; all event and flow normals are finite, independent under the declared scaled rank test, and hash-bound to the control.
- Every limiting combination `2^(2+m)` has a finite objective/gradient record with native/profile value closure. Flux-sector tangent gradients agree within their arithmetic budget. The event/flow normal cones have strict signed margins, the product Clarke hull is stationary within tolerance, and the full reduced Lagrangian Hessian is positive definite in every product sector.
- Each maintained limiter event has strict one-sided outward derivative margins on both sides. If a side has a certified descent direction, release that equality, make a real native-J step, rediscover the event set, and qualify the resulting point before defining a response. Normal rank or an event equality alone does not show the optimizer should remain on that event.
- The bordered KKT matrix is nonsingular with the qualified positive reduced curvature. The event identities and mixture remain stable for the requested parameter neighborhood, or each perturbed solve is stopped and reclassified when they change.
- A sealed response plan binds the base control and parameter hashes, event identities and form, selector-side mixture, chart, current source/archive hashes, runtime, and cumulative budget. Forecast scores do not choose controls.

For the fixed case, retain the existing perturbation definitions: `observation_dbz_0` (parameter index 0) uses `h = observation_std_dbz[0]/100`, and `background_beta` (index 12) uses `h = 1e-3`. Each parameter family would reoptimize at `p±h` and `p±h/2`, then re-audit the actual event set and first-order gate before scoring. The existing `active_two_face_response.py` handles only a qualified exactly-two-flow-face base with no limiter near-tie; it is not a substitute for this active-event KKT response.

## Source qualification

Current `src/advar/transport.py` SHA256 is `6a9f1f48cfc8ebe3d1687845fec5f0c8cff63c57bdfa6b9e56a172bbc478d035`. `active_two_face_response.py` still hard-pins `SUPPORT_AFTER_SHA` to `f6c680da7ca130fcbc505a03b73741591dc023f57a7745e47cfd49dd72622fab`. A future response must requalify its base and plan against the exact source used for that run and update the applicable validator/source seals; do not bypass the old byte-hash check. For `m=0`, the existing two-face route is conditional on its complete base gate and fresh source requalification. For `m>0`, require the event-family qualification and KKT contract above before implementing or launching a response.
