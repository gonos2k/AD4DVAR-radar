# GREEN saved-data review: event-direction resume at 48b0

The frozen resume run completed with exit code 0 in 79.6 seconds and sampled peak RSS of 612 MB under the 2 GiB / 660 second limits. The frozen plan records 158 source and 213 archive pins. Its final receipts report source, input, runtime, and deadline closure.

The producer freshly requalified the 48b0 objective, full mixed residual, theta, side gradients, and branch traces. It then performed 24 residual-row VJPs and one positive-definite 12D solve to build a new GN direction, plus two direct-tape event gradients/JVPs and four fresh selected-face HVPs at the same point. No rows, solves, or HVPs were run after the selected step.

The event-orthogonal direction passed both side cost-descent checks, but its residual-merit slope was positive (`2.0417953e-5`), so it failed the merit gate before line search. The fresh baseline GN direction passed at candidate 5 of 24, with alpha `0.04574449898`. The run therefore ended as a **single-supported-arm fallback**, not a completed two-direction comparison.

The selected baseline step passed one independent P2 repeat and was committed once. J fell from `0.0611841170` to `0.0610941756` (`0.1470%`), and R fell from `0.00437321975` to `0.00429802133` (`1.7195%`). G∞ increased from `0.02858857` to `0.03225294`, so the result does not show improvement in every gradient norm. The endpoint moved the selected limiter event across its local selector boundary; this remains a one-step diagnostic and carries no root, minimum, response, or score claim.

The committed control is `15516fbae36637547a151066dc4af5eaa0ce74af9dbc5e93fe24694ff96c4cc0`, theta `0.6882137924515473`. Postcommit readiness was deliberately not performed.
