# PR265 kink runner: RED preflight

Date: 2026-10-09
Scope: read-only mathematical and execution preflight of the stable
`fv_point_3h_nonsmooth_coupled_probe.py` runner and its frozen plan. No finite-
volume objective, derivative, HVP, or candidate computation was run in this
review. No source or test code was changed.

## Finding

No blocking static math or control-flow defect found. This is a preflight only;
the actual FV branch traces, derivative parity, numerical solve, line search,
resource use, and final closure remain unverified until the single guarded run.

The plan hash was checked read-only and matches
`83ea077a6d7ba615a84729a3588a92b156687dd02a8fc7320cf9b90ecf5ce299`. The
runner uses the corrected PR26495a accepted endpoint and validates its producer
plan, accepted step/run/resource receipts, source map, control digest, runtime,
and fixed input before beginning the diagnostic.

## Mathematical and semantic checks

- The existing full objective is retained through `problem.objective`; its
  robust observation loss, control prior, and field smoothness prior remain in
  the gradient. The runner fixes objective-gradient scale at 1 and scales only
  the face constraint by `max(abs(face_weights))`.
- The side mixture is a convex combination of the two selected gradients with
  `theta` constrained to `[0, 1]`. The 27-variable linearization uses the
  mixed ambient Hessian, the gradient jump as its `theta` column, and
  `normal / qscale` as the face row. Its residual is the mixed gradient and
  `Q / qscale`; these are the Jacobian and residual of the declared coupled
  system.
- The face chart reconstructs one flow coordinate with `atanh`, preserving
  `Q = 0` along the candidate path. It measures the actual curved displacement
  against radius `0.05`. The trial guard requires both side slopes to be
  negative and separately checks actual objective and scaled residual-square
  Armijo decrease.
- The 360-stage analysis trace excludes only the selected face from the raw
  flux-sign gate. It still checks all other face signs and minmod branch
  choices, and rejects near-zero slopes or active limiter ties. At each
  `eta = +/-1e-8` parity point, the production face flux audit verifies the
  requested side before native/extension `J`, gradient, and HVP comparisons.
- The baseline curvature calculation uses the correct constraint correction
  `Hmix - lambda * Hessian(Q)`, restricted to the tangent space. The code labels
  it as a base-point diagnostic and does not use it as a candidate curvature
  or minimum certificate.
- Resource caps are internally consistent: 4 parity HVPs plus 52 basis HVPs
  equal the 56-call maximum; one 27-by-27 dense solve; at most 8 candidate
  checks; 600-second internal deadline, 660-second outer guard, and 1 GiB RSS
  cap. Face, trace, parity, and extension-value qualification refusals occur
  before the Hessian loop.
- A candidate is recorded as accepted only after fresh native objective,
  both side objectives and gradients, residual merit, face value, branch
  traces, source hashes, fixed-input identity, runtime identity, and deadline
  closure agree. The runner does not apply the candidate to the active model.

## Remaining limits

The static audit cannot establish that all prerequisite gates pass at the
selected face, that the dense solve has a small observed residual, or that a
candidate exists within the radius and eight trials. The source-level
preflight also does not establish numerical convergence, a smooth root, a
minimum, a response, or forecast validity. Report the guarded run's actual
status and receipts separately from those claims.
