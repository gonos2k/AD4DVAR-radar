# Common-face curvature certificate

This argument applies only to the observed regular `qy[4,3]=0` event, with a
valid `FVFaceFluxCoordinateChart`, no other active flux faces or minmod ties,
and the same native objective, inputs, and parameters. It does not supply a
certificate for multiple active events or a global minimum.

## The two restricted functions are identical

Let `c = Γ(u)` be the existing chart with its face-flux coordinate set to zero.
Its analytic construction gives `Q(Γ(u)) = qy[4,3] = 0` throughout the open
valid chart domain, not just at one control. The positive and negative selected
extensions differ only in which donor multiplies this chosen volume flux.
On the chart both selected volume parts are zero.

The initial state and boundary data are the same. If the two extensions have
the same state before an Euler stage, their selected-face contributions vanish
and all other flux, limiter, support, and growth operations coincide. Induction
through SSPRK2 and both analysis intervals gives the same state and native cost.
Therefore, in real arithmetic,

\[
\phi_+(u,p)=J_+(\Gamma(u);p)=J_-(\Gamma(u);p)=\phi_-(u,p).
\]

Where the other selectors remain regular, both restricted functions are smooth
and their gradients and Hessians are identical. The Hessian of this scalar
composition includes the derivatives of the nonlinear chart; an ambient
Hessian projection alone is insufficient.

## Why one common Hessian suffices

At a point with zero common tangent gradient and a positive-definite common
Hessian, the restricted cost increases quadratically in tangent coordinates.
The regular level set supplies local coordinates `(u,v)` with `v=Q(c)`.
Strict positive outward cost derivative on the positive side and strict
negative derivative on the negative side give an increase proportional to
`|v|`, uniformly in a sufficiently small neighborhood. Together these imply
a strict local minimum of the original nonsmooth native J in the full control
space, subject to the stated regularity and physical-domain assumptions.

This is not merely a face-constrained minimum. It is also not a smooth
unconstrained-gradient-zero certificate or a global minimum certificate.

## Numerical implementation contract

Keep the agreed chart-gradient and full minimum-mixture-gradient infinity-norm
thresholds `1e-10`, the existing curvature floor, symmetry tolerance, and strict
two-sided normal checks. Check the actual 360-stage branch, chart validity,
nonzero normal, and mixture weight at the final control.

Compute all 25 columns of one exact chart Hessian at that final control.
Check three declared opposite-extension Hessian actions against the matrix
actions with a dtype- and operator-scale roundoff budget. These are numerical
consistency checks of the analytic identity, not an independent reconstruction
of the second matrix. Refuse certification on a failed identity or branch
check. Record the one matrix and its provenance explicitly.

A Hessian computed before a correction cannot certify the corrected point.
Using a shifted or previous-point Hessian as a search metric cannot certify
final curvature. A current 25-column correction Hessian followed by a final
25-column common Hessian and three identity checks costs 53 new HVPs; do not
report this as two independently calculated final Hessians.

## Sources

- `src/advar/transport.py`: `selected_face_extension`, `_volume_parts`,
  SSPRK2 transport and support updates.
- `examples/weather_scenarios/fv_face_flux_coordinates.py`:
  `FVFaceFluxCoordinateChart.from_face_coordinates` and its analytic inverse.
- `qualify_candidate.py`: actual analysis-stage selector audit.

This note changes the redundant calculation of common-face curvature. It
does not relax the original minimum conditions or candidate tolerances.
