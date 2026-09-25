# One externally QC-excluded point: qualified constructed FV response

This is the separately declared R3-Q profile in
`FV_POINT_QC_CENTERED_RESPONSE_PLAN.md`. Middle-time point 1 / flattened
parameter 5 is fixed **status 2, externally QC rejected**, with the
required canonical inactive dBZ fill. Eleven other point observations
remain detected. The FV state and boundaries are complete; the fixed
nonzero dynamics-prior mean, score, time schedule, point sampler and
within-time correlation are inherited from the constructed PR #200
problem. No QC decision algorithm or differentiable QC selection is
implemented. This status is distinct from genuinely missing status 1
and from detected clear sky.

The active middle-time indices are `[0,2,3]`, so whitening uses their
3×3 principal correlation submatrix. The 13-parameter signed direction
changes only active middle-time observation values 4, 6 and 7; the
QC-excluded slot and theta stay fixed. The nominal objective and all 26
gradient components are exactly zero. A fresh 26-column Hessian audit
found `lambda_min=0.9999999999997804`,
`lambda_max=3979.3772547945487`, and relative antisymmetry
`3.5664085271235254e-16`. The 54-stage minmod branch has slope margin
`8.046479676988153e-4` and face-flux margin
`7.752734889703382e-3`, above the retained `1e-4` gates.

The matrix-free adjoint returned the full 13-component direct,
indirect and total sensitivity. QC-excluded parameter index 5 has
**zero in all three components**; changing only its stored number by
`+0.25` leaves objective, score and control gradient exactly unchanged
under the fixed QC status. For a common +1 dBZ direction on the other
three middle-time points, direct response is 0 and indirect/total
response is `0.0016589920915979217`. True transposed-adjoint relative
residual is `9.127977280881999e-14`, and tangent true relative residual
is `1.7026349445225594e-11`.

All four actual signed endpoint reanalyses converged in one Newton
correction, retained the nominal full branch and `>1e-4` margins, and
had final maximum gradient below `1.874e-11`.

| Active-point offset h (dBZ) | Actual central difference | Adjoint relative error |
|---:|---:|---:|
| 0.001 | 0.0016589539149090891 | 2.3011977589233218e-5 |
| 0.0005 | 0.001658982547473372 | 5.752965669963691e-6 |

Both pass the predeclared `1e-4` criterion, and the smaller step has
one-quarter the absolute error to this precision. The guarded child
exited 0 in `139.52877083304338` seconds with 520 child-RSS samples
and sampled peak `345,882,624` bytes under the 600-second / sampled
1-GiB limits. The parent returned `execution_status=completed`,
`numerical_status=eligible`, `response_validation=passed` after
independently recomputing Hessian, adjoint/VJP, tangent, inactive-slot
and endpoint evidence. Source, input, plan and archive hashes stayed
stable before and after the child. The RSS cap is sampled, not a hard
allocation cap.

With the same active mask, canonical fill, prior, observation values
and direction, the *numerical* objective, control response, full
sensitivity vectors, four endpoints and signed pairs match the
status-1 missing run in PR #201 exactly. The input identities differ
because status 2 records an external QC exclusion, while status 1
records genuine missingness. This is an intentional fixed-mask
equivalence, checked by the read-only
`fv_point_qc_missing_parity_check.py` and its log. It does not show the
effect of changing a QC decision or provide an independent physical
forecast validation.

This closes **only the constructed one-QC-excluded-point profile**.
Whole-empty-time point response, the original zero-centered-prior input,
3-hour point sensitivity, finite QC-decision impact and physical skill
remain open. Raw child/resource/parent records and source snapshots
are in `point_qc_centered_attempt1/`.
The focused QC/missing/centered/response/refinement suite passed 64
tests with 18 existing TorchScript warnings; error-level basedpyright
reported zero errors. GREEN and RED prelaunch reviews found no
status/mask/covariance or scope blocker. The aggregate
`FV_POINT_QC_CENTERED_RESPONSE_EVIDENCE.json` binds the base commit,
raw report, source/test hashes, checklists/KG and exact verification
commands. This evidence is distinct from a full CPU/package CI run.
