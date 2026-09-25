# One genuinely missing point: qualified constructed FV response

This is the separately declared R3 profile in
`FV_POINT_MISSING_CENTERED_RESPONSE_PLAN.md`. It reuses PR #200's
constructed fixed nonzero dynamics-prior-mean problem and its complete
4×5 FV model state and boundaries. One **middle-time** point, local index
1 / flattened parameter index 5, has fixed status 1 (genuinely missing)
and the required canonical inactive dBZ fill. The other 11 point
observations are detected. The middle-time correlation whitener uses
their 3×3 principal submatrix. The signed direction perturbs only the
other three middle-time points, indices 4, 6 and 7. This is not a
clear-sky measurement, unknown model state, or original zero-centered
prior input.

At the constructed control, `J=0` and all 26 control-gradient components
are zero. The actual 26-column Hessian audit found minimum eigenvalue
`0.9999999999997804`, maximum `3979.3772547945487`, and relative
antisymmetry `3.5664085271235254e-16`. The 54-stage nominal branch has
scaled slope margin `8.046479676988153e-4` and face-flux margin
`7.752734889703382e-3`, above the retained `1e-4` response gates.

The full 13-component matrix-free adjoint/VJP was computed. In the
three-active-point common-bias direction its direct term is zero,
indirect and total response are `0.0016589920915979217`; the full-vector
projection differs from the independent directional calculation by
`1.929879867024198e-17`. The true transposed-adjoint relative residual
was `9.127977280881999e-14` after 16 PCG iterations. The tangent used
14 PCG iterations and had true relative residual
`1.7026349445225594e-11`.

For missing parameter index 5, **direct, indirect and total gradients
are all zero**. Changing only that stored parameter by `+0.25` left
the nominal objective, score and control gradient exactly unchanged;
the parent independently repeated those checks. The mask remained
fixed, and the missing slot was never perturbed in signed endpoints.

Each of the four actual endpoint reanalyses used its changed 13-vector
and tangent predictor, converged in one Newton correction, and retained
the exact nominal 54-stage signature with slope and face margins above
`1e-4`. The largest final maximum gradient was
`1.8739884298480617e-11`, below `1e-10`.

| Active-point offset h (dBZ) | Actual central difference | Adjoint relative error |
|---:|---:|---:|
| 0.001 | 0.0016589539149090891 | 2.3011977589233218e-5 |
| 0.0005 | 0.001658982547473372 | 5.752965669963691e-6 |

Both pass the declared `1e-4` relative criterion. Their absolute error
ratio is `4.0000199739378`, consistent with second-order central
truncation in this local range. This validates one active-observation
direction, not changing QC policy, removing a measured observation at
finite amplitude, or every one of the 13 parameter components by
nonlinear reanalysis.

The guarded child exited 0 after `112.85639741597697` seconds, with
425 child-RSS samples and a sampled peak of `343,998,464` bytes under
the 600-second / sampled 1-GiB limits. The parent returned
`execution_status=completed`, `numerical_status=eligible`,
`response_validation=passed`. It independently recomputed the Hessian,
adjoint and tangent equations, full direct/indirect VJPs, inactive-slot
independence, four endpoint objectives/scores/gradients/strict branches,
and central differences from its **recomputed** scores. Source, input,
plan and baseline archive hashes were stable before and after the child.
The RSS cap is sampled, not a hard allocation limit.

A direct CLI call initially failed to import `examples` before starting
the child or any FV calculation. The wrappers and shared guarded child
command now use repository-root/absolute-path imports; an unrelated-CWD
CLI smoke test was added before this numerical run. The import failure
is not counted as an FV or reanalysis attempt.

This closes the **constructed one-missing-observation** response item.
QC-excluded and whole-empty-time profiles, the original zero-centered
prior point input, general minmod support, 3-hour point sensitivity and
independent physical forecast skill remain separate open work. Raw
child/resource/parent records and source snapshots are preserved in
`point_missing_centered_attempt1/`.
The focused missing/centered/refinement/response suite passed 60 tests
with 18 existing TorchScript warnings; error-level basedpyright reported
zero errors. GREEN and RED prelaunch reviews approved the bounded run
after the scope and CLI import corrections. The aggregate
`FV_POINT_MISSING_CENTERED_RESPONSE_EVIDENCE.json` binds the base commit,
raw manifest, code/test source, checklist/KG and the **exact test and
typecheck commands**. This is distinct from a full CPU/package CI run.
The missing-profile regression also compares the middle-time whitener
numerically with the explicit `[0,2,3]` principal correlation inverse
square root. As a read-only compatibility check, the current independent
numerical result predicate (`_valid_result`) accepted PR #200's archived
full-valid child payload
(`fv_point_missing_legacy_result_check.py` and its
`fv_point_missing_legacy_result_gate.log`); this replay did not rerun
FV optimization or reanalysis.
