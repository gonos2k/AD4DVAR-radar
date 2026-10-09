# Candidate requalification GREEN final record — 2026-10-09

## Scope and ownership

I executed the one guarded candidate-requalification run recorded below after the refreshed plan and archive preflight passed. This memo documents that run and the evidence I inspected in its saved receipts. It does not claim an independent second execution. The root and RED reviewers subsequently audited the saved raw result independently. No further run was made. I changed no source code; this memo is the only file I wrote for the final record.

## Launch and artifacts

Working directory: `/Users/yhlee/ADVAR`.

Command executed through `functions.exec` → `tools.exec_command` with the repository `.venv` Python; the probe then invoked its guarded diagnostic child. The tool returned session `39058`; I collected its completion with `tools.write_stdin`.

```sh
./.venv/bin/python examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py --plan graphify-out/fv-root-cause-20260919/CANDIDATE_REQUALIFICATION_PLAN_20261009.json --plan-sha256 5b7ebac50ebc38c0eaf34539c5bdf89e00c4d342ef96f2a96bc12a93fa30bfb4 --output graphify-out/fv-root-cause-20260919/candidate_requalification_20261009_attempt1/step.json --resource graphify-out/fv-root-cause-20260919/candidate_requalification_20261009_attempt1/step.resource.json --log graphify-out/fv-root-cause-20260919/candidate_requalification_20261009_attempt1/step.log
```

The run also wrote `graphify-out/fv-root-cause-20260919/candidate_requalification_20261009_attempt1/step.run.json`. The specified `step.log` file exists but is empty; the child, parent, and resource JSON receipts are present. The child SHA-256 in the parent receipt is `14a4b1c05363e316fddd3291946852e3334232388f975f1316030e85a4a0abfc`. The plan SHA above preflighted successfully with 124 source pins and 86 archive pins; the fixed-base loader and archived-direction loader passed before launch.

## Result

The child and parent both report completed execution and one accepted `one_nonsmooth_requalified_step` at alpha `0.013141672926849611` (candidate control SHA-256 `6b29dacd01fc30ee4041e93dd3ad110c29699c4553d098694086eadeb580a743`). The raw `numerical_status` remains the inherited generic label `one_nonsmooth_coupled_step_accepted`; interpret the result using `candidate_type`, policy, and closure fields. The run made 2/2 current HVP calls and zero dense solves.

Native objective decreased from `0.061307587424344834` to `0.06126370581028981` (−0.071576%). The scaled residual norm decreased to `0.0794631`; squared residual reduction was 3.67749%. The accepted candidate passed both J and F-squared Armijo checks. Its current minus/plus traces were complete, strict, tie-free, and matched one another on non-target choices and face signs. This is a local current-side pair result; the changed neighboring selector choices are diagnostic evidence and do not certify a simultaneous active set along the path.

The fresh scaled directional equation `D_F δ + F` had relative residual `5.354634073739147e-13` and absolute norm `4.335422378036445e-14`. The candidate face residual was `1.734723475976807e-18`, below its `5.199265539511744e-16` roundoff bound. Final closure passed proposal/native objective equality, individual side-gradient equality (`gradient_matches_proposal=true`), side-objective equality, branch-pair and trace equality, merit, face audit, fixed-input, source, runtime, and deadline checks. The parent resource receipt reports 50.699 seconds, 656,474,112-byte sampled peak RSS, no termination, and no monitor error.

The mixed gradient infinity norm at the accepted point is `0.0345217`; it is not a root. There is no current full-Hessian refresh or new-point curvature/minimum certificate, no finite-path support certificate, and no response claim. The result is one local requalified step; a later iteration needs an operator at its new point.

## Separate verification evidence

The implementation-pass records report 47 focused tests passing in 2.74 seconds and zero type-check errors, warnings, or notes. I did not rerun those checks for this record. The root and RED reviewers separately inspected the saved raw run; their audit is distinct from my launch and preflight.
