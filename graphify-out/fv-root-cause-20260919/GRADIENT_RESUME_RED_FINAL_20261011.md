# Gradient resume RED final audit

Read-only audit of the guarded successor attempt and its pinned endpoint. No source edits, test reruns, HVPs, candidate evaluations, or forecast-score calls were made.

## Finding

The single guarded attempt failed during current-direction qualification, before HVP/search. The traceback is `one-sided gradient jump is not supported by the common face` from `tangent_direction` (`fv_point_3h_tangent_continuation.py:159-160`), reached by the successor at `fv_point_3h_current_gradient_resume.py:192-193`. This is consistent with its measured common-face residual: tangent jump norm `3.8377744726e-13` versus support budget `7.9753545967e-15` (48.12×). It does not support silently widening tolerance or claiming a usable Q0 direction.

The attempt requalified the inherited P2 exactly: component match linf `0`, theta error `0`, both side gradients and traces match, and native-J error `0`. Outward native-J derivatives were `+0.1943639923` and `+0.0862438514`; release was correctly not required. The attempt started/completed zero HVPs, evaluated zero candidates, committed zero steps, and retained the confirmed `556a25d9…298a8b` control. The child exited 1 after 8.9708 s with no timeout, RSS termination, or monitor error; peak sampled RSS was 512,917,504 bytes.

The final verification receipt also confirms the recomputed minimum-mixture P2 match. The two earlier committed endpoints had common-face tangent-jump norms `6.2834356733e-17` and `7.4729342396e-16`; their receipts remain valid and the new unsupported direction does not retroactively alter their approvals.

One record-quality issue: the raw step leaves `numerical_status` as `preflight` while setting `execution_status=failed` and recording the precise common-face error. The prefix is safe, but a typed terminal state such as unsupported current direction would make this expected support refusal easier to interpret.

## Provenance and diagnostic limits

The receipt-only endpoint loader revalidated the 556 endpoint, its three closed steps, and theta `0.6926534545496984`. The successor plan hash matches the run receipt; all 196 source pins and 271 archive pins still match, including the frozen predecessor producer and test. An independent after-failure audit recomputed fixed-input and runtime identities and rechecked all source/archive pins against the initial anchors. The failed raw itself has no `source_after`, successful finish, or deadline-success fields; verification records the after-failure audit separately and derives `failure_stage=common_face_direction_support_before_HVP` from the traceback and current-point checks without mutating the raw. Launch resource caps were 600 s internal / 660 s outer and 2 GiB RSS. Preflight records 65 passed local tests and zero type errors/warnings; this audit did not rerun them.

The flux-only production diagnostic reports formula `q=0`, production `q=6.9388939039e-18`, face-audit residual `5.27e-16`, and production/formula normal difference `0`. Its stated scope excludes objective, side-gradient, HVP, candidate, and score evaluation. The common-face diagnostic reused archived gradients. These facts confirm the recorded face normal, but do not establish the cause of the tangential gradient jump or validate any tolerance change. Keep the current point unsupported pending the remaining planned diagnostic; no solver or release claim follows from this attempt.
