# RED preflight: PR263 correction resume

Scope: bounded read-only review of the new resume adapter, plan, dispatcher, and focused regression coverage before any guarded FV launch. No FV or HVP was run by this review.

## Finding

**No blocking provenance, stale-gate, provisional-state, or commit/refusal defect found.** The resume plan pins the accepted PR263 producer plan, step, run, resource receipt, and archived source snapshot. Its base is control `6c0fe4858631513f665f91c77006d4641e564862d8141111c0030c91d2bb74bd` with recorded J `0.0613117140406184`, Phi `0.010815707540872811`, and `||g||inf = 0.11651289508164255`. The state is a valid committed continuation point, not a root certificate.

`fv_point_3h_correction_resume.py` checks the plan digest, exact PR263 lineage, policy, source/archive set and hashes, and the pinned step/run/resource hashes (`:48-104`). Its base loader requires finished/completed receipts, exactly two committed sequential corrections, one HVP started/completed per correction, the final accepted control and metrics to match `current_state`, passing dual Armijo and strict branch checks, unchanged source/input/runtime, and closed parent/resource records. It also hashes the stored control tensor against the expected endpoint (`:107-182`). The current HVP, fresh full endpoint recheck, and commit protocol remain in the unchanged guided kernel.

The adapter returns only the final accepted state plus resume provenance; it does not return the PR263 `cycle_reference` or cycle assessment. The generic child therefore will not take the historical 2cd J/Phi recovery exit (`fv_point_3h_model_guided_continuation.py:461-470, 523-531, 700-722`). Requiring the source receipt's `cycle_recovered` status is only an input eligibility check; that status is not propagated as the resumed run's comparator.

Commit and refusal handling appear safe. The kernel updates its current control only after endpoint recheck and closure, then writes the committed state; timeout and direction refusals retain the last committed control and mark any active trial uncommitted (`fv_point_3h_model_guided_continuation.py:381-406, 660-695, 725-768`). The outer runner requires fresh output/resource/log/parent paths and checks child closure, optimizer count, and final iteration/control equality (`:772-824`).

The new adapter CLI defaults to the separate `correction_resume_attempt1/` receipt paths (`fv_point_3h_correction_resume.py:190-198`); those paths were absent at review time. Invoking the shared guided CLI directly with the new plan keeps its old `model_guided_attempt1/` defaults, which are already occupied, and will safely refuse the collision. Use the adapter CLI defaults or pass fresh paths explicitly.

## Review limits and small note

This was a static preflight; no launch receipts exist yet. The existing guided kernel's baseline mismatch message still says “accepted PR #258 endpoint” even for this PR263 adapter (`fv_point_3h_model_guided_continuation.py:522`); that is a stale diagnostic label only, not a gate bypass. The newly added tests cover real receipt normalization without cycle metadata, plan digest rejection, uncommitted/lineage/gradient corruption, dispatcher selection, and reuse of the shared runner (`tests/test_fv_point_3h_correction_resume.py:51-108`). Root reports the combined 38 focused tests passed, 18 pre-existing warnings, zero type errors, and isolated Graphify preflight completed with shared graph preserved; those execution checks were not rerun as part of this RED pass.
