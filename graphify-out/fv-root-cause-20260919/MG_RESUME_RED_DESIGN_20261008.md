# Model-guided continuation resume — RED design

Date: 2026-10-08
Scope: minimal resume extension of the existing model-guided module/test and numerical kernel. This is a provenance and state-transition design, not a new solver. No FV, gradient, HVP, PCG, forecast, adjoint, reanalysis or code change was run/performed for this review.

## Resume point and immutable producer

The October 7 producer completed three commits and exhausted its declared three-step policy normally. The only valid resume base is its last committed control:

- control SHA `69c4a794730750c09c1bcddf542b66131f64f5bfec362e02ac269413a75d4992`;
- J `0.06136925244260338`, Phi `0.018131732819468303`, `||g||inf = 0.09724503382238599`;
- child SHA `92497babfe8eae3c55e2dcd4d244e533740b395f050e1a207612fcb78680cb0d`;
- producer plan SHA `a75b5e474e68fb74475c3fd7fb2f31e85c891edcb06c40e83b4287b16a7b3136`;
- parent SHA `686a43f22088ca44025a29e7b52689848fe4ba37b1f3f14fae631bcf87a98f8a`;
- resource SHA `f1e1ca5e7329a76308dc4e6b58db6129ec5e157324f89fa64c8dee25359f8149`.

The producer is the immutable history, not a restart point at `05889584…` or e29. Require `phase=finished`, process `completed`, numerical `max_iterations_completed`, exactly three accepted/committed iterations, HVP started/completed 3/3, PCG 0, `candidate_committed=true`, no active iteration/candidate, and agreement among top-level `current_control`, its hash, `current_state.control_sha256`, and the last iteration’s accepted control/hash/state. Close the parent child SHA and sidecar resource receipt; never rewrite these files.

## Source-version bridge

The exact producer source/test bytes are archived in `model_guided_source_20261007` and described by manifest SHA `d713ccf9b68295abafeab4e833ac92989aca0a513bc5aac51153effd59828233`:

- old `examples/weather_scenarios/fv_point_3h_model_guided_continuation.py`: `67341f6d53c1a8dc9e867c9b359136575b6a5a4f207c5441a1eb4d75300b6492`;
- old `tests/test_fv_point_3h_model_guided_continuation.py`: `852d8054c0e1d1f180c155df846fde842daa302dcacf9e135d2fff57d371961c`.

The old raw run’s `source_before` hashes match these exact snapshot bytes. A resume plan should include an explicit `producer_source_snapshots` map with **only these two changed-current paths**, pinned to the archive paths/hashes. Current resume module/test hashes are new plan source pins; they need not equal the old bytes. Every other original numerical/operator/input source must remain byte-identical to both the producer plan/raw `source_before` map and the current resume plan. Do not generalize the exception to the full source tree. Keep the old raw `source_before/source_after`, producer files and manifest immutable.

The plan must separately pin the old producer plan, raw child, parent, resource, producer source manifest, and the updated QY/tangent provenance used by the model-guided base. Compare the numerical helper and iteration-kernel AST against the archived producer version after the minimal resume edit; expected differences belong only to resume/bootstrap/state wiring and its tests. If a numerical helper or candidate-loop rule changes, this is no longer a source-compatible resume and requires a new model review/plan.

## Fixed-input identity with a changed control

The producer’s `input_before.control_sha256` is `05889584…`; its `input_after.control_sha256` is the accepted `69c4a794…`. That difference is legitimate and records the three accepted updates. A resume must validate the producer’s accepted `input_after` against the reconstructed original problem at control 69. Compare parameters, archived observations, terminal truth, boundaries, time schedule, objective/prior and other fixed identity fields exactly. Do **not** require `input_after == input_before` wholesale, and do not use the old 058 control as the current control.

At the start of the new invocation, freshly evaluate J, the complete 26-gradient, Phi, production face flux and the current full strict branch/margins at 69. Match these to the producer’s final `current_state` and accepted iteration within the pinned FP64 scalar/hash checks. Any mismatch is an integrity refusal before a new HVP.

## New invocation state and budget

Use a distinct `experiment_kind=model_guided_resume` (or equally explicit resume kind) and fresh attempt directory. The new run starts with:

- `current_control` = producer’s final 69 control;
- `iterations=[]`, `optimizer_steps_applied=0`, `candidate_committed=false` for this invocation;
- `hvp_calls=0`, `hvp_calls_completed=0`, `pcg_solves=0`;
- producer counts/three earlier accepted iterations stored only as lineage fields.

It receives a fresh bounded allocation of at most 3 additional accepted steps and 3 new current-point HVPs under the same 240 s internal / 300 s external / sampled 1 GiB / one-launch policy. The producer’s 3 HVPs and elapsed time are not charged to or reused by this invocation. At each current point, freshly form d=-g, compute one live HVP of the same original objective, recompute alphaPhi from that Hd, and test actual J/Phi and the candidate’s own full strict branch. Do not reuse the old direction, HVP, alpha, candidate, or model predictions.

Each new accepted point must receive a fresh J/g/Phi/branch recheck and source/input/runtime/deadline closure before its commit. The new invocation’s `optimizer_steps_applied` counts only its own committed iterations; a separate producer count may record 3 prior commits. At every write, make `current_control/current_state` identify the last committed point in the resumed invocation, or 69 if no new point commits. An active trial remains explicitly uncommitted.

## Partial completion, refusal and root handling

A later timeout, direction refusal, zero/face refusal, or exhausted candidate grid must leave the producer artifacts untouched and preserve every new commit already closed in this invocation. Mark only the active candidate uncommitted. Keep HVP started/completed counts distinct; a timeout before the JVP start callback is zero started, and interruption during JVP is one started/zero completed. Integrity/source/input/runtime failure is not a budget refusal.

A root-sized gradient may only yield `root_pending_audit`; this continuation does not certify a full root, full-Hessian SPD, response, adjoint/VJP, reanalysis or forecast. An endpoint can pass its own strict branch while the signature changes from the preceding point; record it as a finite branch transition, not a same-branch path. Do not attribute new J/Phi decreases to a single face crossing because each d=-g step changes the full 26-vector.

## RED conclusion

The safest resume is a new bounded invocation over the same numerical kernel, seeded from the final committed 69 state, with a narrow source-version exception for the archived producer and its test only. The earlier three steps remain immutable history; fresh J/g/HVP work starts from 69 with fresh counters. Resume success would mean up to three more actual accepted full-space model-guided steps, not completion of the original stationary-point or response research.
