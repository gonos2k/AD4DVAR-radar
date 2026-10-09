# Tangent resume adapter — final GREEN code and provenance review — 2026-10-09

## Disposition

No blocker found for the reviewed resume adapter and source-bound plan. This review made no code changes and launched no tests, FV, gradient, HVP, guard, or weather calculation. I ran only the read-only plan/base loader against the saved artifacts.

## Frozen plan and input provenance

The final resume plan SHA-256 is `d5c4fa93ff4c3fddedb6d458fedea916c0f06d32edfe8bf55d2fd44449ac9d10`. All **129 source pins** and **105 archive pins** match their current bytes. The resumed base loader returned control SHA `ed106d7bc277906141e9d4a52aea529c575c9ac17df334fcbfb8d9c5013d44f7`, theta `0.332026125873074`, native `J=0.0612512070514904`, and final saved `F²=0.005238386295728529`. The previous-step efficiency diagnostic is explicitly tied to its source control `747e5506c4e2ee966581431e052f75d8a0750844c6712aeca8aab9149f827157` and theta `0.3288283299901934`.

The adapter validates the original PR 267 plan and archive receipts without calling the original tangent plan loader. The resume-specific `_load_plan` pins the new live source set, old source snapshots, shared-math snapshot, previous plan, gzip/run/resource receipts, and output source manifests. The read-only loader check passed at the final plan hash with the 129/105 maps above.

## Endpoint and execution path checks

The loader walks all three historical accepted controls and theta values in order, checks each unique accepted trial against its committed control/theta, and verifies the two HVP history labels at each point (base, theta, side, phase, operator, scope, and completion). It closes the final chain to the expected `ed106d7b…` endpoint. It also requires the top-level last-confirmed closure to equal the last iteration's final repeat, then checks the final trial/repeat J, F², each side gradient, and trace signatures.

The resume wrapper injects its own plan and base loaders through the shared runner entry points and supplies the resumed base hash and theta. The shared child freshly observes the base under the current source, comparing the current fixed input/runtime, native J, F², both side gradients, traces, face, and branch pair with the saved endpoint before it constructs a new tangent direction. New HVPs are made later through the existing `current_tangent` path at the current point. Old HVP vectors are validated only as historical receipt metadata and are never supplied to the resumed operator.

The exception path restores the supplied initial theta for a zero-commit resume. The guarded parent's final closure now requires `current_theta` to equal the plan's initial theta when there are no accepted steps, or the last accepted commit's theta otherwise; it also checks the final control and hash. The preceding-step efficiency summary is labeled separately from the resumed endpoint fields.

No source-map or loader discrepancy remains in this final preflight. Root remains the sole authorized launcher for any later guarded continuation.
