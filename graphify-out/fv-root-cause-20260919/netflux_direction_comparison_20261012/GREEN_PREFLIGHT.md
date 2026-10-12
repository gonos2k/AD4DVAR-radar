# GREEN preflight: same-point direction comparison

**Disposition: clear to launch under the frozen scope.** This was a read-only
plan, code, source-bundle, and receipt review. No FV objective, gradient, HVP,
candidate search, or terminal score was run.

The frozen plan SHA-256 is
`36686ccf05c55ab9dff1d6c0c4658e14049a108eba402b76c86fc0b9ac904b35`; its
source bundle SHA-256 is
`556fec30ca5b0a06c0b00601c8652b616b02705a09d7ac82fcea3203fd0c46bc`.
The archive contains exactly the 209 source paths in the plan, and every
member hash matches. The plan lists 283 pinned archive paths. Calling the
runner's receipt-only `load_plan` verified both exact allowlists and all live
source/archive hashes; endpoint admission reads the already-pinned closed
receipt prefix.

The comparison is anchored at the admitted `b647…88660` control under
`minmod_net_flux_fp64_v1`. The runner first verifies native pair and prior-P2
closure, then checks outward release before the minimum-mixture construction.
It uses one shared 600-second internal budget (660-second external guard) for
common preparation and both arms. The baseline tangent arm runs first; robust
GN preparation is limited to 24 Jacobian-row VJPs and one dense solve, with
two HVPs per arm and at most 24 candidate checks per arm. A commit requires
both arms to be independently P2/full-G eligible and compares their repeated
residual merit, then objective, with scale-aware ties preferring the baseline
gradient. If GN preparation or either arm fails, there is no one-arm fallback.
The contract excludes future-score evaluation and makes no root, minimum,
response, or score claim.

The frozen plan and current code agree on these limits, exact inherited source
and archive pins, endpoint, numerical version, and selection policy. Focused
tests exercise selection eligibility, no-fallback behavior, and the 24-row /
one-solve accounting. Launch is within the reviewed contract; its outcome will
establish only same-point direction-comparison evidence under this evaluator.
