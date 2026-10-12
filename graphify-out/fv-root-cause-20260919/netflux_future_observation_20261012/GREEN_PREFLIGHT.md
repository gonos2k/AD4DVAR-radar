# GREEN preflight: paired terminal observation

**Disposition: clear to launch under the final frozen plan.** The earlier v1
plan was superseded after RED review found that its parent wrapper trusted
child exit code without requiring resource and child-receipt closure. The v1
plan and source bundle remain preserved in `preflight_v1.tar.gz`, documented
as unlaunched in `PREFLIGHT_V1_NOTE.md`. This review performed no FV
objective, forecast, gradient, HVP, or terminal score.

The final plan SHA-256 is
`7c2b7f317e90e49a0c1bd9a289529816b8cef91e9d33048c8017b9b884279ae5`; its
source bundle SHA-256 is
`2a5eb415af2b8f873d319b45fd443894cba8c4533bc5694b4013ea96f8caf833`.
The bundle contains exactly the 209 source paths in the plan with matching
member hashes; the plan lists 283 archive paths. Receipt-only `load_plan`
verified the exact allowlists and live source/archive hashes.

The observer admits the pinned start control `0aaf…655b` and final control
`b647…88660`, the `minmod_net_flux_fp64_v1` transport, and the old fixed-case
asset. Metadata-only loading verified the problem identity, parameters,
truth, observation grid, and saved 18-step, 10-minute, +12000-second score
profile. Runtime and full input identities must match both endpoint receipts
and the fixed-case manifest. Before and after each score, the child checks
the deadline and source/input/runtime/asset identities. It writes atomic
progress receipts and allows exactly two calls, start then final, under a
600-second internal / 660-second external guard. Optimizer steps, candidate
evaluations, and selection use are zero.

The parent now requires the guard's resource status and a closed child receipt
with matching plan/version, exactly two started and completed scores, finite
nonnegative values, zero optimizer/candidates/selection/claims, and unchanged
source, input, runtime, and fixed case. It exits nonzero on an invalid or
missing child receipt. The final source and bundle pins include this wrapper
fix. The 15 focused mocked tests pass and basedpyright reports no errors. The
paired terminal-E result remains descriptive; it does not select a point or
establish forecast skill beyond these two fixed controls and this evaluator.
