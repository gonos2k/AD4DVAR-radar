# Current partial-response preflight after PR181

The open G3b draft incorporated merged PR181's signed/zero-growth tracer.
The three historical partial-observation numerical attempts remain byte-for-
byte tied to their earlier measured source commits and raw preflight records;
none was rerun or reinterpreted as a response success.

The current no-solver preflight and `--validate-only` records were rebound to
the integrated source at merge commit `aae82ed`. Relative to the guarded
attempt-3 preflight, the fixed-problem identity, all observation/mask and
whitener input hashes, warm-start objective/gradient, and 54-stage branch
signature/margins are unchanged (nine compared fields). Only the expected
source hashes changed because the growth tracer now permits representable
signed/zero log growth; this particular partial case still has positive
nominal growth. Its strict limiter/face checks remain intact.

After the merge, 69 affected tests passed with 18 existing TorchScript
warnings and the selected changed modules/tests typechecked with 0 issues.
The exact commands, output logs and SHA256 are recorded in
`fv_partial_reanalysis_preflight_manifest.json`. This is a source/input
compatibility check only. It does not supply a new nominal stationary point,
adjoint, endpoint reanalysis or finite-path certificate. G3b remains open.
