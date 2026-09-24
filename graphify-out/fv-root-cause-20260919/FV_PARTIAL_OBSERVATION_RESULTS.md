# Bounded partial-observation research profile

The shared `FVResearchProblem` now accepts genuinely missing cells in the
middle or last observation frame while the first frame, model initial support,
and all FV boundary support remain fully known. This is a fixed-mask research
profile, not arbitrary missing-data support or a new minmod response
certification. The original all-detected profile keeps its numerical path.

At preparation, a missing dBZ sample receives a finite internal fill, but
`missing_mask=True` and `valid_mask=False` remain authoritative. The binder
canonicalizes inactive parameter slots using the prepared value before
constructing the objective or background. The original residual and
observation whitener already zero invalid residuals and whitener modes; no
loss function or error covariance was changed. The first observation frame
must remain fully detected because its values define `B=y0+theta*pattern`.

The small 4x5/26-control test prepares two NaN missing samples in frames 1
and 2, leaving 58 valid and 2 missing observations. With common-bias standard
deviation 0.25 dBZ, the frozen whitener mode is zero at both missing cells.
Changing both inactive parameter slots to 100 dBZ leaves the fixed-control
objective, forecast, score, and 54-stage strict branch exactly unchanged.
At those slots, both parameter gradients and the mixed control gradient in
their joint direction are exactly zero; a selected valid observation still
has nonzero objective gradient. The nominal fixed-control objective and score
are `0.02127260121736458` and `0.00025526315789473595` in this new partial
problem. These values are not a new optimized analysis result.

Separate regressions reject a missing first frame (the FV preparation
contract requires known initial active support), plus QC-rejected, censored,
and zero-quality observations (outside this narrow profile). A two-lead
partial instance retains 180/240-second forecasts and 72 strict RK stages.
Missing samples are not interpreted as clear sky or censored reports.

The affected one-/two-lead and FV trajectory suite reports 63 passed with
18 existing TorchScript warnings (`partial_observation_tests.log`). Pinned
basedpyright reports 0 errors, 0 warnings and 0 notes
(`partial_observation_typecheck.log`). The exact commands, Python/PyTorch
versions, base commit and changed-code/log SHA256 are in
`partial_observation_run_manifest.json`. Graphify was refreshed code-only:
6,858 nodes, 65,431 edges and 261 communities; no semantic LLM extraction.

No new GN, Newton refinement, adjoint response, signed reanalysis, or
independent weather verification was performed. G3b remains open: qualify a
new stationary point for this partial problem and test its whole response
under fixed masks. No tolerance or physical model was adjusted to make this
profile pass.
