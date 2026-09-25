# G5c fixed QC-rejected point rows

`FVPointResearchProblem` now accepts a third **fixed prepared-observation**
status: `0` detected and used, `1` genuinely missing, `2` externally
QC-rejected. Status `3` for censored data remains unsupported. Both inactive
statuses require the canonical `min_dbz` prepared fill, but remain distinct
in the status tensor and fixed-problem identity. This is consistent with a
prepared product that retains separate missing and QC masks after replacing
excluded values by a fill. The point research problem does **not** infer QC
from a value, zero quality, or a missing mask, and it does not run a QC
algorithm. The source product must supply the fixed status and provenance.

For each time, only `status==0` points enter the standardized residual.
With declared same-time correlation `C`, the inverse square root is formed
from its detected principal submatrix `C[V,V]` before the pseudo-Huber cost.
QC and missing parameter slots are finite but inactive, so their objective
gradient and mixed control-parameter JVP are zero; they cannot be smuggled
back in through full-matrix whitening. At least one detected point per time,
positive quality, finite FP64 prepared data, fixed masks, a fully known model
state and fixed fully known inflow boundaries remain required. An all-QC or
all-missing frame is rejected.

The new 4×5 fixed-control tests use QC rows at different points in all three
times and a nonidentity four-point correlation. Replacing those QC statuses
with genuinely missing statuses leaves the objective, forecast and score
exactly unchanged but changes the fixed-problem identity. Perturbing only QC
parameter slots leaves the objective and forecast unchanged; the QC gradient
components and mixed JVP are zero. Tests also reject noncanonical prepared
fills, zero quality, censored/unknown statuses, all-QC frames and in-place
status mutation. The five affected point-observation suites passed **66
tests**, with 18 existing TorchScript warnings. Pinned basedpyright 1.39.9
reported **0 errors, 0 warnings, 0 notes** after the test typing fix.

This closes only **explicit, fixed QC exclusion in the prepared synthetic
point profile**. It does not validate a radar product's QC decisions, accept
raw QC payloads, support censored or all-invalid frames, cross-time error
covariance, footprint/regridded observations, a newly qualified stationary
response, or physical forecast skill. The existing missing-only and
all-detected problem identities are preserved; a QC-status problem receives
its own identity and cannot reuse their cached evidence.

`FV_POINT_MISSING_RESEARCH.md` and its earlier SHA manifest remain unchanged
as the missing-only baseline. Its historical QC-unsupported statement refers
to the source measured there; the extension and tests here are separate.
