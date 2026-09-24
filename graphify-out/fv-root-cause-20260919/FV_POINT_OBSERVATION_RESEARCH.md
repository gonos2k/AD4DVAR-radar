# G5 bounded off-grid dBZ point observations

This adds a **separate research profile** for fixed point observations at
interior state-grid cell-center index coordinates. The model state remains a
4×5 FV field; the example has four observation points at each of three
regular times, so the observation tensor is `[3,4]` and the parameter vector
contains 12 values plus one background-pattern coefficient. The control
vector remains 26 entries. No observations are replicated onto the state
grid.

The fixed operator applies bilinear interpolation **after** converting each
predicted echo-proxy cell to dBZ:

```text
H_dbz(x) = bilinear_sample(echo_to_dbz(x), fixed_points)
J(c,p) = sum_i pseudo_huber(sqrt(quality_i) * (H_dbz(M(c,p))_i-y_i)/std_i)
         + existing control and field-smoothness priors
```

It represents synthetic *point dBZ values*, not a radar footprint average
of linear reflectivity. A 0/20-dBZ pair gives 10 dBZ under this point
interpolator; averaging linear reflectivity first gives about 17.03 dBZ.
Coordinates use fixed `(row,column)` indices; complete 2×2 interior
stencils, FP64 CPU, nonnegative bilinear weights and in-bounds positions are
required. The sampler preserves constants and affine fields, and its
autograd transpose matches a manually assembled weighted scatter.

The initial background is an independent, fixed **state-grid** field plus
`theta * pattern`. Changing the first point-observation vector at fixed
control changes the innovation objective but leaves the initial background,
forecast and score exactly unchanged. This prevents an unintended
off-grid `y0`→state-grid backdoor. The synthetic background was prepared from
the earlier full-grid fixture, then frozen for this new point-observation
problem; it is not an independent meteorological truth. The fixed geometry,
values, independent diagonal std/quality, source digest, boundaries and
background all enter the problem identity. The source digest is the actual
synthetic case-generator file SHA256; it is not a radar product provenance
claim. Fixed observations, error weights, background, pattern, verification
and boundary tensors cannot carry hidden autograd state.

The branch gate is pointwise by default: it checks actual RK stages and
strict smoothness at the supplied control. A second smooth control may have
a different selector/face-sign signature. A caller comparing signed
reanalyses must bind the nominal signature as `expected_branch`; the
regression demonstrates that an unpinned candidate can pass strict tracing
while the explicitly pinned version rejects it. Pointwise eligibility is
not a finite-path certificate.

The 4×5 warm-control exercise traversed the strict 54-stage minmod branch.
Its objective is `0.07720456720208163`, score is
`0.00025526315789473595`, and max control gradient is about `14.33`, so it
is **not a stationary response result**. The analytic pseudo-Huber data
gradient differs from autograd by at most `2.22e-16` with fixed quality
weights from `0.5` to `1.0`; fixed-control score
gradient in all point-data slots is exactly zero. Mixed control/data
JVP–VJP transpose equality passes, and the exact objective HVP's central-
gradient relative error drops from `5.163e-7` to `1.291e-7` when the step is
halved. Exact numbers and source hashes are in
`fv_point_research_metrics.json`.

This first profile accepts only three complete detected point vectors with
independent std at or above the existing analysis minimum and quality in
`(0,1]`, fixed coordinates, one forecast lead,
fully known initial state/boundaries and no common-bias covariance. It
explicitly rejects missing, censored, QC-like absent values, out-of-domain
points, malformed/nonfinite FV boundary edges, and a correlated-bias setting.
The objective and forecast also reject malformed or out-of-detection-range
parameter vectors when called directly. A regridded product must transform its
covariance consistently; these four synthetic point samples are not a
regridded product. Physical radar positions, footprint averaging, unit
conversion order, correlated errors, independent verification, stationary
adjoint response and general minmod FSOI remain separate extensions.

Final focused validation reports **37 tests passed**, with 18 existing
TorchScript deprecation warnings; pinned basedpyright on the two modules,
fixture and two tests reports 0 errors/warnings/notes. A new shape-refusal
test initially reached a `reshape` error before the common parameter gate;
the objective now checks the contract first, and the final run passes.
GREEN/RED review found no remaining blocker in this bounded profile. The
source, command and log identities are in `fv_point_research_manifest.json`.
