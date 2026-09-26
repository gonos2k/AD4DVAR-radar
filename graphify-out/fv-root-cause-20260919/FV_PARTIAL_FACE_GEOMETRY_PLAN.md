# R4-R diagnostic-only face-flux geometry audit

Use the immutable PR #209 root-attempt manifest SHA256
`d80362f8acbb0feb1c82d60d5e9cde61a4863564d7f8f64e49467ba39d12cddd`
and its child/parent/resource output hashes. Require all 16
geometry/input-critical source hashes from the PR #209 child to match
current source bytes; only the PR #210 status-only probe and runner
may differ. This is a new read-only geometric derivation, not a
replay of the old process. Call the existing fixed-input preflight
before and after and separately build the fixed problem once; these
perform multiple FV forward/branch and warm objective/gradient checks.
Require the same two-hole
input identity and 58/2 observation partition. Do not rerun product
GN, Newton, an adjoint or perturbed reanalysis.

For the saved GN control and **all 52 trial candidate controls**,
confirm the recorded 26 FP64 control values match their SHA256. Use
the current fixed FV transport spec's `psi_basis` and
`coefficient_limits`, the production `bounded_fv_coefficients` map,
`psi = einsum(k,kij->ij)` and `face_volume_fluxes` to reconstruct
the time-constant face fluxes. For each control record the absolute
smallest and largest face flux, the minimizing face kind/index,
sign, and the ratio. Require each ratio to match the archived
54-stage branch summary to absolute `1e-12`. If a minimum is tied at
FP64 resolution, report it as ambiguous rather than naming one face.

Compare each candidate's full archived branch signature with the
preceding accepted point. Tabulate sign crossings only where both
points have the **same unique minimizing face**; otherwise record
the comparison as unknown. Report accepted/rejected trials and the objective/
gradient-merit policy outcomes from the archive without recomputing
the nonlinear objective. A unique face whose **absolute** flux tends
to zero with a stable denominator supports a face-boundary approach
diagnosis; it is not proof of a smooth stationary root's absence or
of a generalized nonsmooth optimum. Endpoint face signs do not
certify all points along a segment, and no classical response is
issued from this diagnostic.

This is a small read-only computation: two fixed-input preflights,
one separate fixture construction, 53 five-basis face-flux evaluations
and JSON reductions. Write only to a fresh path outside the raw
archive. Record source/input/plan hashes and elapsed time; recheck
raw archive hashes afterward and do not modify the archive
or numerical thresholds. If input/hash/ratio checks disagree,
classify the diagnostic as invalid instead of tuning a tolerance.
Pass the reviewed plan and probe SHA256 values as required CLI inputs;
the probe rejects pre-run or in-run drift against those frozen values.
