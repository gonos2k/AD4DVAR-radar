# R4 seed branch refusal status contract

The PR #209 two-hole FV experiment remains an eight-Newton-step
`root_refused` result. Its archived child, parent, resource report and
manifest are unchanged. This follow-up repairs only a status boundary:
when product GN returns a control outside the two declared strict
minmod branch conditions, the probe now records
`numerical_status=seed_branch_refused` at `phase=seed_branch`. It does
not label the point as a Hessian failure because no exact Hessian audit
has occurred, and it never emits a sensitivity.

The catch surrounds only the GN seed's `branch_check` call. The
allowlisted `ValueError` messages are the strict minmod smooth-branch
departure and positive-margin resolvability refusal. A malformed
returned branch summary, unexpected `ValueError` or `RuntimeError`
still propagates as an execution error. Later Newton-candidate branch
failures continue to be backtracked by the unchanged refiner policy.

The parent accepts this numerical refusal only with child exit 2,
normal resource/provenance checks, exactly one `core_branch_refused`
trace linked to the GN control hash, an exact allowlisted reason, no
sector-refinement solve or trial, and exact top-level GN/trace schemas
that exclude seed curvature, final-root and response evidence. Any
recorded linear solve must belong to the preceding GN phase. Existing
root success still requires its admitted seed trace, exact curvature,
stationarity, trial/PCG/merit evidence and final branch. The new
status does not relax any numerical threshold.

Small regression cases exercise known and unexpected errors at the
seed, a malformed branch summary, known and unexpected candidate
errors, plus forged parent reports with wrong GN hash, reason,
admission, exit/resource state or injected curvature/final/response
fields, including nested GN/trace fields and a non-GN solve phase.
These tests use a one-control analytic objective or a fake
guarded child; the candidate cases use the actual refiner/PCG on the
analytic objective. The focused preflight checks do execute a 54-stage
FV forward/branch inspection and evaluate the fixed warm objective and
gradient. **They do not rerun product GN optimization, the guarded FV
root search, an adjoint or signed reanalysis.** The archive from PR #209 is a historical snapshot
bound to its own source and checklist hashes; this follow-up does not
rewrite it.

The affected five-file pytest selection passed **93 tests** with 18
existing TorchScript deprecation warnings. Pinned basedpyright 1.39.9
reported **0 errors, 0 warnings, 0 notes** on the changed probe,
runner and test. The code-only Graphify refresh reported 7,649 nodes,
66,980 edges and 253 communities. These logs are retained with the
separate source/evidence hash packet. GREEN and RED reviewed the
failure classification and negative mutations. No full CPU/package
suite or new guarded FV optimization experiment was performed.
