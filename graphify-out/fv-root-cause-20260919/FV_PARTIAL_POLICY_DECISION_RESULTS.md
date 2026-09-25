# R4 two-hole collocated partial case: explicit current-policy refusal

This is a **read-only decision** under
`FV_PARTIAL_POLICY_DECISION_PLAN.md`; it did not run another FV
optimization, tangent, adjoint or reanalysis. The six archived
attempt-3 outputs were checked against their original SHA256 manifest.
The raw output and resource report say the preflight exited 0 and the
numerical child exited **1**, without wall/RSS termination or monitor
error. Its sampled child peak was `344,702,976` bytes over 528 samples
within the 240-second / sampled 1-GiB budget; child elapsed time was
`136.74595075001707` seconds. We preserve this as a **nonzero process
exit**, not a normal zero-exit success.

The numerical child reached `partial_newton_refinement` after GN's
four outer iterations and 83 PCG iterations. At Newton iteration 4,
all 16 line-search candidates were rejected **before** finite
objective/gradient Armijo evaluation. Three first failed the fixed
limiter/face-sign signature; thirteen first failed the scaled face-flux
margin `<=1e-4`. Because the margin check preceded the signature
comparison, those 13 may also have changed signature. The historical
raw `failure_category=face_flux_margin_refusal` is therefore too narrow
to describe the mixed 3+13 causes on its own; it is retained as raw
metadata, with the complete reason counts used here.

No qualified nominal stationary point was returned. The raw response
status is `not_established`: zero tangent, adjoint, signed endpoints or
nonlinear reanalyses were reached. The archive's after-run preflight
recheck remained `pending`; this decision is bound by archived
preflight/source hashes and output hashes, **not** a newly certified
after-run source snapshot.

The chosen support contract is:

> **This exact two-hole collocated input is unsupported by the current
> fixed-signature, final-margin nominal-correction policy. No exact
> local sensitivity is issued.**

This is a policy refusal, not evidence that the underlying inverse
problem has no stationary point or that all partially observed inputs
are unsupported. Simply deleting the signature or `1e-4` margin gate
while retaining the old-sector Newton–Armijo prediction would not
justify a response across a minmod/upwind kink. A future sector-aware
nominal search requires a separately declared numerical method and
fresh final stationarity, Hessian/PCG, branch margin, adjoint and signed
endpoint validation. It remains an open research target.

`FV_PARTIAL_POLICY_DECISION.json` records process execution,
resource, numerical policy eligibility and response issuance as
separate fields. The read-only classifier's seven regressions pass,
including tampered child response/status, exit code, resource
termination, reason counts, manifest path escape and output collisions
with the plan/manifest/raw archive; error-level
basedpyright reports zero errors. The archived attempt-3 report and
manifest are preserved unchanged.
The aggregate `FV_PARTIAL_POLICY_DECISION_EVIDENCE.json` binds the
current base commit, six historical output hashes, decision source,
test/typecheck/Graphify logs, checklist/KG and exact read-only
commands. This is evidence accounting, not a repeat FV integration test.
