# R4 declared support decision for the archived two-hole collocated case

The same fixed 4×5 partial-observation inverse problem has already been
attempted three times under its declared GN branch and exact Newton
correction policy. The third run recorded the full **first-failing**
reason counts. Do not launch a fourth identical FV solve, relax the
`1e-4` face-flux gate, replace a source-bound archived report, or infer
that no stationary point exists.

Read only the six archived attempt-3 outputs and their existing SHA256
manifest `fv_partial_branch_gate_detail_run3_manifest.json`. Check every
output hash, preflight exit0, numerical child exit1 without resource
termination/monitor error, phase `partial_newton_refinement`, declared
nominal eligibility `refused`, zero tangent/adjoint/endpoints and
`response_validation=not_established`. Parse the raw branch reason
counts: three first failed the fixed signature and thirteen first
failed face-flux margin `<=1e-4`; the latter thirteen were never tested
for signature equality, so these categories may overlap. All sixteen
candidates were rejected before finite objective/gradient Armijo
evaluation. Record the raw aggregate `failure_category` as historical
metadata, not as an exclusive scientific cause.

Choose **intentional refusal for this exact input under the current
fixed-signature, final-margin local-response policy**. Return separate
fields for process execution, bounded resource completion, numerical
eligibility and response issuance. Because the archived numerical
child exited 1, do not relabel process execution as a normal zero-exit
run. Because its exact branch/margin gate refused, set
`support_status=unsupported_by_declared_nominal_policy` and
`response_computed=false`. Distinguish this from a physical-data
failure or proof of root nonexistence.

A future cross-sector search would be a different numerical method:
measure actual merit/objective at any candidate branch switch,
relinearize after acceptance, then independently qualify the final
stationary point, Hessian/PCG, margin, adjoint and signed endpoints.
That research target remains open as R4-R. This decision closes only
the R4 support-policy choice and failed-run accounting, using no new
FV numerical result.
