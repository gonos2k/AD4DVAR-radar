# R2-O original correlated point input: current-policy support decision

This is the read-only decision declared in
`FV_POINT_ORIGINAL_POLICY_PLAN.md`. It does **not** run another FV
optimizer, adjoint, VJP or nonlinear reanalysis. The original
zero-centered-prior 4×5 correlated off-grid point input retains its
frozen problem and parameter identity. The classifier pins that
problem digest, every original input tensor digest, both historical
raw-manifest digests and the distinct constructed problem/report
digests. Seven bounded numerical
attempts are preserved in the R2-O ledger; the classifier rechecks
every file in the two latest sector-attempt manifests, their child,
parent and resource statuses, and the restart lineage between them.

The triple-decrease sector attempt accepted six corrections, including
two full-signature switches. Its seventh Newton step rejected all 16
candidate scales under the declared three-metric rule. The last
accepted maximum gradient was `0.006432750851827679`, far above the
unchanged `1e-10` stationary-response gate. The later
gradient-merit-only restart used that sixth accepted control, accepted
one switch and five same-signature steps, then rejected all 16
seventh-step candidates because measured gradient merit increased.
Its last accepted maximum gradient was `0.006602351499307973`.
Both attempts had seven converged PCG solves, child exit code **2**,
no resource termination and parent execution classification
`completed` for the bounded *refusal*. Neither produced a final
qualified root, adjoint, full VJP or signed endpoint response.

The exact original input identity is
`3de086e6436c21913899fcc7518fcf3eca62a42d552f94c6eb1e2b293665a6e5`.
The separately constructed fixed-centered-prior problem verified in
PR #200 has a different problem identity,
`464664431b230dbbbc997b3e5f5bcf669c8f2fb629ae298b6c0a6d61502e5b79`.
Its successful response cannot be substituted for the original input.

The support decision for **this exact input under the currently
declared nominal-search policies** is
`unsupported_by_current_declared_nominal_search_policies`; no
sensitivity is issued. This is not evidence that the stationary
equation lacks a root, that all point observations are unsupported,
or that a different sector-aware search could never succeed. A new
method would need its own bounded plan, fresh final stationarity,
curvature and branch margins, true adjoint residual and signed
reanalysis pairs. That research target stays open.

The read-only classifier and twelve focused regressions reject altered
responses, gradients, final acceptance, restart seed, constructed
identity, child/parent/resource status, manifest hash and an archive
path escape. A coordinated alteration of both attempt input identities,
their restart lineage and freshly sealed raw manifests is also refused
by the immutable original input contract. Error-level
basedpyright reports zero errors. `FV_POINT_ORIGINAL_POLICY.json`
records source identities, separate attempt execution and numerical
states, and `numerical_solver_runs=0` for this classification. The
historical raw reports were not changed.
The aggregate `FV_POINT_ORIGINAL_POLICY_EVIDENCE.json` binds the base
commit, original and constructed input records, two raw manifests,
source/test/plan/checklist/KG hashes and exact read-only commands.
The classifier itself pins the **preexisting** original input and raw
manifest digests. The aggregate also hashes the newly emitted decision
JSON, so its integrity is checked **after** emission; requiring that
postrun packet while generating its own output would be circular.
