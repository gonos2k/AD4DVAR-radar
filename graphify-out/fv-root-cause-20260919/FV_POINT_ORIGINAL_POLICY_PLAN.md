# R2-O support decision for the original zero-centered correlated point input

The original full-valid 4×5 correlated off-grid point problem has one
frozen input identity and seven distinct bounded numerical attempts:
warm exact Newton, two basin searches, a terminal fixed-margin Newton,
the core-strict fixed-signature Newton, a triple-decrease sector switch,
and a gradient-merit-only sector switch. These reports remain intact.
Do not infer seven independent weather cases, retune gates to pass,
or assert that a stationary point cannot exist.

This is a **read-only support classification** anchored to the last
two source-/plan-/resource-bound sector reports and their SHA256
manifests, plus the prior R2 sequential report ledger. Verify all raw
manifest entries and the two final child/parent/resource statuses.
For the triple-decrease attempt, verify two signature switches and four
same-signature accepted corrections followed by all 16 iteration-7
candidate refusals; its last accepted maximum gradient is about
`0.00643275`. For the later gradient-merit-only restart from that sixth
accepted control, verify one switched and five same-signature accepted
corrections, seven converged PCGs, then 16 iteration-7 policy refusals;
its last accepted maximum gradient is about `0.00660235`. Both remain
far above the unchanged `1e-10` stationarity requirement. Neither ran
adjoint, full VJP or signed nonlinear reanalysis.

Classify this **exact original input** as
`unsupported_by_current_declared_nominal_search_policies`, with
`response_computed=false`, while retaining the independent execution
and resource statuses of each attempt. This is a support policy for the
available research path, not a proof of root absence or a claim that
the point observation operator is invalid. The separately constructed
fixed-centered-prior profile and its missing/QC/empty-time variants
are different statistical problems; their successful responses must
not be transferred to this input.

Any future root search across minmod/upwind sectors must be designed
as a new numerical method and checked against fresh stationarity,
curvature, final branch margins, true adjoint residual and two signed
endpoint pairs. No new FV optimization or reanalysis is authorized by
this classification itself. Preserve every historical report and its
source hashes. Emit a small JSON support record with provenance and
open-research limitations, and test fail-closed tampering of statuses,
gradients, manifests and response publication.
