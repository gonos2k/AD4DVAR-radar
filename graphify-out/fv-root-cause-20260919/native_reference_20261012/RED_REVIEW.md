# RED review: native-reference A/B/C reset

## Stable evidence reviewed

This review covers the native-reference work in this folder, not the prior QR
progress task. No FV, HVP, Powell, score, or forecast computation was run by
RED. I read the original Powell/continuation receipts, the support-only source
transition, the fb0a and face-chart qualification records, the C forecast
receipts and derivation, the B response gate, and the current HTML case review.

The continuation3 source/plan closure passes read-only verification: its plan
SHA is `67fa44ac…9c52a01`, it pins 237 source members, 335 archive members, and
three extra source members, and the 240 current source/archive files match.
The only numerical transition is
`src/advar/transport.py` from `b0335b7a…937ee1a` to
`c9dde24c…1f7da9`. The archived bundle hash also matches. The support update
uses incoming/outgoing material differences for support only; echo update and
minmod reconstruction are unchanged. This preserves constant coverage on the
declared divergence-free domain while keeping the original capped J/R search
records immutable.

The two Powell starts remain budget-limited reference points, not qualified
minima. The resumed zero start replays its prior 699 successful calls and the
support fix reproduces callback 700. Its 298 continued Powell calls, two zero
equivalence rechecks, and the separate fb0a equivalence check close the
original 2,000-successful-native-J cap exactly; the “300 new zero native
calls” field includes the two zero rechecks, so do not add all three
equivalence probes to it again. The final zero start is best-so-far after a
cumulative call refusal. fb0a stopped at 1,000 calls. The full 26-control
fb0a Hessian is positive definite at a nonstationary point: its gradient
infinity norm is `0.06246855`, not below the `1e-10` gate.

The selected `8eff6b1f…` point is on the conditional `qy[4,3]=0` face. Its
exact 25D chart Hessian is indefinite (`lambda_min=-0.0306827`) and its native
full-control gradient is not zero. Two later modified-Newton face corrections
lower native J to `0.0601979708822`; the final face tangent gradient remains
`0.00680491`. The positive Hessian eigenvalue `0.373106` belongs to the
preceding correction point, not the final control. No eligible native
minimum is currently established. The HTML A summary preserves those point
identities and the distinction between Powell's native 26-control searches
and the conditional 25D face corrections.

B remains unexecuted. The response entry point requires a stationary,
positive-definite, smooth full-space base with stable selector signature and
fresh qualification at each perturbation. The available fb0a point fails
stationarity, and the later face point is a nonsmooth conditional-face point.
The response record correctly reports zero response HVP/adjoint work and zero
perturbed Powell runs.

C is an observation on the fixed `8eff6b1f…` candidate, not a selection or
minimum certificate. The canonical field indices are 1, 3, 6, and 18 for
10, 30, 60, and 180 minutes after the 20-minute analysis window; their
absolute model times are 30, 50, 80, and 200 minutes from initialization. The
forecast receipt checks the fixed-case hash and exact equality between its
terminal truth and the archived independent-reference truth. The reference
comparison correctly states that the `96108…` hash is prior candidate
metadata, while truth was generated from the original fixed-case control; it
reports a large candidate-versus-generator state mismatch rather than
claiming truth-control recovery. The preliminary index-0-as-10-minute attempt
is superseded and excluded; the two correctly indexed trajectories and all
three native-J evaluations are retained in the cumulative erratum. The
current HTML calls E observational only, discloses 2,003 total native-J calls
against the already exhausted 2,000 search cap, and makes no forecast-skill
claim.

## HVP accounting

Known returned HVP work is stage-specific: 26 exact full-space columns at
fb0a; 56 columns in the earlier single-face event path; 44 new columns in the
face-chart resume; and 25 new columns at the first accepted modified-Newton
face correction. The face-chart resume reports 50 columns but reuses six
columns from the earlier event at the exact same `ba4cb3ee…` control. The
modified-Newton total 75 is its inherited logical 50 plus 25 new; its 25
source-point Hessian columns are reused afterward as an approximate metric,
which adds no HVP. Thus, before any fixed-metric work, the known physical
returned total across those four independent tasks is 151, not the naïve
`26+56+50+25=157`. The v6 `prior_hvp_started/returned=75` is the face-lineage
carry-forward count and is not the total over all A work. Separate prior
qualification receipts include one started HVP with zero return, and an
earlier aborted face event has explicitly unknown partial HVP status. Keep
started, returned, reused, and unknown counts distinct.

## Fixed-metric A continuation result

The fixed-metric A continuation is scoped as conditional-face native-J
minimization. It uses the exact SPD Hessian from the preceding face point only
as an approximate descent metric after moving; the plan forbids treating it as
current exact curvature. Any minimum label requires fresh first-order gates
and two-sided exact chart Hessians at the same final control.

The native-J counter was corrected to include the scalar objective used by
each branch audit and reserve two values for a line-search trial plus its
accepted-point audit. The frozen v6 plan matches the final script and input:
plan SHA `6c17da4e…6cac3bce`, 239 source/archive entries checked with no
mismatches, continuation input checkpoint hash matched, and source/plan/input
closure are true.

The run accepted 16 fixed-metric steps on the same conditional face and ended
at control `16793291…07bbce`, native J `0.0601211072374`, down from
`0.0601979708822`. The last audited tangent/mixed gradient infinity norm is
`0.0121337385`, far above `1e-10`; first-order qualification is false, so no
minimum claim is supported. The run made zero new HVP calls, with prior
face-lineage count 75. Its phase charge is `2975.968s` of the 3600s envelope,
leaving `624.032s`.

The terminal status `native_J_value_cap_exhausted` is more precise as a reserve
refusal: 127 run values were counted against a 128-call cap, and the earlier
preflight call brings the phase total to 128. The runner stopped because it
reserved two values for a trial and its accepted-point audit while one slot
remained. It did not exceed the cap.

The saved HTML case review still summarizes A through the earlier `6a5d…`
point and should add this 16-step endpoint if presented as current. Keep B
unexecuted because no qualified smooth full-space stationary minimum exists.
Keep C's `8eff…` forecast labeled as an earlier fixed-candidate observation,
not as a forecast of the new `16793291…` face point.

No A-phase minimum, B response, or C selection claim is supported by this
continuation. A smooth full-space minimum remains distinct from a
conditional-face first-order point.

## Pending common-P2 dedup question

The proposed real-arithmetic identity is valid in the ideal chart: fixing the
normalized face coordinate to zero and solving the pivot with `atanh` makes
the linear face functional zero, so the two selected face branches have the
same restricted scalar function if all other limiter selectors remain in the
same strict cells. The implementation does not make that face flux bitwise
zero in every production reduction. The chart docstring explicitly warns
that its basis-weight reduction can differ from the production combined
streamfunction reduction; the saved current point reports production
`face_q=6.938893903907228e-18`. `selected_face_extension` routes this nonzero
roundoff residue to different positive/negative flux pieces. Thus the ideal
identity is a useful analytic statement, but it is not an exact identity of
the two finite-precision implementation paths at the saved control. Three
opposite-side action checks can document local numerical consistency; by
themselves they do not certify equality of the full 25×25 implementation
Hessians. Any deduplicated P2 receipt should distinguish the real-arithmetic
identity proof from its finite-precision forward residual and sampled action
checks, without claiming two independently computed exact matrices.
