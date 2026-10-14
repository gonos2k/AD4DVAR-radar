# Final RED review: native reference, accounting, rays, and latest HTML

## Claims supported by the saved records

The master report, task checklist, and final accounting agree on the current
best-J tracked endpoint `6ca56498…c0c4d5`, native J
`0.0601191808341518`, and a finite branch-gradient hull proxy of about
`0.00114502`, far above `1e-10`. The report calls that a sampled proxy rather
than a complete Clarke stationarity certificate. No terminal Hessian/P2 was
computed at 6ca, B is gated off, and the response/perturbed reoptimizations
are unrun. The known prior full Hessian is at fb0a; the face Hessians are at
other controls. No current minimum is established.

The current endpoint observer independently reproduced stored native J at
`6ca` and used one forecast trajectory with 18 leads. Its plan pins 45
source/archive files, all of which I rehashed successfully; the plan hash,
endpoint, endpoint closure, input hashes, and observer result links match.
Future truth is the stored independent NumPy/DOP853 trajectory generated from
the original case control (`adc5bfd0…ad3721e`), not the candidate. The truth
arrays and lead times match the fixed problem. The observer is marked
observation-only, with no score selection or minimum claim.

The 18-lead mean forecast MAE `1.163358` exceeds analysis-persistence MAE
`1.144573`; forecast MAE is lower through +100 minutes and higher from +110
through +180. The +180 MSE (`7.341130`) is distinct from the all-lead MSE
(`2.434426`). This is a same-grid independent time-integrator comparison, not
a held-out weather forecast. The raw observer field naming +10 as a
“crossover” is corrected in the report: +10 is the first improvement and +110
the first deterioration.

The direct-J arithmetic is consistent: 2,000 reference calls plus 506 later
tracked calls equals 2,506. `FINAL_ACCOUNTING.json` limits that count to the
audited lineage; it is not a census of every early interrupted qualification
or debugging call. Its 3,423.985-second charge is an estimated ledger with
unmeasured historical reservations, not a measured total-session wall-time
certification. These caveats are present in the report.

## Ray diagnoses and their limits

The growth ray pins its input, parameters, evaluator, transport, variational,
and matrix-free sources; its before/after source and input checks pass. It
uses five native-J points and four gradients, with no Hessian or score calls.
All five sampled limiter signatures agree, and the positive
`2e-7` growth change reduces native J by about `1.72e-10`. This supports a
one-dimensional fixed-flow/field growth descent observation. The branch audit
still classifies the base as `multiplefaces`; the result does not certify a
smooth full neighborhood or full stationarity.

The coupled ray reuses the saved current gradient with an explicitly stale
positive-definite metric, makes seven native-J calls, and makes no gradient,
Hessian, or score calls. At alpha `1/128`, J increases by `5.69e-11`; at alpha
1 it increases by `2.38e-7`, despite the negative linear slope estimate. Its
limiter signature changes at every nonzero sample, while the independent
flux-sign audit finds zero sign flips among nonselected faces. This supports
rejecting the extrapolated direction and does not identify one limiter as the
unique cause.

## Actionable provenance issues before the final PR

The 18-case HTML check is the latest
`HTML_UPDATE_VERIFICATION.json`: it reports 34 matching evidence-source
hashes, 1,031 valid relative links, no missing links, and a PNG visual check.
I independently verified all 34 `CASE_REVIEW.json` evidence hashes. The older
`HTML_VERIFICATION.json` still reports 15 stored cases and stale artifact
hashes; final packaging should cite the newer update verification or mark the
older file superseded.

The coupled-ray A row links only PLAN and RESULT. Its actual executor
`coupled_ray_20261013/executed_ray.py` has SHA
`0f3f5bd6839c657b9913573516cc2be7fbab911bf10587617d9a33544f896e1b`, but that
hash is absent from the plan, report evidence map, and HTML source links. The
ray imports `qualify_candidate.py` and `two_face_intersection/.../diagnose.py`
without pinning those helper files in its own plan. `FLUX_SIGN_COMPARISON.json`
records the executor SHA and ray-result SHA, but it too is absent from the
HTML evidence list. Add these hashes/links or state that coupled-ray source
reproduction is incomplete. Growth-ray source provenance is complete.

The report's replay-profile timing/equivalence paragraph refers to
`replay_off_equivalence_probe.json` at control
`16793291a9848e432ee9c567fa39828848dbce9fee2eb2ab1d4ffd57cf07bbce`, not the
current 6ca endpoint. Name that probe control in the report so its J/gradient/
Hessian-row equivalence and speedup are not read as measurements at 6ca.

The v6 partial `fixed_metric_face_qualification.json` prefix was overwritten
and is not restorable. Its previous digest
`6191484c29ac6ff2079a2d9bf58d4ddc42d709c31439b2efc67bfe8da04f23d8` remains
in `QUALIFICATION_CONTINUATION_PLAN_v6.json`; the surviving final receipt hash
is `4418f5d6…`. The report says a mutable prefix was overwritten, but does
not print the old digest or explicitly say those bytes cannot be restored.
Add the digest and restoration limit if final readers need to audit that
history.

The reports also omit the separate retained-graph continuation prefix that
ended at control `7748a873…2390806` with native J `0.0601210989417` after
eight accepted steps and no terminal P2. It is worse than the 6ca endpoint,
so 6ca remains the best-J point, but this later run should be identified as a
nonselected A-path result if the report is intended to summarize all
qualification work. The existing accounting limit prevents it from changing
the stated 2,506 complete-lineage count without a source-backed recount.

No physical, FV, HVP, score, ray, or test execution was performed for this
review.
