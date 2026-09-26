# R4-R current-source two-hole partial FV root search: attempt 1

This is the one root-only experiment declared in
`FV_PARTIAL_SECTOR_ROOT_PLAN.md`. It used the same fixed 4×5 collocated
input tensors as the archived two-hole attempt, but the current
`FVResearchProblem` identity/source differs from the historical binary.
The probe compared the frozen observation/mask/whitener, parameter,
warm-control, verification and direction hashes, 58 valid plus two
missing observations, warm objective/gradient, and 54-stage branch
before the solve. The historical and current problem identity digests
were reported separately. Source, input, plan, warm-control and
parameter checks all remained unchanged after the run.

The guarded child completed its declared experiment with exit code **2**
and `numerical_status=root_refused`. The parent classified the execution
as `completed` because this is a valid bounded *refusal*, not a
qualified stationary response. Child wall time was 369.468 s under
600 s; 1,292 sampled RSS checks observed a peak of 357,384,192 bytes
under the sampled 1-GiB limit. The monitor reported no error or resource
termination. The child elapsed numerical time was 366.021 s. The
original problem, objective, prior, masks, fixed verification field and
normality/response thresholds were not changed during this attempt.

At the product-GN seed, the maximum objective gradient was
`0.008980677576039823`. The full 26-column exact-HVP Hessian audit was
finite and symmetric at relative discrepancy `2.29e-16`, with minimum
and maximum eigenvalues `1.0189732073166096` and
`4772.690094205241`. Its ratio `2.1350e-4` passed the declared
`sqrt(eps64)` gate. The seed passed the 54-stage core branch check;
scaled slope/face margins were `4.0805e-4` / `3.4301e-4`.

The exact-HVP Newton–PCG refiner made eight accepted corrections.
The first two changed the full limiter/face-sign signature and passed
the measured gradient-merit decrease rule; the second returned to the
seed signature. The remaining six accepted corrections retained that
signature and passed normalized Armijo. All 52 candidate traces were
admitted by the strict 54-stage core branch oracle with positive
pointwise margins. Of those candidates, 44 changed-signature trials
were rejected by the measured merit policy. Eight PCG solves each used
29 iterations and reported relative residuals at most `1.76e-13`;
the accepted-trial record separately reports true `Hs+g` relative
residuals at most `1.76e-13`. The parent-side success validator was
prelaunch tested to require linked seed/trial branch, curvature,
linear-residual and acceptance-policy evidence; this run did **not**
enter that success branch.

The accepted maximum gradient fell to
`0.006824468701446666`, still about `6.82e7` times the strict
`1e-10` threshold, so the refiner raised
`stationarity refinement iteration budget exhausted`. The final
accepted trial's scaled face margin was `5.5370e-7`, below the
`1e-4` response qualification margin as well. No fresh final-root
Hessian audit, adjoint, full VJP, tangent, signed endpoints or
nonlinear reanalysis was performed. `response_validation` and
`physical_validation` are both `not_performed`, and no sensitivity
was issued.

After iteration 2, the accepted step scales shrank (last three
backtracks 7, 8, 9) and the face margin decreased monotonically.
This is consistent with an approach toward a branch boundary under
this local policy. It does not prove root absence or a nonsmooth
optimum. The predeclared cross-sector policy with positive-margin
intermediate endpoints did not find a qualified root within eight
steps for this exact input. R4-D remains closed as a current-policy
support refusal; R4-R remains open as a distinct research target.

The focused affected suite passed **76 tests**, with 18 existing
TorchScript deprecation warnings. Pinned basedpyright 1.39.9 at error
level reported **0 errors, 0 warnings, 0 notes**. A code-only Graphify
refresh yielded 7,646 nodes, 66,971 edges and 311 communities.
`fv_partial_sector_final_tests.log`,
`fv_partial_sector_final_typecheck.log` and the Graphify log/snapshot
retain these separate local verification records. GREEN and RED reviewed
the branch policy, callback classification, parent publication gates,
negative mutations and final raw result. These checks are not a full
CPU/package regression or independent FV solver reproduction.

One non-blocking status edge remains: a *known* strict-branch failure
at the initial GN seed would currently be an execution failure rather
than a categorized numerical seed refusal. The present seed passed,
so it did not affect this attempt. A later response attempt requires
a newly declared numerical method, a fresh qualified final branch
and stationary point, then separately guarded adjoint/VJP and signed
nonlinear endpoints. This report does not authorize post-hoc retuning
of attempt 1.
