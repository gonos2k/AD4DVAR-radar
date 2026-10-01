# R4-R fixed-dynamics field correction, attempt 1

Run one source/input-bound correction of the original two-hole 4×5
collocated problem from the PR #212 alternate seed, control SHA256
`125e0200fdf85f996715ba1bc04aa3f631333614bf15cf2c16630b94c1897421`
and branch SHA256
`50d3b1a4bad6761b14806c62708400d87d16e506ab08ef545f7ba9d870bece6d`.
Reuse the existing validated PR #212 seed-evidence loader, including
PR #209/#212 raw manifests and source/input/runtime checks. Keep the
original 58 valid plus two missing observations, parameter vector,
objective, zero-centered prior, boundary, time and verification field.
No product GN, response, score or perturbed reanalysis is run.

Partition the existing control as `c=(z,d)`, with field `z=c[:20]`
and six fixed flow/growth controls `d=c[20:]`. Optimize the exact
reduced scalar `J_field(z,p)=J(cat(z,d),p)` using the existing
`refine_stationary` gradient-JVP Newton–PCG implementation. Keep `z`
in AD; `p` is fixed objective data with its original values, and the
declared constant `d` is a detached clone. No parameter derivative
is requested in this correction stage.
The archived full SPD Hessian at the seed implies its field principal
block is SPD **at that seed**. This is not a certificate at later
points; retain PCG curvature and independently recomputed residual
checks on every reduced solve.

The production face fluxes depend on the fixed flow controls, not the
twenty field controls. Thus this phase cannot cross the `q_y[2,0]`
surface or shrink its face margin. Verify that the measured face
margin remains exactly the seed value at every admitted trace;
limiter choices and slope margins can still change and must be checked.

Use at most eight reduced Newton corrections, sixteen backtracks per
correction and eighty PCG iterations per solve. Preserve reduced
stationarity `<1e-10` and true linear relative residual `<=1e-10`.
Require each candidate's complete 54-stage strict branch, positive
slope/face margins and the same full signature as the frozen seed.
The existing normalized Armijo condition on reduced gradient merit
must pass. Also require actual `J` not to increase beyond
`128*eps64*max(|J_current|,|J_candidate|,tiny64)`; this permits
roundoff-level objective changes at a small reduced residual, not a
new statistical objective or a relaxed gradient gate.
Matching endpoint signatures do not certify the entire connecting
step segment; no smooth-path certificate is issued.

Store the seed, each accepted field control and the last accepted
endpoint, fixed-dynamics hash, complete signature digest/margins,
J, field/full gradient norms, true linear residuals and iteration/HVP
cost. Independently recheck the final reduced gradient and full
26-gradient; verify the six dynamic controls are byte-identical to
the seed. A successful field stage is a **block-stationary candidate**,
not a full root. Report dynamic gradient components and require both
final branch margins `>1e-4` for a response-margin-supported handoff.
Do not call a full root solver or issue a sensitivity in this attempt.
On numerical refusal retain the last accepted metrics and reason;
unexpected callback/programming errors fail execution.

Run one serial child with a 300-second wall trigger and sampled
1-GiB RSS trigger using the pinned shared resource runner. Require
caller-supplied reviewed probe SHA256 and fixed plan hash before
launch, then compare source, raw archive, fixed inputs and parameters
before/after. Write only a fresh attempt directory outside all old
archives. Separate execution completion, reduced stationarity, full
stationarity and response eligibility. Do not try a second seed or
increase the iteration/resource limits after observing the result.

This tests whether correcting the measured field-dominated residual
at fixed dynamics provides a better handoff for a later full nominal
method. It does not establish a smooth interior full root, global
convergence, root absence, physical skill, or an implicit response.
R4-R remains open until all of its original final-point and signed
response gates pass.
