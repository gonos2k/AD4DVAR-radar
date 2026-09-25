# R2 branch-conditioned root-only diagnostic after fixed-margin refusal

The prior locked-terminal Newton child report SHA256 is
`aff2a5e1374888b190c783f30bd7b8a468a0cc8bdb67a2cb562603308021d4f9`.
It recorded a locally SPD seed Hessian and four converged PCG solves; at
Newton iteration 4 all 16 candidates were rejected before Armijo merit
evaluation (nine full-signature mismatches and seven combined
stage/slope/face-margin refusals). These first-failure counts do **not**
show that the seven margin-refused candidates have the pinned signature.
The prior plan and report remain unchanged and refused.

This is a **new one-start, root-only policy**, using exactly the same
attempt-2 terminal FP64 control (SHA256
`8f5110f2755217dece8ca9f6bbab23ceadb184b4f1a32b8e4654ddfde5f4b258`),
same full 54-stage signature (digest
`0783f66a062d5144894513c65a4e221befca4d23a22d8fba0adc43ead5011327`),
unchanged correlated point objective, p, C, background, boundaries, frozen
verification, exact HVP and `refine_stationary` implementation. There is
no alternate seed, Hessian shift, changed data/prior, or altered Newton
8-step /16-backtrack /104-PCG budget, 1e-10 true linear residual,
gradient or SPD criterion.

Only the **Newton candidate admission rule** changes for this new run.
Each candidate must pass the existing `inspect_branches` pointwise
roundoff-aware oracle: all 54 RK stages, resolved minmod input signs and
active left/right comparisons at its `128*eps64` scaled threshold,
resolved nonzero face fluxes, finite measured scaled slope and face-flux
margins `>0`, and the **same complete limiter choices and face signs** as
the locked seed. This is not simply a bare `margin>0` replacement for the
oracle. A different signature or an unresolved tie/face is rejected.
The refiner still evaluates the actual candidate gradient-merit and its
existing Newton-slope Armijo condition; no old-branch Newton step is
accepted on a different signature. This is the refiner's measured-merit
filter, **not** a smooth line-search convergence guarantee if the segment
between two admitted endpoints crosses a kink. Root classification rests
on the fresh final endpoint gradient, branch and Hessian checks.

For each trial record control values/hash, actual core-trace outcome,
signature digest if available, measured slope/face margins, whether the
full signature matches, and finite actual objective/gradient where the
core tracer succeeds. In particular, for a core-valid but mismatched
signature, compute objective/gradient **for diagnostics before returning a
branch refusal**; the unchanged refiner itself will not evaluate that
candidate. Keep the diagnostic and refiner evaluations distinct in the
record. Flush each trial atomically so a timeout retains the last completed
candidate record. These are candidate diagnostics, not evidence that
the segment from the previous iterate stayed on one smooth branch.
Retain PCG iterations/HVP calls and the refiner's actual residual/refusal
message. A branch, PCG, nonfinite or Armijo failure is `root_refused`.

Before Newton, repeat the exact 26-column seed Hessian finite/symmetry/SPD
gate. If the final control reaches `||grad_c J||_inf<1e-10`, repeat the
exact Hessian SPD gate, then perform a **fresh final** 54-stage branch
trace and measure both margins. No adjoint runs in this child.
Classify that result **separately from response eligibility**:

- `core_strict_root_margin_qualified` only if both final scaled
  slope/face margins are `>1e-4` and finite;
- `core_strict_root_low_margin` if a pointwise stationary/SPD root is found
  but either margin is `<=1e-4`. It remains response-ineligible.

Even the first status is **not** adjoint or signed-reanalysis validation;
`response_validation=not_performed` in this entire run. Endpoint-equal
signatures cannot certify an unobserved Newton segment or stationary
solution path, and no global minimum or weather skill is claimed.

One new serial child has a 600-second wall and sampled 1-GiB RSS cap. The
existing `run_guarded` runner monitors the child PID with 250-ms `ps` RSS
samples, terminates its process group at the wall deadline or observed RSS
excess, and records exit/termination/monitor errors. The sampled RSS is
neither a hard allocation cap nor a bound on parent/descendant memory or
between-sample spikes. The parent requires exact preflight, attempt-2,
prior-terminal-report, runner/probe/tracer/objective/transport/refiner
source and input/plan hashes before and after, matching child PID, observed
elapsed/RSS and status-specific root evidence. On timeout/resource kill,
the external execution status is `failed` and any last atomic child
checkpoint remains partial, never a root. A finished child with explicit
`seed_curvature_refused` or `root_refused` is a completed numerical refusal;
the two root statuses above require final gradient/Hessian/branch evidence.
An identity mismatch is failed publication even if a partial numerical
record exists. If the run refuses, no threshold or signature policy is
changed within it.
