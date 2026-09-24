# G9 process-isolated concurrent FV response plan

Run 1's two-thread 4×5/8×10 response refused with a forward-AD level error
before either response was published. Its raw report, resource record, log,
producer/runner source snapshots and independent tiny dual-level failure
remain in `concurrent_response_attempt1/` and
`FV_CONCURRENT_RESPONSE_RUN1_RESULTS.md`. No numerical tolerance or strict
branch gate is relaxed for this follow-up.

The same two archived controls, fixed problems, verification fields and
middle-time +1 dBZ directions from `FV_CONCURRENT_RESPONSE_PLAN.md` will be
computed in **separate OS child processes**. The parent launches the two
guarded children concurrently, each using the current ContextVar minmod
stage and PCG diagnostic observers only within its own process. This avoids
overlapping PyTorch forward-AD levels within one process; it is not an
in-process thread-safety claim.

Each child must independently pass:

- exact 54/108 archived pointwise minmod choices and face signs, with the
  outer stage observer count matching its own branch stages;
- nominal max gradient `<1e-10`, true relative transpose-adjoint residual
  `<=1e-10`, one attributed PCG solve with HVP count, iteration count and
  reported residual matching `LocalResponse`;
- total middle-time directional response within relative `1e-6` of the
  pinned same-problem 4×5 or 8×10 value;
- complete before/after fixed input and current source SHA-256 equality,
  with archived reports pinned separately by their original SHA-256.

Each child receives an independent **300-second wall cap** and **768-MiB
sampled RSS cap**. The parent requires at least one RSS sample, the exact
reported cap values and the monitored child PID to equal the worker PID.
The two sampled child caps sum to 1.5 GiB; sampling does
not provide an OS hard allocation limit or include parent overhead.
The parent requires two zero exits, no resource/monitor termination,
finished phases, stable source identities, complete response records, and
positive overlap between child response intervals **and at least one pair of
actual PCG HVP operator-call intervals** from the two workers. It preserves each
refusal and partial report. External child exit/resource state remains
separate from local numerical eligibility.

This is one bounded process-isolation execution check, not an independent
nonlinear reanalysis, a finite minmod path certificate, arbitrary-case
response rate, or physical forecast verification. If it succeeds, it
establishes a supported **process-level** concurrent research execution
pattern in this environment while in-process simultaneous forward-AD FV
responses remain explicitly unsupported.
