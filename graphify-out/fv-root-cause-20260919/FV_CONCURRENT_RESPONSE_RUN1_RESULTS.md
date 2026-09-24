# G9 in-process concurrent FV response: bounded refusal

The predeclared 4×5/8×10 two-thread run in `FV_CONCURRENT_RESPONSE_PLAN.md`
was executed once under a 300-second wall / sampled 1.5-GiB RSS guard. The
external runner returned **exit code 1**, with no resource termination or
monitor error, after **6.211 seconds** and sampled peak RSS **348,700,672
bytes**. Its source hashes before and after the child were equal. The child
report is `execution_error`; neither worker produced a qualified response,
and no adjoint total or overlap-success claim was issued. Raw child, runner,
resource and log records are preserved in `concurrent_response_attempt1/`.

The failure occurred during a PCG Hessian–vector product inside
`torch.func.jvp(grad(objective))`, through the existing recompute autograd
function. The runtime error was:

```text
Trying to access a forward AD level with an invalid index.
This index was either not created or is already deleted.
```

This is not evidence of a wrong FV derivative, a missing stationary root,
or a resource shortage. The two archived controls still passed fresh
preflight branch and stationarity checks at 54/108 stages, with maximum
control-gradient components `6.96204e-12` and `1.01083e-11`. A separate
**serial** current-source 4×5 response completed in about 20.49 seconds,
returned `-0.003038272528696361` for the middle-time direction (the archived
same-problem value), and had true relative adjoint residual
`9.88816e-11`. That check did not share a forward-AD level with another
thread.

The minimal independent `fv_forward_ad_thread_probe.py` contains no FV model
or observer. It forces two threads to enter `torch.autograd.forward_ad`
`dual_level()` together under local PyTorch 2.13.0. One received
`RuntimeError: Nested forward mode AD is not supported at the moment`; the
other timed out at the rendezvous with `BrokenBarrierError`. This supports
the inference that simultaneous forward-AD level use, rather than the
new diagnostic observer's tensor arithmetic, is the immediate concurrency
boundary in this environment. It does not isolate every nested PyTorch
implementation detail of the FV traceback. PyTorch documents a single
supported forward-AD dual level in its
[dual_level reference](https://docs.pytorch.org/docs/main/generated/torch.autograd.forward_ad.dual_level.html).

The current supported execution rule is therefore **one forward-AD FV
response per process at a time**, with separate OS processes if cases must
run concurrently. `ContextVar` observers solve report routing, but they do
not make `torch.func`/recompute safe for two simultaneous thread-level
response solves. No lock or threshold change was introduced to hide the
failure; in-process parallel FV response remains unqualified. The subsequent
process-isolated attempt is recorded separately in
`FV_PROCESS_ISOLATED_RESPONSE_RESULTS.md`; its success does not change this
failed thread-level finding.

This is a numerical execution-domain finding, not a physical forecast or
independent response validation. The failed child performed no new GN,
stationary refinement or signed reanalysis. Its source snapshots preserve
the exact uncommitted producer/runner bytes used for run 1; historical
FV86/4×5 records are untouched.
