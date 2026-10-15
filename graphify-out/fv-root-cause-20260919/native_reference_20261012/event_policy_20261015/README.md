# Evidence reproduction

Start from repository root with the shared interpreter:

```sh
.venv/bin/python graphify-out/fv-root-cause-20260919/native_reference_20261012/event_policy_20261015/verify_saved.py
```

This checks the frozen source/input provenance and saved arithmetic only. It does not rerun FV or solve the original optimization problem. `TRACES.tar.gz` contains all311 immutable raw operand traces at their recorded repo-relative paths. The verifier reads the archive directly; to inspect individual NPZ files, extract with `tar -xzf graphify-out/fv-root-cause-20260919/native_reference_20261012/event_policy_20261015/TRACES.tar.gz` from root. Loose original traces are preserved locally, while the PR includes the complete archive instead of311 binary diff entries.

`attempt1/PLAN.json`, its SHA pin, `source.tar.gz`, and `RESULT.json` are the execution authority. Re-running the runner refuses its existing attempt path; a subsequent authorized phase requires a unique prefix, new frozen plan and cumulative ledger, not deletion/reset. `ACCOUNTING.json` carries905 tracked directJ calls and490 gradients; complete older Hessian/AD/wall history is unavailable.

Endpoint9c106af4… has current47gap+123right-zero. Both outgoing sides are positive, but chart hullInf1.1755e-7 exceeds1e-10. It is not a minimum and is not eligible for P2 or the response callbacks. `RESPONSE_READINESS.md` defines the event-conditioned eligibility and KKT route. The old two-face-only response is not invoked on this endpoint.
