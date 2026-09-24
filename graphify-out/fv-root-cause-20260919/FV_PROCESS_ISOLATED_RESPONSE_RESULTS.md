# G9 process-isolated concurrent FV response: bounded success

The follow-up in `FV_PROCESS_ISOLATED_RESPONSE_PLAN.md` ran the same fixed
4×5 and 8×10 local-response problems as failed in-process thread run 1,
but launched **one response per separate Python child process**. The parent
used two independent 300-second / sampled 768-MiB child guards. Both child
processes exited 0 with no resource or monitor termination, both completed
their local numerical and source/input gates, and the parent returned
`execution_status=completed` after **63.198 seconds**. The source and four
pinned archived-report hashes were identical before and after both runs.

| Measured item | 4×5 / 26 controls | 8×10 / 86 controls |
|---|---:|---:|
| Child PID / parent-observed PID | 23458 / 23458 | 23457 / 23457 |
| Child process time | 24.638 s | 63.193 s |
| Sampled maximum child RSS | 345,276,416 B | 345,686,016 B |
| Strict branch / outer observer stages | 54 / 54 | 108 / 108 |
| Final maximum control-gradient component | 6.96204e-12 | 1.01083e-11 |
| True relative transpose-adjoint residual | 9.88816e-11 | 8.24379e-12 |
| PCG iterations / observed HVP calls | 25 / 26 | 36 / 37 |
| Middle-time +1 dBZ total response | −0.003038272528696361 | −0.0029978391161887084 |
| Same-problem archived total difference | 0 recorded | 0 recorded |

The response intervals overlapped for **23.398 seconds**. The parent found
an intersection of **0.876 seconds** between an actual HVP operator-call
interval in each child; it did not infer overlap merely from first/last
event timestamps. The sum of the two *separately sampled* peak child RSS
values is **690,962,432 B**. This excludes the parent process and can miss
between-sample spikes, so it is not an OS hard memory bound or a controlled
speedup measurement.

Each worker matched the complete archived pointwise branch choices and
face signs, its exact stage count, the named middle-time direction and fixed
verification field. Its PCG monitor HVP calls, iterations and reported
residual equal the returned `LocalResponse` diagnostics. Each worker and the
parent rechecked current source identities; each worker rechecked the fixed
input and archive identities. The parent also matched its guarded PID and
exact command to the worker report. All raw child reports, resource reports,
logs, parent verdict and exact producer/runner source snapshots are preserved
in `process_response_attempt2/`. Those source snapshot hashes match the
parent's prelaunch source map.

This result establishes a **bounded process-level concurrent research
execution pattern for these two fixed synthetic FV cases**. It does not
make simultaneous in-process thread-level forward AD safe: run 1 refused
with an invalid forward-AD level, and the independent tiny two-thread
`dual_level` check also refused. It also does not repeat GN/refinement,
perform signed nonlinear reanalysis, establish a finite minmod path,
measure independent radar forecast skill, or generalize to other grids and
observation conditions. The matching archived local derivatives are a
regression to prior evidence, not a new independent validation.

Local PyTorch 2.13.0 was used. PyTorch's
[forward-AD dual-level reference](https://docs.pytorch.org/docs/main/generated/torch.autograd.forward_ad.dual_level.html)
documents that only one level is supported. The observed two-thread errors
are consistent with that constraint; the exact internal thread failure mode
is an inference from these runs, not a universal theorem about PyTorch.

Source checking detects persistent file edits, not a transient edit and
revert between import and hashing. The per-child RSS guard samples one PID
every 250 ms and does not include parent overhead. These limitations remain
in the reported supported domain.

The focused producer, runner and resource-guard suite passed **36 tests**.
Pinned basedpyright 1.39.9 reported **0 errors, 0 warnings, 0 notes** for the
changed scripts and tests. Graphify's code-only incremental update reported
7,137 nodes, 65,990 edges and 266 communities. The attempt-2 manifest binds
the exact raw reports, execution-time source snapshots, current scripts,
focused check logs and this report; the execution-time snapshots remain
separate from later type-only source edits. No full FV regression or new GN
solve was performed for this concurrency check.
