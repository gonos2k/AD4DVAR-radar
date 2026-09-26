# Fixed point-observation three-hour branch: first refusal located

One planned, guarded run located the first strict minmod predicate that
rejects the unchanged PR #204 4×5 point-observation input/control. The
child and parent completed with exit code 0 and stable source, input,
plan and archive checks. The result is a **diagnostic refusal**, not an
execution failure, stationary root, sensitivity or proof that no
supported long-horizon branch exists.

The first failing callback is global stage **0**, before any stage is
admitted by the strict oracle. This is the first SSPRK Euler stage of
substep 0 in the first 0–600-second **analysis replay** interval of
`forecast()`, not a future lead. At the zero-based y-face-array index
`q_y[3,1]`, the absolute face volume flux is
`5.551115123125783e-17`, below the source-defined strict threshold
`4.5474735088646444e-15`. The ratio is about 0.0122. The diagnostic
found **one violating entry in one predicate** at this stage; its
field-scale, interior slope and remaining face-flux predicates passed.
This explains the prior generic `minmod joint oracle left its strict
smooth branch` refusal at the fixed control. It does not determine
whether this nearly zero face flux is structural for all admissible
controls or can move away from zero at another control.

The child profile and its 26-control/13-parameter identity match the
archived PR #204 input and R5 response-policy record. The earlier
3-hour forward result remains a same-operator forward check. In this
new run, `make_case()` constructs that fixture, including a full
20-interval/3,600-Euler-stage synthetic truth trajectory, before the
diagnostic forecast starts; the diagnostic `forecast()` itself stops at
callback 0. The reported runtime and RSS include that fixture work.
No GN, Newton,
HVP, adjoint, full VJP or signed nonlinear reanalysis was run.

The guarded child PID was **94612** and exited 0. The resource guard
recorded **3.755 seconds** elapsed; the raw child recorded **2.692
seconds** for its own calculation. Fourteen child-RSS samples measured a peak of
**239,304,704 bytes**, below the sampled 768 MiB limit; no wall/RSS
termination or monitor error occurred. The 180-second wall value is a
guard trigger with termination/reap grace, not a hard elapsed bound.
The raw child finished with `diagnostic_status=refused_first_branch`,
`stationarity_passed=not_tested`, `response_computed=false`, and
`physical_validation=not_performed`; the parent kept the same
distinction.

Focused classifier/guard tests passed **14/14**, and the affected
suite passed **46/46** with 18 existing TorchScript deprecation
warnings. Error-level basedpyright 1.39.9 reported 0 errors, warnings
or notes. GREEN and RED approved the prelaunch math/provenance/guard
contract. Graphify's code graph was incrementally refreshed after the
final helper removal; `graph.json` has 7,898 nodes and 206,277 edges.
The no-cluster update did not refresh the older `GRAPH_REPORT.md`
counts. Exact commands and file hashes are in
`FV_POINT_3H_FIRST_BRANCH_EVIDENCE.json`.

This closes the **first-refusal-cause diagnostic only**. R5-R-R stays
open: another declared control/search policy would need a strictly
smooth final long-horizon branch, gradient below `1e-10`, appropriate
curvature and true linear residuals, full 13-component response and
two adjacent signed reanalysis sizes. The current first-face finding
does not justify relaxing the nonzero-flux threshold.
