# PR171 follow-up: fixed-parameter isolation and current GN integration

Baseline: merged PR171 `6354ecc7`. Historical reports remain unchanged.

## Boundary correction

The explicit refiner receives an independent detached parameter copy. Its only
permitted output change is the control vector. If it modifies its parameter copy,
the workflow refuses response publication; if it raises, caller parameters remain
unchanged. Candidate controls must preserve shape, dtype, device and finiteness.
Small regressions distinguish normal refinement, mutation followed by success,
mutation followed by failure, and malformed candidate layouts.

The detach applies only at the external numerical root-refiner boundary. The
actual objective, score and response still use the original parameter tensor,
so full VJP differentiation remains intact. This is not a sandbox for arbitrary
closure/global state mutation; such side effects remain unsupported.

## Current result forwarding

`fv_gn_response_probe.py --fresh-gn` executes current `solve_analysis` with the
same declared six-component dynamic seed and fixed synthetic case. It passes the
returned object's `control` directly to `prepare_response`, not an archived GN
control. The old oracle is used only to define the already declared, frozen
verification field and for post-run comparisons; it does not initialize GN.

The workflow explicitly uses the existing small dense oracle if the actual GN
gradient misses the unchanged 1e-10 gate, then computes the whole parameter
response via matrix-free adjoint. GN, assessments, refinement, response and total
producer costs are measured in one process. No perturbed reanalysis is run.

Relevant source hashes are captured before numerical work and checked unchanged
before writing the report. This prevents publication when those files change
during execution. It does not certify all transitive/environment dependencies.

This is current execution on the same controlled synthetic case, not a new
meteorological case or arbitrary-input GN convergence proof. Refinement retains
the 32-control dense-oracle bound. General minmod FSOI, typed-prior learning,
finite impact/path and whole-chain D7 across general workloads remain open.

In fresh mode the legacy identity key `gn_report_sha256` identifies the archived
reference report only. Current analysis provenance is `analysis_origin`,
`analysis_control_sha256` and the newly produced `gn_result`; the archived digest
does not claim that the current GN output was loaded from that file.

## Measured outcome

- One current GN run; 4 outer iterations / 65 PCG iterations, ending
  `step_tolerance_unverified`. Its returned control is passed directly onward.
- Before/after gradient: 1.334876035182e-04 -> 6.962040319247e-12.
- Actual adjoint relative residual: 9.888163804832e-11.
- Refined control, full gradient and all three directional responses match
  the existing reference exactly in this environment. All 54-stage choices
  and face signs remain in the declared branch.
- GN 27.635s; assessment 0.276s;
  refinement 61.579s; response 23.374s;
  full producer 113.054s. These are measured in one process,
  not a sum of historical runs. No peak-memory measurement was made.

83 distinct focused tests passed (18 existing deprecation warnings); pinned
basedpyright: 0 errors/warnings/notes. GREEN/RED final reviews found no blocker.
Full CPU/package CI was not requested. Graphify code-only update: 6,688 nodes,
65,079 edges, 247 communities, no semantic LLM extraction.

Aside desktop inspection confirmed the current-GN result and stage costs in the
original demo. Screenshot: `current_gn_desktop.png`; original animation arrays
are unchanged. No new mobile check is claimed.
