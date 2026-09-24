# 86-control FV discrete execution

Baseline: PR173 merged as d5dcf94. Resource approval: user “go”, 2026-09-24
12:37 JST. This follows FV_86_CONTROL_EXECUTION_PLAN.md; numerical gates,
physical model and product sum objective are unchanged.

## Problem and limits

The explicit case has 8×10 cells on the old 40×50 spatial-unit domain,
spacing (5,5), five physical-coordinate flow basis coefficients, one growth
coefficient, 86 controls, 241 parameters, and 108 RK Euler stages. Truth uses
three 60-second intervals, 18 substeps per interval, boundary pairs at both RK
stages, and the same declared interval growth (not repeated per substep).
Verification is the fixed 180-second synthetic truth plus 0.1 times the fixed
pattern, constructed before either GN run. This is not independent weather
truth or a grid-independent statistical inverse problem.

Two starts are fixed in advance: the declared six dynamics controls and half
those values, with all initial-field controls zero. Each GN endpoint must pass
its own strict branch checker. Refinement then preserves that branch; the two
starts are not forced to share a branch or local solution. The state and
boundary are completely known; detected observations and positive growth are
still required. The generic public donor-cell restriction is unchanged.

The conservative coefficient-box CFL bound is 0.19150000000005407. The actual
first-seed CFL is 0.0013099045428013308. Actual endpoint CFL uses the production
positive-outflow formula on the controlled face fluxes; coefficients are
constant in time in this model, so this is shared by its substeps.

## Preflight (no GN or response)

The capped cost check completed in 5.2375 seconds (child 4.0416 seconds), with
sampled maximum RSS 337,199,104 bytes. Gradient cost was 0.9380 seconds and
HVP costs were 1.5081/1.4015 seconds for two predetermined unit directions.
Both HVPs were finite; all 108 stages passed the strict seed branch check.
Sampled Rayleigh values are directional diagnostics, not a condition-number
estimate or a whole-Hessian positive-definiteness certificate.

These measured HVP costs suggest about 2 minutes for 80 HVPs or 6.5 minutes
for approximately 264 HVPs, plus GN, branch checks and other evaluations.
Those counts are illustrative, not convergence predictions; actual iteration
counts and the 30-minute per-start cap govern execution.

The runner's shutdown-race handler was hardened after preflight and before
seed A. Preflight's recorded runner hash is consequently historical. The
numerical producer and case were unchanged, so the cost check was not repeated.
Each child checks its selected source files before and after its own run.

## Resource and evidence policy

Each child is launched by `.venv/bin/python`. The parent samples that serial
Python process RSS via `ps` every 0.25 seconds and terminates the process group
at the wall/RSS limit. This is a sampled 2 GiB guard, not an OS hard allocation
limit or a measurement of hypothetical future subprocesses. Approved caps:
120 seconds for preflight, 1800 seconds for each of the two starts.

Child JSON is atomically checkpointed at phase boundaries and periodically
inside correction/response PCG. A timeout can leave a historical `running`
checkpoint; the corresponding `.resource.json` supplies final termination.
Raw checkpoints are not rewritten into fictitious successful outputs.

Full nominal stage selectors/face signs are stored once per admitted GN
endpoint. Subsequent branch checks compare all entries but store compact
hashes/margins and control hashes. Seed/GN/refined endpoint controls remain
available. Rayleigh ranges sampled by PCG are not spectral condition numbers.

No perturbed-observation reanalysis is scheduled. Directional JVP versus whole
VJP projection checks, if reached, are algebraic checks only; independently
validated response fraction is unmeasured. No success-only selection is used.

## Final results: both predeclared inputs

| Quantity | Seed A | Seed B |
|---|---:|---:|
| Supplied seed J | 6.316235045 | 9.894661067 |
| Reference zero-control J | 14.29067422 | 14.29067422 |
| GN max gradient | 2.937572e-03 | 2.941639e-03 |
| Final max gradient | 1.010832e-11 | 8.585585e-12 |
| Final J | 0.0591580038374185 | 0.0591580038374182 |
| True adjoint relative residual | 8.243789e-12 | 8.235807e-12 |
| Refinement Newton iterations | 2 | 2 |
| Refinement HVP calls | 76 | 76 |
| Child wall seconds | 261.310 | 259.045 |
| Sampled peak RSS bytes | 346800128 | 344653824 |

Both GN calls stopped with `maximum_outer_iterations` after four outer steps;
GN PCG totals were 106 and 100. The later explicit refinements reached the
strict max-gradient gate. Each used two Newton steps, each with 36 PCG
iterations, 37 PCG HVP calls including its actual residual check, plus one
additional independent refiner HVP. Thus refinement used 76 HVPs per case.
The response used 36 PCG iterations / 37 HVPs, plus its separate VJP residual
check and parameter VJP. No dense Hessian was constructed.

The reported GN `initial_objective` is the product solver's zero-control
reference cost, not the caller-supplied seed cost. `summarize_fv86_execution.py`
performs two objective-only evaluations at the exact stored seed vectors to
report their separate costs. These postprocessing evaluations are outside the
measured child times; they are not new GN or reanalysis runs. The product
solver's reference fallback and initial candidate exploration remain active;
the seeds are its input arguments, not a claim about the first internal GN
iterate, which was not captured.

Each refinement preserves its GN endpoint's selectors and face signs across
all 108 RK stages. Both starts also happened to select the same branch.
Refined control distance between starts is 4.3855767e-14; full sensitivity
relative difference is 6.9634042e-12; score difference is 3.5670251e-17.
These are consistency diagnostics for two starts, not proof of uniqueness,
state accuracy or observational identifiability. The weak-curvature and
singularity qualifications in the review remain applicable.

Eligible response count is **2/2**. Observed refusals by support, nonfinite
initial point, branch, curvature, linear budget, stationarity and resources
are all zero for these two predefined cases. The independently validated
response fraction is **unmeasured**, not 100%: no perturbed-observation
reanalysis was run. Whole-VJP/directional-JVP projection discrepancies are at
most 6.68e-17 (A) / 1.72e-16 (B); they are algebraic checks.

| Measured interval | Seed A seconds | Seed B seconds |
|---|---:|---:|
| GN | 74.197 | 70.108 |
| Assessment | 0.564 | 0.566 |
| Refinement | 123.838 | 124.113 |
| Response | 61.232 | 62.728 |

No timeout/RSS termination occurred. The measured resource records cover the
entire child process, including its evidence checkpoints. Costs are from this
8×10 problem and do not establish a scaling law or a controlled speedup.

## Provenance, code checks and remaining limits

Both child runs have identical problem/input and measured-source identities,
and their start/end source checks passed. After measurement, module-loader
None checking and typing annotations were added to the producer; the case
received precise boundary tuple annotations and an identity `typing.cast`.
These do not alter the numerical path. Exact measured sources are retained as
`fv86_measured_producer.py.txt` and `fv86_measured_case.py.txt`, whose hashes
match the raw reports. `fv86_preflight_runner.py.txt` matches the preflight's
pre-race-fix runner. Source changes were not made during either measured run.

The final four research scripts pass pinned basedpyright (0 errors/warnings/
notes). Initial type checks exposed the loader/heterogeneous-report and tuple
annotation issues; those were corrected after preserving measured sources.
Only cheap affected checks were rerun after typing changes; the FV experiment
was not repeated for annotations.

The case's initial echo uses the declared lower-left sampling, not exact cell
averages; the product summed likelihood/prior weights remain unchanged as the
number of state/observation cells grows. This is a larger discrete execution
milestone, not a common physical/statistical inverse-grid convergence study.
No full-spectrum condition number, full SPD proof, field-conditioned
observational information analysis, missing-observation extension, zero/negative
growth extension, unknown boundary support, finite-impact response validation,
typed learning, real-weather skill or whole-workload D7 is established here.


Final checks: **128 unique affected tests passed**, 18 existing torch.jit.script
warnings (`fv86_final_tests.log`); four research scripts have 0 errors/warnings/
notes (`fv86_typecheck.log`). Tests include actual fixture/time/branch contracts,
resource cutoff and clean-exit cases, failure classification, and summary/raw
artifact mutation rejection. The full CPU/package suite was not run. GREEN and
RED final reviews found no remaining blocker. Code-only Graphify refresh: 277
files, 6769 nodes, 65213 edges, 209 communities; no semantic LLM extraction.

Directional responses (same direction units in both starts; no cross-unit importance comparison):

| Direction | Seed A | Seed B | Relative difference |
|---|---:|---:|---:|
| observation_sine240 | -2.060536691021e-03 | -2.060536691043e-03 | 1.0923e-11 |
| theta | 2.345850140459e-07 | 2.345850136122e-07 | 1.8487e-09 |
| middle_time_bias | -2.997839116189e-03 | -2.997839116184e-03 | 1.5190e-12 |

The original demo's separate 86-control panel was checked in Aside desktop DOM
and screenshot (`fv86_desktop.png`). No new mobile check was performed. The
publisher confirmed unchanged original animation frame-data hashes for both
template and local index. The new summary and its referenced child/resource
artifacts are pinned and mutation-tested before display.
