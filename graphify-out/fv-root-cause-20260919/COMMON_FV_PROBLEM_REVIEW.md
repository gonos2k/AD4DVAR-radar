# PR174 follow-up: execution status and common research problem

Baseline: PR174 merged as 0956709a. Its 86-control execution milestone is
closed; original child/resource reports and the original summary are preserved.
This change does not repeat GN, refinement, PCG response or reanalysis runs.

## Separate execution and numerical meaning

The old summarizer ignored a nonzero exit code when an `eligible` checkpoint
remained. The replacement classification reports execution_status,
numerical_status and response_validation separately. Completion requires an
integer zero exit code, finished phase, completed source-identity check and no
resource-monitor termination. A saved numerical result can remain eligible
while the process is failed; it is not counted as a completed eligible run.
Wall/RSS caps differ from monitor failures. Response validation remains
`not_performed` for the archived two experiments, independent of algebraic
JVP/VJP projection agreement.

The revised summarizer defaults to a new status-summary file and retains the
raw reports, source snapshots and old published summary. No old success record
is overwritten to manufacture new evidence.

## One thin definition, existing model and numerical policy

`advar.fv_research_problem.FVResearchProblem` references the existing
AnalysisObservations, FrozenOuterState, FVAnalysisTransport and future boundary
schedules. It exposes objective, forecast/score, pointwise branch checker,
layout, support description and fixed-input identity. Numerical tolerances,
GN, Newton-PCG refinement and adjoint/VJP implementations are unchanged.

The old 4x5 `make_research_functions` and 8x10 `functions` remain callable and
now delegate to this definition; their fixed-profile guards remain. Synthetic
field/observation generation stays in the two fixtures. No third fixed case,
new physical model, generalized covariance or resampling operator was added.

Layout derives field controls from active_field_index, flow controls from the
transport basis count, parameter length from observation values plus the one
mean-background coefficient, and stage count from the regular observation/
forecast schedule and substeps. It does not derive observation count from the
control count. The current supported observation operator is explicitly
`collocated_dbz`; arbitrary observation positions, missing frames, partial
observations and unknown boundary contributions remain unsupported here.

The error transform is the existing frozen product whitener, not a new claim
about arbitrary covariance support. Time is three regularly spaced observations
and one following forecast interval, not irregular-time generality. The two
measured profiles have 0/60/120-second observations and 180-second verification.
The injected strict tracer still has its positive-growth/nonzero-flow/limiter
restrictions and remains serial. Construction alone does not certify a point's
normality, minimum, sensitivity or forecast skill.

Identity fingerprints fixed model/observation inputs, layouts and fixed score;
it is not a cache certificate. Runtime parameters/control, code and branch
identity must also be bound by any caller implementing cache reuse. Producer
source lists now include the shared module; historical measurement sources
remain separate.

## Compatibility evidence

Tests use immutable pre-refactor source: the 4x5 adapter at 0956709 and the
already archived measured 8x10 case. These snapshots are repository fixtures;
no git history/network is required when the test runs. The new adapters are
not used as their own reference. For each preserved nominal point, tests
compare J, E, full control/parameter gradients and one sinusoidal control HVP,
plus every stage's selector/face signature. Parameter and fixed verification
identities are checked against the archived experiment, and the 8x10 fixture
values are compared directly with the old generator.

The first strengthened 4x5 fixture test queried a verification-hash key from a
report that did not contain it (six fixture errors); it was corrected to use
the existing parameter-VJP report's verification identity. This was a test
setup issue, not a changed numerical result. Final evidence follows below.

Supported/refused cases keep their meaning: full detected observations and
known support are accepted; missing observations, unknown boundary support,
unsupported observation operators and changed branch signatures are rejected.
These are interface-compatibility checks, not independent FV response accuracy.


## Final evidence and limits

- **158 unique affected tests passed**, 18 existing TorchScript warnings;
  `common_problem_tests.log`. This includes 14 common-problem comparisons/
  refusals, 15 status tests, existing numerical/interface tests and publication
  guards. Repeated development runs are not added to this count.
- Pinned basedpyright: 0 errors, 0 warnings, 0 notes for the shared module and
  affected typed research scripts (`common_problem_typecheck.log`).
- New `fv86_execution_status_summary.json` reinterprets the original reports:
  2 completed executions, 2 numerically eligible results, 2 completed eligible
  results, response validation `not_performed`. No optimizer was invoked;
  two small objective-only seed-cost evaluations were retained by the summary.
- Early prepare cutoffs can be summarized without seed/identity data; their
  numerical status is not_reached and no seed objective is evaluated. Unknown
  resource terminations and monitor failures are not successful executions.
  Saved numerical eligibility remains visible after a nonzero process exit.
- The fixed 26/86-control profile checks remain in the legacy wrappers even
  though the shared object's layout is derived from actual inputs. A new
  regression checks altered active-field counts are still rejected there.
- Producer provenance includes the shared module for future executions.
  Existing raw measurements and their source snapshots/hashes are untouched.
- The existing public donor-cell restriction, stationarity and residual gates,
  physical transport, observation weighting and verification fields are unchanged.

The per-profile parity checks found zero numerical differences in J, E, their
full partial gradients and the tested HVP. This is not a fresh whole-response
solve or independent reanalysis verification. The common definition does not
claim arbitrary observation geometry, general correlated/missing errors,
irregular time, unknown boundaries, general limiter regularity or learned priors.

The next independently budgeted proposal is `FV86_REANALYSIS_PLAN.md`: a
middle-time bias tangent/predictor and endpoint corrections, with a 20-minute /
2-GiB total cap. It has not run and is not covered by the prior two-start budget.

Final GREEN/RED reviews found no remaining blocker. GREEN verified the new
summary's canonical hash, source/artifact identities and preservation of all
raw historical measurements. RED checked profile guards, compatibility test
independence and the stated support/validation boundaries. Code-only Graphify
refresh covered 280 files (6813 nodes, 65346 edges, 242 communities); no semantic
LLM extraction was used. Full CPU/package regression remains unrun.

Aside desktop DOM and screenshot (`common_problem_desktop.png`) confirm the
separate execution/numerical/reanalysis columns and the review link. Original
animation frame-data hashes are unchanged. No new mobile check was performed.

Subsequent approved execution: the proposed local 86-control reanalysis was
performed and passed two local sizes for the seed-A middle-time direction. See
FV86_REANALYSIS_RESULTS.md and the scoped closures in PR174_REVIEW_RESOLUTION.md.
