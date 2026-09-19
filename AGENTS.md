# ADVAR Working Principles

## Team and Environment

- Actively delegate bounded, independent tasks to small teams of `gpt-5.6-luna`
  (`lunar`) agents with `high` reasoning to reduce token usage. Reuse existing
  evidence, avoid duplicate work, and report concise findings with source
  locations. The main agent integrates results and makes decisions.
- Separate team reviews into GREEN and RED roles: GREEN checks correctness and
  supporting evidence; RED looks for counterexamples, failure modes, and gaps.
- Before ending each session, have the GREEN and RED subagent teams review the
  session's changes and verification evidence. Resolve actionable findings or
  record remaining issues explicitly before the final handoff.
- Do not recreate the shared `.venv` or dependency lock files. Use `.venv/bin/python`
  explicitly from the repository root; use a temporary environment if isolation
  is needed.

## Mathematical and Numerical Reasoning

- Think from mathematical and numerical analysis perspectives before designing,
  implementing, reviewing, or changing code. Identify the governing
  equations, discrete operators, assumptions, boundary conditions, and invariants.
  Check consistency, stability, convergence, error, and derivative correctness.
- Establish the cause and supported domain before choosing a fix. Prefer theoretical
  consistency over patches that merely hide symptoms.
- Check input/output contracts, resource costs, and runtime constraints.
- Preserve the distinction between reflectivity and the echo proxy, missing data
  and clear sky, and model capability and observational evidence. Check spatial and
  temporal alignment and the evaluation domain.

## Implementation

- Always use the Graphify skill for code improvement work. Consult the existing
  structural graph before making changes and incrementally refresh changed code
  afterward. Reuse cached evidence; avoid full semantic rebuilds unless needed.
- Code must be simple, clear, concise, and intuitive.
- Choose the smallest theoretically consistent change. Avoid unrelated refactoring,
  unnecessary abstractions, duplicate logic, and unnecessary or redundant validation
  layers. Preserve validation required by mathematical and input/output contracts.
- Use names and control flow that explain intent. Comments should explain
  mathematical reasons or non-obvious constraints, not repeat the code.
- Check cancellation, overflow, underflow, limits at zero, and boundary behavior.
  Keep derivatives finite in inactive numerical branches as well.
- Preserve automatic differentiation. Do not use `.item()`, `float()`, or `detach()`
  to disconnect quantities being differentiated. Separate fixed branch decisions
  and diagnostics from differentiable calculations.
- Set tolerances from units, dtype, and the relevant component scale. Do not hide
  errors behind unrelated large values.

## Verification and Records

- Evaluate execution success, numerical convergence, forecast accuracy, and model
  validity separately; evidence for one does not establish the others.
- Reproduce defects with small inputs and add regressions that distinguish the
  corrected behavior from the defect. Run affected tests and reuse existing results;
  repeat broader baseline tests only when changes or unresolved failures justify it.
- Record plans, progress, findings, test evidence, and remaining limits in KG and
  task checklists. Keep this file limited to reusable working instructions.
- Preserve existing user changes. Distinguish verified results from assumptions,
  local numerical checks from integration evidence, and proposals from applied fixes.
