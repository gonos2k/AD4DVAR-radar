# Working instructions review — 2026-09-19

Completed documentation-only changes:
- Mathematical and numerical analysis perspectives apply before design, implementation, review and edits.
- Explicit active delegation to small Luna high GREEN/RED teams, with session-end review as requested.
- Repository-root .venv/bin/python is explicit and was verified executable; shared environment remains unchanged.
- Unnecessary/redundant validation layers are discouraged; required mathematical and I/O validation is preserved.
- Execution, convergence, forecast accuracy and model validity require distinct evidence.

GREEN confirmed intent and wording. RED found no material conflict in the three final clarifications. Earlier RED noted that mandatory review has token overhead; the user's explicit requirement remains in force. Existing user changes were preserved. git diff --check -- AGENTS.md passed. No numerical code, dependencies, or simulation results were changed; no numerical tests or graph rebuild were needed.

## Project relevance audit

A subsequent GREEN/RED review found no confirmed project-unrelated rules to remove. GREEN recommended retaining the current instructions. RED questioned KG/checklist records only if no established workflow existed; existing project checklists and the user's explicit record-keeping instructions satisfy that condition. No AGENTS.md text was removed in this audit, preserving the user's requested working principles.

The requested Aside skill entry point was read and `aside guide` executed. It returned only a generic request to describe a goal, not operational instructions; no browser operation was attempted through undocumented commands.

Aside follow-up: CLI help showed the installed release did not expose `guide`.
Following the skill's update instruction, Aside CLI was updated from
1.26.810.1915 to 1.26.916.1741. `aside guide` and `aside guide repl` then returned
the actual documented instructions successfully. This is an environment tool
update, not an ADVAR numerical dependency change.
