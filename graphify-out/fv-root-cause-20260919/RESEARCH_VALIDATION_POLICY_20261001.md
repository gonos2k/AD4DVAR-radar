# Research validation scope (user steering)

The user directed on 2026-10-01: reduce unnecessary CI; this is not for deployment.

Reuse the completed official run `36836402831` and its six raw logs/artifacts. It executed the complete 2,311-ID Linux inventory but failed; retain that result unchanged. Package/CLI and UI passed at its tested commit. Do not repeat full CPU, wheel/offline-install/CLI or deployment gates automatically.

For current repairs, run only affected local CPU tests and necessary mathematical checks. Reuse already passing checks unless code changes or a concrete unresolved concern justify repetition. Distinguish synthetic protocol tests, archive integrity, current-input calculations, numerical response accuracy and physical validity. Local affected-test success does not retroactively make the whole Linux run pass.

A further external CI run needs a concrete research verification reason that local evidence cannot resolve. No deployment or operational-acceptance claim is part of this milestone. Keep full-regression evidence explicit as failed/unverified after local repairs, and continue the research checklist independently.
