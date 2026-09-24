# Initial-field lab homepage QA (2026-09-24)

Scope: `examples/initial_field_lab` at `http://127.0.0.1:8766`, served by
`.venv/bin/python examples/initial_field_lab/server.py --port 8766`. Port
8765 was occupied by an unrelated server rooted at `/Users/yhlee/Competency`
and was not used as evidence. The fixed synthetic P0 experiment invokes the
real `advar.nowcast`; it is not a live-radar or physical skill evaluation.

## Browser and API checks

Aside Browser loaded the real page and `GET /api/default` with HTTP 200. The
response contained four 48×48 fields. Baseline MAEs were 0.3034 dBZ for
ADVAR and 2.5873 dBZ for persistence, with 1,980 evaluated pixels. All four
canvases had nonempty rendered pixel data; no unexpected console/page errors
were observed. Desktop at 1440×900 CSS pixels had no document horizontal
overflow or detected clipped container.

Aside exercised background on/off, age 60 minutes, north/south +4 px,
east/west −3 px, intensity +6 dBZ, coverage 50%, and lead +60 minutes. Input
changes reached the POST payload and affected the corresponding result or
confidence. Pinning A, changing B, comparing on A's fixed 1,722-pixel domain,
clearing A, and retrying after an incompatible-lead error all worked. The
baseline comparison kept A's persistence MAE at 5.2591 dBZ and reported
`A MAE − B MAE = −0.5749 dBZ` for one +6 dBZ B case.

Direct HTTP checks also returned 200 for the page, JS, CSS, default API and
valid POST, 400 JSON for invalid lead, mismatched A/B lead, malformed JSON,
non-object JSON, oversized body and invalid setting types, and 404 for
unknown routes. `http_checks.py` reproduces these 14 route checks and saves
their statuses in `http-checks.json`. The Python homepage tests passed (10
tests). The JavaScript UI tests passed (3 tests).

## Defects found and fixed

1. A/B lead mismatch previously displayed the backend's English internal
   message, `A/B comparison requires the same lead_minutes`. The form now
   checks the pinned A lead before POST, leaves A intact, and gives a Korean
   instruction to restore the same lead or clear A. In an Aside browser
   recheck, changing the form from pinned +30 to +60 minutes produced the
   Korean message and the `/api/run` count remained 1 before and after the
   blocked Run.
2. On a 390px mobile capture the `결측` legend key wrapped vertically. The
   narrow-screen CSS now moves it to its own horizontal row. Compare
   `mobile-before.png` and `mobile-after.png` in this directory.
3. The settings could change while a calculation was pending, and the displayed
   result could look current after controls changed. Inputs now freeze during
   requests. The status marks settings that differ from the last successful
   result, including when pinning A recalculates the displayed result while
   the form has different settings. A small punctuation separator was also
   added to the A/B improvement label.

The backend's A-fixed scoring formula and the forecast model were not changed.
The client guard duplicates an existing server constraint; the server remains
the authority for direct API callers.

## Limits of this verification

The mobile layout was visually checked with a local Chrome/Playwright 390px
capture. Aside's desktop interactions were tested in its browser at 1440px;
Aside did not complete native mobile emulation. Touch input, other browser
engines, offline operation, unexpected HTTP 5xx responses and every slider
boundary were not exhaustively tested. The canvas labels and summary metrics
are present, but this QA is not a complete screen-reader accessibility audit.
The synthetic MAE values do not establish independent radar forecast skill.

GREEN and RED code reviews found no remaining P1/P2 issue in the fixes. The
source and evidence hashes, exact test commands, and browser scope are in
`manifest.json`.
