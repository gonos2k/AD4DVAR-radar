# Initial-field homepage verification checklist

- [x] Identify the ADVAR local server; avoid unrelated process on port 8765.
- [x] Serve current source on port 8766 and inspect the actual page with Aside.
- [x] Verify initial API, four 48×48 rendered canvases, controls and Run.
- [x] Verify A pin, fixed-domain B comparison, A clear, error and retry.
- [x] Check valid/invalid HTTP routes and request payloads.
- [x] Reproduce and fix Korean A/B lead-mismatch guidance without sending POST.
- [x] Mark form settings that differ from the displayed result; freeze inputs
  while a request runs.
- [x] Reproduce and fix the 390px `결측` legend wrap; keep before/after captures.
- [x] Run affected Python and JavaScript tests; obtain GREEN/RED review.
- [ ] Separate future work: touch and other browser engines, offline/HTTP 5xx
  recovery, exhaustive slider boundaries and screen-reader spatial alternatives.
- [ ] Independent weather forecast skill remains outside this synthetic P0 page.
