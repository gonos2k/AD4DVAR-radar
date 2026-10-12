# RED final: cached-gradient original-J/R progress run

## Frozen-input audit

I independently reloaded the frozen plan with `cached_gradient_progress.load_plan`.
Plan SHA-256 is
`5410fd4ae986d07a8b9a1d112c106fd79acdc6d228d18153d6adb24f361eb866`; the
frozen source bundle hash is
`0166b2a66e55c4a3e182b96f24a79f4d903147dd1d5076f4871b30d1fa281b3a`. The plan
contains 228 source hashes and 296 archive hashes, with exactly two numerical
source transitions: `fv_point_3h_subspace_model.py` and
`fv_point_3h_netflux_continuation.py`. Source hashes remained unchanged across
the run. The inherited PR302 cache had already admitted its 222 source members,
291 archive pins, five receipt members, and closed 7839 endpoint snapshot.

The frozen policy is 19 maximum commits, 24 total candidate evaluations, six
per point, 36 fresh HVPs plus four historical HVPs under a total cap of 40,
one guarded launch, two times the base radius, a 600-second internal and
660-second outer cap, and a 2 GiB sampled RSS cap. The prior 79.7018435 seconds
were charged. The rolling comparison is logged without stopping the run or
gating technique selection. There was no new GN operator path.

## Run closure and numerical record

`step.run.json` links to the SHA-256 of `step.json`, and its resource object
matches `step.resource.json`. The outer guard completed with exit code 0, no
monitor error or resource termination, 531.935 seconds elapsed, and sampled
peak RSS 497,696,768 bytes. Including the charged preparation, guarded elapsed
time was 611.637 seconds, below the 660-second outer cap.

The child ended with `budget_exhausted` after nine accepted/closed commits and
nine candidate evaluations. It completed 16 of 17 started *new* HVP calls; the
17th event is point 9, side -1, stopped at `precheck` before the JVP. Point 0
reused exactly two cached baseline side HVPs after fresh native pair, P2,
direction, full-G, input, runtime, and cached-DGd/model closure checks. Point 0
records zero fresh HVP calls; points 1 through 8 each record two completed
fresh side HVPs. Their previous-P2, minimum-mixture-P2, and full-G checks all
passed. All nine committed J and R values decreased monotonically. The
three-step rolling comparison trigger remained false.

Source, fixed-input, and runtime closure are true. Root, minimum, response,
score, original-problem-solved, and terminal-E claims are false; no future score
was evaluated. This supports bounded local original-J/R progress at this
cached 7839 endpoint only.

One timing detail should remain visible in the handoff: child `elapsed_seconds`
is 527.458 seconds, or 607.160 seconds after adding the charged prior work,
which is 7.160 seconds beyond the nominal 600-second internal budget. The child
records `deadline_passed=false` and `budget_exhausted`; the last new HVP is only
at precheck. The outer 660-second guard remained within budget. This is
consistent with internal deadline detection followed by final identity checks
and serialization, but the receipt does not time those separately, so that
explanation is an inference rather than a measured split.

## Test caveat and compatibility limit

The saved protocol test log reports **54 passed and 2 failed**; the independent
active QR/rank suite passed all 27 tests. The two failures are old endpoint
tests whose ancestry loader compares the updated continuation dispatcher to a
historical source-byte pin before reaching their endpoint assertions. They do
not report a numerical assertion failure in the new progress path. The new
driver admits the immutable closed endpoint snapshot and verifies the updated
source transition separately. Preserve this as a legacy-loader compatibility
limit; do not change old source arrays or fabricate hashes to make those tests
green.

The targeted contracted-alpha test passes in the protocol suite. The live run
itself had no contracted chart candidate (one chart call per accepted trial),
so the contraction assertion is established by that test rather than by this
run. No additional physics run was performed by RED.
