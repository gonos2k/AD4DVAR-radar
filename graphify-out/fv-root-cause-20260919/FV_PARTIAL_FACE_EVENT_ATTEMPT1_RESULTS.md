# R4-R two-hole face-event diagnostic: two archived accepted chords

The one predeclared, guarded diagnostic completed. It located a
`q_y[2,0]=0` event on each of the first two **accepted** straight chords
from the PR #209 two-hole collocated root search. At the two prescribed
offsets on each side, all eight 54-stage traces were strict-branch
admitted and their objective gradient/JVP values were finite. This is
finite-offset branch/merit evidence, **not** a new GN/root solve, an
event-point derivative, an eligible implicit response or physical skill.
R4-R remains open.

The frozen plan is `FV_PARTIAL_FACE_EVENT_PLAN.md` (SHA256
`bf6a5634062233167d68a927b7c079da069dcbc7f245daf9d08b28153ebe4d50`).
The child source SHA256 was
`14fd0659353056d1d801b1ba84eb2b2efd9f22421f2456ce8b56320c52d21f36`;
the guarded parent source SHA256 was
`786f510ec820fa448fbfbe3bbea2f980cfc8ca0c20f1a8fd27f9081b13a0dce1`.
Both checked the immutable PR #209 archive manifest
`d80362f8acbb0feb1c82d60d5e9cde61a4863564d7f8f64e49467ba39d12cddd`,
its four raw files, source snapshots, the 58-valid/2-missing fixed
input and exact accepted-control hashes. Parent and child source/input
checks remained unchanged before and after; the parent independently
reconstructed `q_y[2,0]` at both endpoints and bracket/event points,
and every available sample control hash.

| Archived chord | Endpoint `q_y[2,0]` | Located event `t` | Bisections |
|---|---:|---:|---:|
| GN seed → accepted step 1 | `−2.2849761492e−5 → +2.6041837408e−6` | `0.8976906842801782` | 42 |
| Accepted step 1 → step 2 | `+2.6041837408e−6 → −2.0221724001e−5` | `0.11408863233083366` | 44 |

Both event brackets stopped at an exact FP64 zero (`−0.0` in the
record). No derivative was evaluated there. At each chord's near and
far side offsets (`2^-10` and `2^-8` in chord parameter), the two
samples on a given side had the same complete signature. Comparing
opposite sides at the near offset, all limiter choices and every other
face sign matched; only the `q_y[2,0]` sign flipped in each of the 54
stages. That is a property of the **sampled endpoints**. It does not
certify the signature throughout either connecting segment or isolate
this face as the cause of the archived optimizer refusal.

| Chord | Offset | Left `g·H d` | Right `g·H d` | Right − left gradient merit `Phi` |
|---|---:|---:|---:|---:|
| Seed → step 1 | `2^-8` | `−2.98921589e−5` | `+3.48626675e−6` | `+2.80445816e−5` |
| Seed → step 1 | `2^-10` | `−2.98639707e−5` | `+3.48528136e−6` | `+2.81219027e−5` |
| Step 1 → step 2 | `2^-8` | `−4.65961541e−6` | `+2.52475822e−5` | `−2.80658773e−5` |
| Step 1 → step 2 | `2^-10` | `−4.65918839e−6` | `+2.52268566e−5` | `−2.81261639e−5` |

Here `g=grad_c J`, `d` is each archived chord direction and `H d` is
the gradient JVP computed separately at each finite off-event point.
The merit is `Phi=||g||²/2`. The objective difference across near-side
samples shrank from about `3.8e−9` to `9.6e−10` on the first chord
when the offset was quartered, while the merit difference remained
about `2.8e−5`. This is consistent with a gradient change across
the selected face switch, but two finite offsets do **not** establish
a limiting discontinuity or a unique nonsmooth cause.

Every sampled scaled slope margin was about `4.066e−4`, above the
response gate `1e−4`; scaled face margins were only
`3.3454e−7–1.49225e−6`. Thus **all eight** samples fail the existing
response-margin qualification despite strict trace admission and
finite derivatives. No terminal stationarity, exact final Hessian,
adjoint, VJP, signed reanalysis or finite influence was computed.
The diagnostic narrows the numerical question: a next branch-aware
globalization method must handle a local switch where gradient merit
differs strongly between sides, then still find an interior root with
the unchanged final gates. Repeating the prior eight-step policy from
nearby seeds is not established as a solution.

The child exited 0; the parent classified execution `completed` and
numerics `face_event_diagnostic_only`. The guard observed 39 child-RSS
samples, peak **345,260,032 bytes**, and **10.516 seconds** elapsed
under the 300-second wall and sampled 1-GiB triggers; child elapsed
time was **9.092 seconds**. Parent/resource/raw identities and the
fixed input all matched. The final affected selection passed **97
tests** with 18 existing TorchScript deprecation warnings; targeted
basedpyright 1.39.9 reported zero diagnostics. Graphify's code-only
incremental AST refresh produced a parseable graph with 8,294 nodes
and 210,382 links; it did not perform semantic extraction. GREEN and
RED independently checked the actual raw, parent and resource records.
The full CPU/package CI and real radar validation were not run.
