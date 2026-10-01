# Portable synthetic unit fixtures with strict research runtime gates

During attempt 3, run `36813848349` remains live at exact commit
`39156508f085970a6985b73047f37059178ae488`; its UI shows failure
markers but final failure details are not yet available. Do not cancel
or redispatch it. Independently, a small local process simulating only
the CI Python/Torch metadata reproduces five field-correction fixture
failures: 23 passed / 5 failed, against 28 passes with actual local
metadata. This is metadata simulation, not a Linux numerical replay.

The fake-child tests call the real archived execution certification
loader. That loader correctly refuses a host other than the saved
Python 3.12.13 / Torch 2.13.0 environment. Keep that production
runtime gate and all archive/hash/source checks unchanged.

Make only the synthetic tests supply the hash-bound archived control
vectors through a scoped test provider; build their numerical and
branch fixture measurements freshly on the current host. The provider
does not certify the old execution on a new platform, and no real
optimizer or scientific-response result is exercised. Keep the final
audit and forged-data rejection assertions. Add one independent
regression proving the real loader still rejects runtime drift.

Run the affected tests with actual local metadata and in a separate
simulated CI-metadata process. Preserve before/after logs and label
the latter narrowly. Refresh only the changed tests in the structural
graph, retaining untouched records. Do not edit old numerical reports,
production loaders, source/environment certificates, shared `.venv`,
locks, workflow isolation or test selection. Hold the new candidate
push until the current official run terminates and its full failures
are inspected, then record any next official run at its exact head.
