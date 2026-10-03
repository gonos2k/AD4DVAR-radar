# Targeted Linux repair check

Run `37097672788` completed successfully at source
`121ef5ad50eb8c08437f819bc3a468b96a9663b1`. The source-anchor gate, Python
3.12.14 / pytest 9.1.1 / PyTorch 2.13.0+cpu checks, and registered SHA-256
fingerprints passed on `ubuntu-24.04`.

JUnit reports exactly two tests, zero failures, zero errors, and zero skips;
pytest reports `2 passed` in 7.84 seconds. The selected tests were:

- `test_new_direction_clears_old_pairs_and_computes_its_cross`
- `test_completed_direction_resume_validates_without_repeating_reanalysis`

The uploaded artifact is ID `11264827585`, SHA-256
`d9fd7d12b9234550649765f98dbf867001f2517aad5baa678710959d885cb5da`.
`SHA256SUMS` records the preserved run metadata, log, provenance, pytest log,
and JUnit report.

Run `37095628674` was a selector collection failure with no tests executed.
Run `37096279408` passed five tests; its two resume cases exposed an
absolute-path bug in the test helper. Both cases now pass after rebasing only a
temporary test copy of the archived source-path map. These targeted results
are from separate commits and do not turn the historical full run
`36836402831` into a pass. No whole Linux suite, package job, or deployment
check was run.

The cancellation-sensitive theta case passed in run `37096279408` while
requesting adjoint tolerance `1e-12`. This confirms current Linux behavior for
that registered case; it does not isolate that tolerance as the cause of the
original full-run failure at `a9e321d`.
