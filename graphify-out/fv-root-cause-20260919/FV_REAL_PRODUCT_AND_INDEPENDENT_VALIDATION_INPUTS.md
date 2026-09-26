# R6/R8 external product and validation inputs

This note translates [R6](PR177_195_REVIEW_RESOLUTION.md#L22) and
[R8](PR177_195_REVIEW_RESOLUTION.md#L25) into the smallest external inputs
needed to exercise the current scientific contracts. It does not assert that a particular radar product supplies these
fields or that a cohort has been obtained.

## R6: minimum product contract

Choose one real radar product and freeze its exact product version, native
decoder/version, QC/report-flag interpretation, and legal-use terms. The
minimum product handoff is:

1. **Immutable source and authority.** Native volume/scan bytes and SHA-256;
   radar/site and scan identity; product, decoder, and QC algorithm identities;
   calibration epoch; and a source-authority record rooted in an independently
   managed trust registry that authorizes the signing key and role for the stated
   site/product/time/use scope, plus legal-use terms. Bind canonical valid times,
   per-source acquisition times, time-reference convention, grid/product identity,
   and upstream artifact identity. The current source identity includes an
   authority ID, presented public key, and signature and verifies the signature
   against that presented key ([`sensitivity.py:2150-2165,2223-2291`](../../src/advar/sensitivity.py#L2150)); this protects signed-identity integrity but does
   not establish that the key is trusted or authorized. Raw volume identity and
   ingestor attestation bind site, acquisition time, scan, raw-content digest,
   product and grid ([`promotion.py:697-768`](../../src/advar/promotion.py#L697)).
2. **Measurement meaning.** Calibrated reflectivity values in dBZ, with
   product resolution, quantization origin/bin rule, and the meaning of floor
   and no-data encodings. Give an explicit report kind that distinguishes
   detected echo, confirmed clear, and below-detection censoring. The current
   verification bundle has fields for reflectivity quantization and floor
   policy ([`sensitivity.py:7469-7494`](../../src/advar/sensitivity.py#L7469)); current clear/censor
   semantics and their source evidence are summarized in
   [PR135_CHECKLIST.md:46-48](../../PR135_CHECKLIST.md#L46).
3. **Footprint and time.** Grid CRS/projection identity, shape, spacing,
   pixel-to-projected affine, cell-center origin/convention and grid digest;
   canonical valid times; per-source nominal acquisition times; and
   cell-local time offset/age. These correspond to
   `RadarSpatialGridIdentity` ([`nowcast.py:2655-2673`](../../src/advar/nowcast.py#L2655)),
   `VerificationObservationSourceIdentity` ([`sensitivity.py:2150-2165`](../../src/advar/sensitivity.py#L2150)),
   and bundle time/grid fields ([`sensitivity.py:7473-7494`](../../src/advar/sensitivity.py#L7473)). Current
   chronology uses a signed `volume_end` reference and derives cell age from
   nominal time and offset ([PR135_CHECKLIST.md:46-48](../../PR135_CHECKLIST.md#L46)).
4. **Product-owned QC, censoring and source selection evidence.** For every
   radar/time/cell, provide or deterministically derive availability,
   source assignment and its score, range/elevation, beam-blockage fraction,
   attenuation-QC score, time validity, detection limit, confirmed-clear and
   censor state. Preserve the ordered source/calibration registry and source
   identity. These are the current raw mask-evidence fields at
   [`sensitivity.py:3467-3495`](../../src/advar/sensitivity.py#L3467); registered source calibration,
   quality/error baselines, detection-limit parameters and radar geometry are
   represented by `ObservationRadarSource` ([`sensitivity.py:1725-1742`](../../src/advar/sensitivity.py#L1725)).
   For a mosaic, evidence must remain source-specific and assignment must be
   replayable; a prepared mask alone is insufficient ([PR132_CHECKLIST.md:38-43](../../PR132_CHECKLIST.md#L38)).
5. **Error/covariance semantics.** State the observation-error model, quality
   interpretation, per-cell quality weight and standard deviation, censoring
   rule, source/calibration mapping and covariance assumption, each with
   provenance and a preregistered derivation. Current score weighting is
   `quality × normalized inverse variance`, with invalid cells assigned zero
   weight; the spatial-correlation block is explicitly diagnostic-only
   ([`sensitivity.py:5895-5937`](../../src/advar/sensitivity.py#L5895); [PR131_CHECKLIST.md:30-41](../../PR131_CHECKLIST.md#L30)). Thus the
   current supported minimum is an explicitly declared diagonal error model.
   A product with consequential correlated errors needs independent support
   for that assumption or a separately implemented and validated covariance
   treatment; a correlation-block digest alone cannot establish it.

The current typed path already binds the resulting verification tensors,
observation-error contract, derivation artifact, and source/product/QC
identities ([`sensitivity.py:7469-7494`](../../src/advar/sensitivity.py#L7469)). Product-specific native
decoding and the truth of that product's QC, flags, calibration, and geometry
remain to be supplied and validated; the generic contracts do not establish
those facts.

## R8: minimum independent validation and finite-amplitude plan

Each independent verification case must provide:

- Future reflectivity target frames shaped `[lead, y, x]`, matching the fixed
  forecast grid/domain, canonical valid times/leads, valid mask, product/QC
  identities, and the R6 cell-state/error/source semantics. The current bundle
  enforces frame/mask shape, increasing valid times, and product/grid/QC
  digests ([`sensitivity.py:7469-7494,7498-7510`](../../src/advar/sensitivity.py#L7469)).
- A source authority and native target-source closure independent of candidate
  training and tuning. Verify source authorization against an independently
  managed trust registry; a valid signature under a supplied key alone proves
  integrity under that key, not its authority. Identify the physical storm/event
  from independently sourced track and native/target source, time, interval, and
  domain evidence; bind their digests and provenance. Require weather/range
  assignments to use a preregistered, versioned classification method with
  source attributes, thresholds or adjudication records and reviewer provenance;
  caller labels alone do not establish event identity or regime membership. Do
  not count multiple windows from one event as independent cases
  ([PR129_CHECKLIST.md:39,59](../../PR129_CHECKLIST.md#L39);
  [PR131_CHECKLIST.md:73-75](../../PR131_CHECKLIST.md#L73)). Record the
  storm/event, day, contributing radar(s), weather and range regime, lead,
  valid area/coverage, and exact case/source/grid digests so preregistered
  cohorts and per-regime cells can be reconstructed. Current promotion policy
  represents minimum physical-event counts per weather/range/metric/lead cell
  ([`promotion.py:18446-18505,19307-19410`](../../src/advar/promotion.py#L18446)); the required cohort size
  is the approved policy's preflight result, not a number to infer from this
  review row.
- Paired parent and candidate outputs produced from identical inputs, masks,
  background, grid/time/domain and scoring configuration, plus the declared
  metrics and all exclusions. A different input or target closure cannot be
  treated as the paired comparison.
- A split manifest assigning whole physical events/source closures to
  train, validation and test before fitting. Record the split, feature/target
  tensor and derivation digests, weights and augmentation seed per member;
  derive learned normalization only from training members. Existing
  `TrainingDatasetMember` and `NormalizationDerivationArtifact` represent the
  member split/lineage and train-only statistics ([`promotion.py:5240-5317`](../../src/advar/promotion.py#L5240)).

Before evaluating influence, the scientific protocol must also preregister
the perturbed quantity, units, sign, finite-amplitude interval and selected
endpoint amplitudes; the fixed event/domain cohort; and the reanalysis/forecast
and score to rerun at each endpoint. The R8 ledger calls for a predeclared
finite-amplitude range ([PR177_195_REVIEW_RESOLUTION.md:25](PR177_195_REVIEW_RESOLUTION.md#L25)) but specifies no
numeric range. The study owner must choose it before seeing outcomes. Local
derivative agreement or a local linearized error estimate is not a substitute
for those finite reanalysis endpoints or independent performance evidence
([LEARNING_RESULTS.md:39-43](LEARNING_RESULTS.md#L39)).

## Work that can proceed locally and external blockers

Repository work can proceed without radar observations: write a field-by-field
adapter mapping for a selected product once its public specification is
available; exercise deterministic replay and mutation rejection with synthetic
fixtures; ensure event-level split/preflight and train-only normalization are
bound in the dataset manifest; and define a protocol artifact for the
finite-amplitude plan. These activities test contracts, not radar science.

R6/R8 closure still requires external owners to provide the selected product's
native format/decoder documentation, legally usable immutable case volumes,
authoritative source and calibration/QC metadata, and independent future
verification sources; radar scientists must approve empirical censor/detection,
age, quality/error and covariance assumptions. The study owner must preregister
the independent event cohort, fixed domain/metrics, train/validation/test
separation, sample-size preflight, and finite-amplitude interval. The repository
cannot generate independent physical events or grant source authority. Related
external holds are recorded in [PR131_CHECKLIST.md:71-75](../../PR131_CHECKLIST.md#L71),
[PR135_CHECKLIST.md:55-58](../../PR135_CHECKLIST.md#L55), and
[PR136_CHECKLIST.md:40-44](../../PR136_CHECKLIST.md#L40).

## Workspace data inventory

A read-only scan of this repository (excluding `.git`, `.venv`, and
`graphify-out`) found no `.npy`, `.npz`, NetCDF, HDF5, GRIB, BUFR, radar
manifest, or verification-bundle case files. The existing readiness record
reports that `/Users/yhlee/RADAR` contains simulated radial-velocity scripts
and plots, while `/Users/yhlee/4DVAR` contains Lorenz/ODE experiments and a
model archive, not radar reflectivity cases or independent targets
([real_case_readiness.md:8-23](../review-confirmation-20260906/real_case_readiness.md#L8)). The
local FV learning report is a same-operator synthetic experiment; its small
holdout gain and local error estimate do not establish practical physical
forecast skill or statistical uncertainty ([LEARNING_RESULTS.md:17-21,32-43,54-58](LEARNING_RESULTS.md#L17)).
