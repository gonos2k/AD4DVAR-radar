# Contract-domain boundaries

This document is the mandatory routing map for future contract changes.  It
does not move or rewrite current payload classes; current JSON, SQLite rows,
legacy audit decoding and digest preimages remain governed by their existing
implementations until the external real-case acceptance gate is complete.

| Trust domain | Current authority | May consume | Must not authorize |
|---|---|---|---|
| Native and analysis provenance | `promotion.py`, `ledger.py` | native volume, raw resolution, processor/ingestor trust | training, promotion, deployment |
| Training derivation | `promotion.py`, `ledger.py` | approved provenance, target-source trust, immutable shards | holdout scoring or deployment |
| Verification | `sensitivity.py`, `promotion.py` | typed verification bundle, target identity, observation-error contract | candidate selection by itself |
| Promotion | `promotion.py`, `ledger.py` | completed semantic replay and preregistered policy | runtime installation or activation |
| Deployment | `promotion.py`, `ledger.py`, `build_deployment_bundle.py` | ledgered promotion certificate, signed hermetic bundle, runtime activation receipt | data ingestion or training |
| Legacy audit | explicit `Legacy*Audit` decoders | retained bytes from the exact historical contract | every current producer/selector path |

## Refactor gate

A future behavior-preserving module split may begin only after a report-only
real-case acceptance run covers the required scenario matrix and the current
sample-size preflight.  The refactor PR must demonstrate all of the following
against a frozen fixture set:

1. Current product payload JSON and every domain digest are byte-identical.
2. Historical payloads decode to the same audit-only types and audit digests.
3. SQLite schema, migrations, rows and trigger behavior are unchanged.
4. Public imports, CLI output and wheel contents remain compatible.
5. Semantic replay produces the same ordered evaluation and bundle digests.

New contract generations belong in the owning domain above.  A legacy type
must never be re-exported as a current producer, and trust-store validation
must not be duplicated across domains; consumers call the owning validator.
