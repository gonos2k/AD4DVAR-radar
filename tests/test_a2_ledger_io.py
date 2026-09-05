"""Regression tests for the A2 ledger I/O transaction boundaries."""

from __future__ import annotations

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
import sqlite3
import tempfile
from threading import Barrier
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey as RealEd25519PrivateKey,
)
import torch

from advar import ledger as ledger_module
from advar import promotion as promotion_module
from advar.ledger import EpisodeLedger


class _DelayedCoverageLedger(EpisodeLedger):
    def __init__(self, root: str | Path, clock: object, after_deadline: datetime):
        self._test_clock = clock
        self._test_after_deadline = after_deadline
        super().__init__(root)

    def _connect(self):  # type: ignore[no-untyped-def]
        self._test_clock.now.return_value = self._test_after_deadline
        return super()._connect()


class LedgerIORegressionTests(TestCase):
    def test_failed_receipt_insert_removes_only_new_action_directory(self) -> None:
        test_ledger = importlib.import_module("test_ledger")
        keys: list[RealEd25519PrivateKey] = []

        class KeyFactory:
            @staticmethod
            def generate(*args, **kwargs):
                key = RealEd25519PrivateKey.generate(*args, **kwargs)
                keys.append(key)
                return key

            @staticmethod
            def from_private_bytes(*args, **kwargs):
                return RealEd25519PrivateKey.from_private_bytes(*args, **kwargs)

        original = EpisodeLedger.append_realized_intervention_receipt
        state: dict[str, object] = {}

        def append_once_then_retry(self, decision, receipt, **kwargs):
            result = original(self, decision, receipt, **kwargs)
            if not state:
                duplicate = test_ledger.RealizedInterventionReceipt.from_decision(
                    decision,
                    actual_input_before_context=kwargs[
                        "actual_input_before_context"
                    ],
                    actual_input_before_run=kwargs["actual_input_before_run"],
                    actual_input_after_context=kwargs[
                        "actual_input_after_context"
                    ],
                    actual_input_after_run=kwargs["actual_input_after_run"],
                    action_policy=kwargs["action_policy"],
                    action_generator=kwargs["action_generator"],
                    executor_key_id=receipt.executor_key_id,
                    executor_trust_store_digest=receipt.executor_trust_store_digest,
                    executor_private_key=keys[0],
                    executor_sequence_number=receipt.executor_sequence_number + 1,
                    applied_time=receipt.applied_time,
                    receipt_time=receipt.receipt_time,
                )
                try:
                    original(self, decision, duplicate, **kwargs)
                except Exception as error:  # duplicate decision is intentional
                    state["error"] = (type(error).__name__, str(error))
                state["orphan_digest"] = duplicate.receipt_digest
                state["artifact_dirs"] = tuple(
                    sorted(path.name for path in self.interventions_dir.iterdir())
                )
                with sqlite3.connect(self.index_path) as connection:
                    state["indexed_receipts"] = tuple(
                        row[0]
                        for row in connection.execute(
                            "SELECT receipt_digest "
                            "FROM realized_intervention_receipts"
                        )
                    )
            return result

        case = test_ledger.EpisodeLedgerTests(
            "test_prospective_decision_and_executor_receipt_round_trip"
        )
        case.setUp()
        try:
            with patch.object(test_ledger, "Ed25519PrivateKey", KeyFactory), patch.object(
                EpisodeLedger,
                "append_realized_intervention_receipt",
                append_once_then_retry,
            ):
                case.test_prospective_decision_and_executor_receipt_round_trip()
        finally:
            case.tearDown()

        self.assertEqual(state["error"][0], "IntegrityError")  # type: ignore[index]
        self.assertEqual(
            set(state["artifact_dirs"]),  # type: ignore[arg-type]
            set(state["indexed_receipts"]),  # type: ignore[arg-type]
        )
        self.assertNotIn(
            state["orphan_digest"],  # type: ignore[arg-type]
            state["artifact_dirs"],  # type: ignore[operator]
        )

    def test_source_coverage_rechecks_deadline_inside_write_transaction(self) -> None:
        test_promotion = importlib.import_module("test_promotion")
        test_case = test_promotion.NeuralPriorPromotionTests("runTest")
        plan, input_plan, registry = test_case.mosaic_issuance_context()
        nominal = torch.ones((1, 2, 2), dtype=torch.bool)
        resolved = promotion_module.ResolvedSourceCoverageArtifact.from_observations(
            plan,
            input_plan,
            registry,
            nominal_source_coverage_mask=nominal,
            source_radar_index_map=torch.tensor(
                [[0, 1], [0, 1]], dtype=torch.int64
            ),
            outage_mask=torch.zeros((2, 2), dtype=torch.bool),
            dynamic_qc_valid_mask=torch.ones(
                (2, 2), dtype=torch.bool
            ),
            input_bundle_digest="e" * 64,
            full_analysis_input_digest="1" * 64,
            resolved_at="2026-08-09T00:01:00Z",
            data_ingestor_id="trusted-radar-ingestor",
            data_ingestor_private_key=test_case.scheduler_key(),
        )
        domain = promotion_module.OperationalIssuanceDomainArtifact.from_masks(
            plan,
            publication_eligible_mask=nominal,
            source_coverage_mask=nominal,
            permanent_exclusion_mask=torch.zeros_like(nominal),
            resolved_source_coverage=resolved,
        )
        before_deadline = datetime.fromisoformat("2026-08-09T00:01:30+00:00")
        after_deadline = datetime.fromisoformat("2026-08-09T00:03:00+00:00")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ledger_module, "datetime", wraps=datetime) as clock:
                clock.now.return_value = before_deadline
                ledger = _DelayedCoverageLedger(Path(directory), clock, after_deadline)
                with self.assertRaisesRegex(ValueError, "pre-issue"):
                    ledger.append_resolved_source_coverage_artifact(
                        plan, input_plan, resolved, domain
                    )
            with sqlite3.connect(ledger.index_path) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM "
                    "neural_prior_resolved_source_coverage_artifacts"
                ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_concurrent_recovery_returns_success_to_both_workers(self) -> None:
        """Exercise the public typed operational recovery path after a crash."""

        test_promotion = importlib.import_module("test_promotion")
        holdout_case = test_promotion.NeuralPriorPromotionTests("runTest")
        holdout = holdout_case.plan()
        planned = holdout.cases[0]
        input_plan = next(
            item for item in holdout.input_plans if item.plan_digest == planned.input_plan_digest
        )
        sampling_unit = next(
            item
            for item in holdout.meteorological_sampling_units
            if item.sampling_unit_digest == planned.meteorological_sampling_unit_digest
        )
        slots = tuple(
            item
            for item in holdout.raw_observation_slot_plans
            if item.slot_digest in sampling_unit.raw_observation_slot_digests
        )
        range_band = next(
            item
            for item in holdout.range_band_contracts
            if item.contract_digest == planned.range_band_contract_digest
        )
        geometry = next(
            item
            for item in holdout.range_geometry_contracts
            if item.contract_digest == range_band.range_geometry_contract_digest
        )
        issuance = next(
            item
            for item in holdout.operational_issuance_domain_plans
            if item.plan_digest == planned.operational_issuance_domain_plan_digest
        )
        processor_key = promotion_module.Ed25519PrivateKey.from_private_bytes(b"\x23" * 32)
        operational_plan = ledger_module.OperationalAnalysisInputProvenancePlan(
            plan_id="concurrent-operational-cycle",
            input_plan=input_plan,
            raw_observation_slot_plans=slots,
            raw_ingestor_trust_store=holdout.raw_ingestor_trust_store,
            analysis_processor_id="test-analysis-processor",
            analysis_processor_public_key_hex=processor_key.public_key().public_bytes_raw().hex(),
            analysis_processor_trust_store_digest="7" * 64,
            range_geometry_contract=geometry,
            operational_issuance_domain_plan=replace(
                issuance, case_id="concurrent-operational-cycle"
            ),
            registered_at="2026-08-07T00:00:00Z",
        )
        _, _, _, _, _, base_run, _, receipts, _ = holdout_case.analysis_input_context(
            1, plan=holdout
        )
        resolution = ledger_module.OperationalRawVolumeResolutionReceipt(
            provenance_plan_digest=operational_plan.plan_digest,
            input_plan_digest=input_plan.plan_digest,
            slot_identity_bindings=tuple(
                (item.slot_plan_digest, item.raw_volume_identity.identity_digest)
                for item in receipts
            ),
            history_entries=tuple(
                ledger_module.OperationalRawResolutionHistoryEntry.issue(
                    provenance_plan_digest=operational_plan.plan_digest,
                    slot_digest=item.slot_plan_digest,
                    resolution_identity_digest=item.raw_volume_identity.identity_digest,
                    resolution_kind="resolved",
                    previous_entry_digest=ledger_module.OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST,
                    transition="original",
                    reason="concurrent-public-path-probe",
                    issued_at=max(
                        retained.raw_volume_attestation.received_at for retained in receipts
                    ),
                    authority_id=operational_plan.analysis_processor_id,
                    authority_private_key=processor_key,
                )
                for item in receipts
            ),
            resolved_at=max(
                item.raw_volume_attestation.received_at for item in receipts
            ),
        )
        derivation = ledger_module.AnalysisInputDerivationArtifact.from_products(
            case_id=operational_plan.plan_id,
            input_plan=input_plan,
            resolved_raw_observations=receipts,
            global_resolution_receipt=resolution,
            run=base_run,
            resolved_source_coverage=None,
            background_frames_dbz=None,
            processed_at=(
                promotion_module._canonical_datetime(resolution.resolved_at)
                + timedelta(seconds=10)
            ).isoformat(),
            processor_id=operational_plan.analysis_processor_id,
            processor_private_key=processor_key,
        )
        run = replace(
            base_run,
            analysis_input_derivation_artifact_json=json.dumps(
                derivation.payload, sort_keys=True, separators=(",", ":")
            ),
            analysis_input_derivation_artifact_digest=derivation.artifact_digest,
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            authority_trust = holdout_case.provenance_authority_trust_store(
                ledger,
                authority_id=operational_plan.analysis_processor_id,
                private_key=processor_key,
                content_digest="7" * 64,
            )
            signer = holdout_case.provenance_ledger_signer(
                (
                    promotion_module._canonical_datetime(derivation.processed_at)
                    + timedelta(seconds=1)
                ).isoformat()
            )
            with (
                patch.object(
                    ledger_module,
                    "_load_promotion_deployment_authority_trust_store",
                    return_value=authority_trust,
                ),
                patch.object(
                    ledger_module,
                    "_load_raw_ingestor_trust_store",
                    return_value=holdout.raw_ingestor_trust_store,
                ),
                patch.object(ledger_module, "datetime", wraps=datetime) as clock,
            ):
                clock.now.return_value = datetime.fromisoformat(
                    "2026-08-08T00:00:00+00:00"
                )
                ledger.append_operational_analysis_input_provenance_plan(
                    operational_plan,
                    analysis_processor_trust_store_path="/unused/deployment.json",
                )
                clock.now.return_value = promotion_module._canonical_datetime(
                    derivation.processed_at
                ) + timedelta(seconds=1)
                with patch.object(
                    EpisodeLedger,
                    "reconcile_prepared_analysis_input_provenance",
                    side_effect=RuntimeError("simulated postcommit crash"),
                ), patch.object(
                    EpisodeLedger,
                    "_issue_analysis_provenance_preparation_receipt",
                    return_value=(None, None),
                ):
                    with self.assertRaisesRegex(RuntimeError, "postcommit crash"):
                        ledger.append_operational_analysis_input_provenance(
                            operational_plan,
                            run=run,
                            resolved_raw_observations=receipts,
                            raw_resolution=resolution,
                            derivation=derivation,
                            resolved_source_coverage=None,
                            raw_ingestor_trust_store_path="/unused/raw.json",
                            analysis_processor_trust_store_path="/unused/deployment.json",
                            provenance_commit_signer=signer,
                        )

                def recover(_: int) -> str:
                    return ledger.reconcile_prepared_analysis_input_provenance(
                        derivation.artifact_digest,
                        raw_ingestor_trust_store_path="/unused/raw.json",
                        analysis_processor_trust_store_path="/unused/deployment.json",
                        provenance_commit_signer=signer,
                    )

                receipt_barrier = Barrier(2)
                issue_receipt = (
                    EpisodeLedger._issue_analysis_provenance_preparation_receipt
                )

                def issue_receipt_at_the_same_time(*args, **kwargs):
                    value = issue_receipt(*args, **kwargs)
                    receipt_barrier.wait(timeout=10)
                    return value

                with patch.object(
                    EpisodeLedger,
                    "_issue_analysis_provenance_preparation_receipt",
                    side_effect=issue_receipt_at_the_same_time,
                ), ThreadPoolExecutor(max_workers=2) as executor:
                    results = tuple(executor.map(recover, (0, 1)))
            self.assertEqual(results, (derivation.artifact_digest,) * 2)
            with ledger._connect() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT status,usable FROM analysis_input_provenance_commits "
                        "WHERE artifact_digest = ?",
                        (derivation.artifact_digest,),
                    ).fetchone(),
                    ("active", 1),
                )


if __name__ == "__main__":
    import unittest

    unittest.main()
