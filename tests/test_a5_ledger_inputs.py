from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import advar.ledger as ledger_module
from advar._digest import tensor_digest
from advar.ledger import EpisodeLedger, ModelContract
from advar.promotion import (
    CanonicalRawGridVolumeArtifact,
    OperationalAnalysisInputProvenancePlan,
    OperationalIssuanceDomainPlan,
    OperationalRawResolutionHistoryEntry,
    RawIngestorTrustStore,
    RawObservationSlotPlan,
    ResolvedRawObservationReceipt,
    NeuralPriorInputPlan,
)


class A5LedgerInputTests(unittest.TestCase):
    def test_raw_receipt_accepts_equivalent_uppercase_trust_key(self) -> None:
        private_key = Ed25519PrivateKey.from_private_bytes(b"\x11" * 32)
        public_key = private_key.public_key().public_bytes_raw().hex()
        slot = RawObservationSlotPlan(
            radar_site_digest="a" * 64,
            acquisition_valid_time="2026-08-01T00:00:00Z",
            scan_strategy_rule_digest="b" * 64,
            source_selection_rule_digest="c" * 64,
            canonical_geodetic_footprint_digest="d" * 64,
        )
        volume = CanonicalRawGridVolumeArtifact.from_tensors(
            reflectivity_dbz=torch.ones((2, 2), dtype=torch.float32),
            qc_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            quality_weight=torch.ones((2, 2), dtype=torch.float32),
            observation_std_dbz=torch.full((2, 2), 2.0),
            radar_site_digest=slot.radar_site_digest,
            acquisition_valid_time=slot.acquisition_valid_time,
            canonical_scan_identity_digest=slot.scan_strategy_rule_digest,
            radar_product_digest="e" * 64,
            grid_contract_digest="f" * 64,
        )
        receipt = ResolvedRawObservationReceipt.from_ingestor(
            slot=slot,
            raw_grid_volume=volume,
            raw_ingestor_id="ingestor",
            raw_ingestor_private_key=private_key,
            received_at="2026-08-01T00:01:00Z",
        )
        trust = RawIngestorTrustStore(
            authorities=(
                (
                    "ingestor",
                    public_key.upper(),
                    "2026-01-01T00:00:00Z",
                    "2027-01-01T00:00:00Z",
                    None,
                ),
            )
        )

        receipt.validate_against(slot, trust)

    def test_episode_listing_orders_by_utc_without_rewriting_legacy_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(directory)
            rows = (
                ("older", "2026-07-26T06:00:00+01:00"),
                ("newer", "2026-07-26T05:30:00+00:00"),
            )
            with ledger._connect() as connection:
                for episode_id, issue_time in rows:
                    connection.execute(
                        "INSERT INTO episodes ("
                        "episode_id,issue_time,radar_id,contract_hash,"
                        "model_commit,trust_score,promotion_eligible,"
                        "impact_available,indirect_observation_sensitivity_available,"
                        "manifest_sha256,arrays_sha256,path,created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            episode_id,
                            issue_time,
                            "radar",
                            "contract",
                            "model",
                            1.0,
                            0,
                            0,
                            0,
                            "1" * 64,
                            "2" * 64,
                            f"episodes/{episode_id}",
                            "2026-07-26T00:00:00Z",
                        ),
                    )

            listed = ledger.list_episodes()

        self.assertEqual([item["episode_id"] for item in listed], ["newer", "older"])
        self.assertEqual(
            [item["issue_time"] for item in listed],
            ["2026-07-26T05:30:00+00:00", "2026-07-26T06:00:00+01:00"],
        )

    def test_operational_plan_rejects_duplicate_single_site_time_slot(self) -> None:
        input_plan = self._input_plan()
        site = "a" * 64
        source_rule = "b" * 64
        slots = tuple(
            RawObservationSlotPlan(
                radar_site_digest=site,
                acquisition_valid_time=valid_time,
                scan_strategy_rule_digest=(
                    "c" * 64 if index == 0 else "0" * 64
                ),
                source_selection_rule_digest=source_rule,
                canonical_geodetic_footprint_digest="d" * 64,
            )
            for index, valid_time in enumerate(
                (
                    input_plan.valid_times[0],
                    input_plan.valid_times[0],
                    input_plan.valid_times[1],
                    input_plan.valid_times[2],
                )
            )
        )
        grid_xy_digest = tensor_digest(torch.zeros((2, 2)))
        geometry = ledger_module.RangeGeometryContract(
            radar_site_digest=site,
            radar_site_location_digest="e" * 64,
            grid_contract_digest=input_plan.grid_contract_digest,
            radar_x_m=0.0,
            radar_y_m=0.0,
            range_regime_labels=("near", "far"),
            radial_distance_edges_m=(0.0, 1.0, 2.0),
            horizontal_range_rule_digest="f" * 64,
            grid_x_m_digest=grid_xy_digest,
            grid_y_m_digest=grid_xy_digest,
        )
        mask = torch.ones((1, 2, 2), dtype=torch.bool)
        issuance = OperationalIssuanceDomainPlan(
            case_id="cycle",
            grid_contract_digest=input_plan.grid_contract_digest,
            radar_source_contract_digest=source_rule,
            lead_minutes=(60,),
            publication_policy_digest="1" * 64,
            source_coverage_policy_digest="2" * 64,
            permanent_exclusion_policy_digest="3" * 64,
            publication_eligible_mask_digest=tensor_digest(mask),
            source_coverage_mask_digest=tensor_digest(mask),
            permanent_exclusion_mask_digest=tensor_digest(~mask),
        )

        with self.assertRaisesRegex(ValueError, "one per time"):
            OperationalAnalysisInputProvenancePlan(
                plan_id="cycle",
                input_plan=input_plan,
                raw_observation_slot_plans=slots,
                raw_ingestor_trust_store=RawIngestorTrustStore(
                    authorities=(
                        (
                            "ingestor",
                            "11" * 32,
                            "2026-01-01T00:00:00Z",
                            "2030-01-01T00:00:00Z",
                            None,
                        ),
                    )
                ),
                analysis_processor_id="processor",
                analysis_processor_public_key_hex=("22" * 32),
                analysis_processor_trust_store_digest="4" * 64,
                range_geometry_contract=geometry,
                operational_issuance_domain_plan=issuance,
                registered_at="2026-07-31T00:00:00Z",
            )

    def test_operational_coverage_binding_rejects_cross_plan_and_raw_digest(self) -> None:
        input_plan = self._input_plan()
        source_registry = "a" * 64
        issuance = OperationalIssuanceDomainPlan(
            case_id="cycle",
            grid_contract_digest=input_plan.grid_contract_digest,
            radar_source_contract_digest="b" * 64,
            lead_minutes=(60,),
            publication_policy_digest="c" * 64,
            source_coverage_policy_digest="d" * 64,
            permanent_exclusion_policy_digest="e" * 64,
            publication_eligible_mask_digest="f" * 64,
            source_coverage_mask_digest="1" * 64,
            permanent_exclusion_mask_digest="2" * 64,
            radar_source_kind="mosaic",
            source_radar_registry_digest=source_registry,
            source_radar_count=2,
            data_ingestor_id="data-ingestor",
            data_ingestor_public_key_hex="33" * 32,
        )
        coverage = SimpleNamespace(
            issuance_domain_plan_digest=issuance.plan_digest,
            case_id="cycle",
            grid_contract_digest=input_plan.grid_contract_digest,
            radar_source_contract_digest=issuance.radar_source_contract_digest,
            source_coverage_policy_digest=issuance.source_coverage_policy_digest,
            input_bundle_digest="4" * 64,
            full_analysis_input_digest="5" * 64,
            input_available_at=input_plan.input_available_time,
            decision_deadline=input_plan.decision_deadline,
            publication_time=input_plan.publication_time,
            source_radar_registry_digest=source_registry,
            source_radar_count=2,
            data_ingestor_id="data-ingestor",
            data_ingestor_public_key_hex=("33" * 32).upper(),
            nominal_source_coverage_mask_digest=issuance.source_coverage_mask_digest,
            source_radar_site_digests=("6" * 64, "7" * 64),
        )

        with patch.object(ledger_module, "validate_resolved_source_coverage_artifact"):
            ledger_module._validate_operational_source_coverage_binding(
                coverage,
                issuance_plan=issuance,
                input_plan=input_plan,
                expected_case_id="cycle",
                expected_input_bundle_digest="4" * 64,
                expected_full_analysis_input_digest="5" * 64,
                expected_site_digests=("6" * 64, "7" * 64),
            )
            with self.assertRaisesRegex(ValueError, "operational plan"):
                ledger_module._validate_operational_source_coverage_binding(
                    SimpleNamespace(
                        **{
                            **coverage.__dict__,
                            "issuance_domain_plan_digest": "8" * 64,
                        }
                    ),
                    issuance_plan=issuance,
                    input_plan=input_plan,
                    expected_case_id="cycle",
                    expected_input_bundle_digest="4" * 64,
                    expected_full_analysis_input_digest="5" * 64,
                    expected_site_digests=("6" * 64, "7" * 64),
                )
            with self.assertRaisesRegex(ValueError, "operational plan"):
                ledger_module._validate_operational_source_coverage_binding(
                    SimpleNamespace(
                        **{
                            **coverage.__dict__,
                            "input_bundle_digest": "9" * 64,
                        }
                    ),
                    issuance_plan=issuance,
                    input_plan=input_plan,
                    expected_case_id="cycle",
                    expected_input_bundle_digest="4" * 64,
                    expected_full_analysis_input_digest="5" * 64,
                    expected_site_digests=("6" * 64, "7" * 64),
                )

    def test_model_contract_rejects_non_string_annotated_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "model contract fields"):
            ModelContract(
                model_commit="model",
                residual_contract_version="residual",
                forecast_metric_version="metric",
                observation_contract_version="observation",
                forecast_integrator_version="integrator",
                grid_geometry_version="grid",
                radar_qc_version="qc",
                nowcast_config_digest=1,  # type: ignore[arg-type]
                sensitivity_config_digest="b" * 64,
            )

    def test_operational_history_duplicate_is_idempotent(self) -> None:
        private_key = Ed25519PrivateKey.from_private_bytes(b"\x44" * 32)
        entry = OperationalRawResolutionHistoryEntry.issue(
            provenance_plan_digest="1" * 64,
            slot_digest="2" * 64,
            resolution_identity_digest="3" * 64,
            resolution_kind="resolved",
            previous_entry_digest=ledger_module.OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST,
            transition="original",
            reason="initial",
            issued_at="2026-08-01T00:01:00Z",
            authority_id="processor",
            authority_private_key=private_key,
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(directory)
            with ledger._connect() as connection:
                ledger._record_operational_raw_resolution_history(
                    connection,
                    entry=entry,
                    raw_resolution_receipt_digest="4" * 64,
                    recorded_at="2026-08-01T00:02:00Z",
                    expected_authority_id="processor",
                    expected_authority_public_key_hex=private_key.public_key().public_bytes_raw().hex(),
                )
                ledger._record_operational_raw_resolution_history(
                    connection,
                    entry=entry,
                    raw_resolution_receipt_digest="4" * 64,
                    recorded_at="2026-08-01T00:03:00Z",
                    expected_authority_id="processor",
                    expected_authority_public_key_hex=private_key.public_key().public_bytes_raw().hex(),
                )
                count = connection.execute(
                    "SELECT count(*) FROM operational_raw_resolution_history"
                ).fetchone()[0]
        self.assertEqual(count, 1)

    @staticmethod
    def _input_plan() -> NeuralPriorInputPlan:
        valid_times = (
            "2026-08-01T00:00:00Z",
            "2026-08-01T00:10:00Z",
            "2026-08-01T00:20:00Z",
        )
        return NeuralPriorInputPlan(
            valid_times=valid_times,
            grid_contract_digest="6" * 64,
            radar_product_digest="7" * 64,
            qc_pipeline_digest="8" * 64,
            background_cycle_rule_digest="9" * 64,
            mask_policy_digest="a" * 64,
            observation_valid_time=valid_times[-1],
            input_available_time="2026-08-01T00:21:00Z",
            decision_deadline="2026-08-01T00:25:00Z",
            publication_time="2026-08-01T00:30:00Z",
        )


if __name__ == "__main__":
    unittest.main()
