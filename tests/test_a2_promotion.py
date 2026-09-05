from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_promotion as promotion_fixtures  # noqa: E402
import advar.promotion as promotion  # noqa: E402


class A2PromotionTests(unittest.TestCase):
    @staticmethod
    def _catalog_pair(
        *,
        displacement_m: float,
    ):
        key = Ed25519PrivateKey.from_private_bytes(bytes([0x51]) * 32)
        scheduler_key = Ed25519PrivateKey.from_private_bytes(
            bytes([0x52]) * 32
        )
        start_a = "2026-08-10T00:05:00Z"
        end_a = "2026-08-10T00:15:00Z"
        start_b = "2026-08-10T00:00:00Z"
        end_b = "2026-08-10T00:10:00Z"
        track_a = promotion.PhysicalEventTrackArtifact(
            timestamps=(start_a, end_a),
            centroid_xy_m=((0.0, 0.0), (0.0, 0.0)),
            object_mask_digests=("a" * 64, "b" * 64),
            source_radar_ids=("radar", "radar"),
            association_edge_digests=("c" * 64,),
            spatial_reference_digest="d" * 64,
        )
        track_b = promotion.PhysicalEventTrackArtifact(
            timestamps=(start_b, end_b),
            centroid_xy_m=((0.0, 0.0), (displacement_m, 0.0)),
            object_mask_digests=("e" * 64, "f" * 64),
            source_radar_ids=("radar", "radar"),
            association_edge_digests=("1" * 64,),
            spatial_reference_digest="d" * 64,
        )
        common = {
            "participating_radar_ids": ("radar",),
            "association_algorithm_digest": "2" * 64,
            "adjudication_policy_digest": "3" * 64,
            "adjudicator_id": "labeler",
            "adjudicator_private_key": key,
        }
        event_a = promotion.PhysicalEventCatalogEvidence.from_members(
            event_id="a",
            member_case_ids=("case-a",),
            member_full_analysis_input_digests=("4" * 64,),
            start_time=start_a,
            end_time=end_a,
            spatial_envelope_xy_m=(-1.0, -1.0, 1.0, 1.0),
            object_track_artifact=track_a,
            **common,
        )
        event_b = promotion.PhysicalEventCatalogEvidence.from_members(
            event_id="b",
            member_case_ids=("case-b",),
            member_full_analysis_input_digests=("5" * 64,),
            start_time=start_b,
            end_time=end_b,
            spatial_envelope_xy_m=(
                -1.0,
                -1.0,
                displacement_m + 1.0,
                1.0,
            ),
            object_track_artifact=track_b,
            **common,
        )
        plan = promotion.PhysicalEventCatalogPlan(
            holdout_case_ids=("case-a", "case-b"),
            association_algorithm_digest="2" * 64,
            spatial_membership_rule_digest="6" * 64,
            adjudication_policy_digest="3" * 64,
            adjudicator_id="labeler",
            adjudicator_public_key_hex=promotion.regime_reference_public_key_hex(
                key
            ),
            catalog_completion_deadline="2026-08-10T01:00:00Z",
            spatial_reference_digest="d" * 64,
            motion_association_rule_digest="7" * 64,
            scheduler_id="scheduler",
            scheduler_public_key_hex=promotion.regime_reference_public_key_hex(
                scheduler_key
            ),
            scheduler_trust_store_digest="8" * 64,
        )

        def membership(event, case_id, full_digest):
            track = event.object_track_artifact
            return promotion.PhysicalEventCaseSpatialEvidence(
                case_id=case_id,
                full_analysis_input_digest=full_digest,
                physical_event_identity_digest=(
                    event.physical_event_identity_digest
                ),
                observed_spatial_envelope_xy_m=event.spatial_envelope_xy_m,
                event_spatial_envelope_xy_m=event.spatial_envelope_xy_m,
                spatial_membership_rule_digest="6" * 64,
                source_object_evidence_digest=track.object_mask_digests[0],
                track_artifact_digest=track.artifact_digest,
                track_sample_index=0,
                track_sample_time=track.timestamps[0],
                track_object_mask_digest=track.object_mask_digests[0],
                input_available_time="2026-08-10T00:00:00Z",
                spatial_reference_digest="d" * 64,
            )

        memberships = (
            membership(event_a, "case-a", "4" * 64),
            membership(event_b, "case-b", "5" * 64),
        )
        return key, plan, event_a, event_b, memberships

    def test_overlapping_tracks_are_symmetric_and_connected(self):
        key, plan, first, second, memberships = self._catalog_pair(
            displacement_m=1_000.0
        )
        self.assertEqual(
            promotion._overlap_track_distance(
                first.object_track_artifact,
                second.object_track_artifact,
            ),
            promotion._overlap_track_distance(
                second.object_track_artifact,
                first.object_track_artifact,
            ),
        )
        self.assertTrue(promotion._events_associate(first, second, plan))
        self.assertTrue(promotion._events_associate(second, first, plan))
        for events in ((first, second), (second, first)):
            with self.assertRaisesRegex(
                ValueError,
                "association graph has split connected components",
            ):
                promotion.PhysicalEventCatalogResult.from_plan(
                    plan,
                    event_evidences=events,
                    case_spatial_membership_evidences=memberships,
                    cataloged_at="2026-08-10T00:30:00Z",
                    adjudicator_private_key=key,
                )

    def test_overlapping_tracks_are_symmetric_and_disconnected(self):
        key, plan, first, second, memberships = self._catalog_pair(
            displacement_m=20_000.0
        )
        self.assertFalse(promotion._events_associate(first, second, plan))
        self.assertFalse(promotion._events_associate(second, first, plan))
        for events in ((first, second), (second, first)):
            result = promotion.PhysicalEventCatalogResult.from_plan(
                plan,
                event_evidences=events,
                case_spatial_membership_evidences=memberships,
                cataloged_at="2026-08-10T00:30:00Z",
                adjudicator_private_key=key,
            )
            self.assertEqual(len(result.event_evidences), 2)

    def test_disjoint_association_uses_chronology_and_time_gap_limit(self):
        key = Ed25519PrivateKey.from_private_bytes(bytes([0x61]) * 32)
        track_a = promotion.PhysicalEventTrackArtifact(
            timestamps=(
                "2026-08-10T00:00:00Z",
                "2026-08-10T00:10:00Z",
            ),
            centroid_xy_m=((0.0, 0.0), (1_000.0, 0.0)),
            object_mask_digests=("a" * 64, "b" * 64),
            source_radar_ids=("radar", "radar"),
            association_edge_digests=("c" * 64,),
            spatial_reference_digest="d" * 64,
        )

        def event(track, event_id, case_id, x_end):
            return promotion.PhysicalEventCatalogEvidence.from_members(
                event_id=event_id,
                member_case_ids=(case_id,),
                member_full_analysis_input_digests=(event_id * 64,),
                start_time=track.timestamps[0],
                end_time=track.timestamps[-1],
                spatial_envelope_xy_m=(-1.0, -1.0, x_end, 1.0),
                object_track_artifact=track,
                participating_radar_ids=("radar",),
                association_algorithm_digest="2" * 64,
                adjudication_policy_digest="3" * 64,
                adjudicator_id="labeler",
                adjudicator_private_key=key,
            )

        def plan(case_ids):
            return promotion.PhysicalEventCatalogPlan(
                holdout_case_ids=case_ids,
                association_algorithm_digest="2" * 64,
                spatial_membership_rule_digest="6" * 64,
                adjudication_policy_digest="3" * 64,
                adjudicator_id="labeler",
                adjudicator_public_key_hex=promotion.regime_reference_public_key_hex(
                    key
                ),
                catalog_completion_deadline="2026-08-11T01:00:00Z",
                spatial_reference_digest="d" * 64,
                motion_association_rule_digest="7" * 64,
                scheduler_id="scheduler",
                scheduler_public_key_hex=promotion.regime_reference_public_key_hex(
                    Ed25519PrivateKey.from_private_bytes(bytes([0x62]) * 32)
                ),
                scheduler_trust_store_digest="8" * 64,
            )

        for gap_minutes, expected in ((10, True), (31, False)):
            start_minute = 10 + gap_minutes
            start = f"2026-08-10T00:{start_minute:02d}:00Z"
            end = f"2026-08-10T00:{start_minute + 10:02d}:00Z"
            track_b = promotion.PhysicalEventTrackArtifact(
                timestamps=(start, end),
                centroid_xy_m=(
                    (1_000.0 + 100.0 * gap_minutes, 0.0),
                    (2_000.0 + 100.0 * gap_minutes, 0.0),
                ),
                object_mask_digests=("e" * 64, "f" * 64),
                source_radar_ids=("radar", "radar"),
                association_edge_digests=("1" * 64,),
                    spatial_reference_digest="d" * 64,
                )
            first = event(track_a, "a", "case-a", 1_001.0)
            second = event(
                track_b,
                "b",
                "case-b",
                3_001.0 + 100.0 * gap_minutes,
            )
            associated = promotion._events_associate(
                first, second, plan(("case-a", "case-b"))
            )
            reverse = promotion._events_associate(
                second, first, plan(("case-a", "case-b"))
            )
            self.assertEqual(associated, expected)
            self.assertEqual(reverse, expected)

    def test_automatic_ucb_is_invariant_to_unused_bootstrap_resolution(self):
        fixture = promotion_fixtures.NeuralPriorPromotionTests("runTest")
        evaluations = (
            fixture.evaluation(1, -0.2),
            fixture.evaluation(2, -0.3),
        )
        template = fixture.policy()
        results = []
        for bootstrap_samples in (64, 16_384):
            policy = replace(
                template,
                allow_shadow_small_sample_bootstrap=False,
                bootstrap_samples=bootstrap_samples,
                minimum_bootstrap_tail_replicates=20,
                maximum_exact_sign_clusters=1,
            )
            results.append(fixture.compute_with_policy(evaluations, policy))
        low, high = results
        self.assertEqual(low.rejection_reasons, high.rejection_reasons)
        self.assertNotIn(
            "insufficient_bootstrap_tail_resolution",
            low.rejection_reasons,
        )
        self.assertEqual(
            low.simultaneous_inference_method,
            high.simultaneous_inference_method,
        )
        self.assertEqual(
            low.simultaneous_inference_effective_replicates,
            high.simultaneous_inference_effective_replicates,
        )
        self.assertEqual(low.simultaneous_inference_tail_replicates, 0.0)
        self.assertEqual(high.simultaneous_inference_tail_replicates, 0.0)
        self.assertEqual(
            low.prior_support_brier_increase_upper_bound,
            high.prior_support_brier_increase_upper_bound,
        )

    def test_automatic_ucb_does_not_construct_multiplier_samples(self):
        fixture = promotion_fixtures.NeuralPriorPromotionTests("runTest")
        policy = replace(
            fixture.policy(),
            allow_shadow_small_sample_bootstrap=False,
            maximum_exact_sign_clusters=1,
            bootstrap_samples=16_384,
        )
        clusters = tuple(f"event-{index}" for index in range(20))
        comparison = promotion._UncertaintyComparison(
            component="support",
            group=None,
            values=tuple(-0.02 + 0.001 * index for index in range(20)),
            clusters=clusters,
        )
        with patch.object(
            promotion.random,
            "Random",
            side_effect=AssertionError("automatic UCB used bootstrap multipliers"),
        ):
            result = promotion._simultaneous_uncertainty_upper_bounds(
                (comparison,),
                policy,
                candidate_family_size=16,
            )
        self.assertEqual(result.method, "support_bounded_hybrid")
        self.assertFalse(result.randomized_multiplier)


if __name__ == "__main__":
    unittest.main()
