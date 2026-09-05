import unittest

import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from advar.intervention import (
    InterventionActionGenerator,
    ProspectiveInterventionDecision,
    RealizedInterventionReceipt,
    ReusableInterventionPolicyEvidence,
)
from test_ledger import _prospective_run_and_context


class _DeweightQcAction(torch.nn.Module):
    def forward(self, context: torch.Tensor):
        return (
            context[1].to(torch.bool),
            context[2] * 0.5,
            torch.tensor(True, device=context.device),
        )


class _RejectOneQcAction(torch.nn.Module):
    def forward(self, context: torch.Tensor):
        valid = context[1].to(torch.bool).clone()
        valid[0, 0, 0] = False
        quality = torch.where(valid, context[2], torch.zeros_like(context[2]))
        return valid, quality, torch.tensor(True, device=context.device)


def _policy(
    generator: InterventionActionGenerator,
    context,
    *,
    policy_id: str,
    quality_limit: float = 4.0,
) -> ReusableInterventionPolicyEvidence:
    return ReusableInterventionPolicyEvidence(
        policy_id=policy_id,
        action_generator_digest=generator.generator_digest,
        context_schema_digest=context.context_schema_digest,
        applicability_region_digest=context.applicability_region_digest,
        execution_policy_digest="e" * 64,
        allowed_intervention_types=("realized_qc_intervention",),
        maximum_absolute_delta_dbz=2.0,
        maximum_changed_fraction=1.0,
        maximum_global_quality_precision_scale_l2=quality_limit,
        maximum_tile_quality_precision_scale_l2=quality_limit,
        validation_evidence_digests=("d" * 64,),
    )


def _decision(policy, generator, context, run, plan, decision_id):
    return ProspectiveInterventionDecision.from_policy(
        policy,
        action_generator=generator,
        decision_id=decision_id,
        case_id="case-1",
        radar_id="radar-1",
        intervention_type="realized_qc_intervention",
        actual_input_context=context,
        actual_input_before_run=run,
        input_plan_digest=plan,
        decision_basis_digest="d" * 64,
        decision_policy_digest="e" * 64,
        decision_trust_store_digest="f" * 64,
        decided_at="2026-08-08T00:22:00Z",
        observation_valid_time="2026-08-08T00:20:00Z",
        input_available_time="2026-08-08T00:21:00Z",
        decision_deadline="2099-08-08T00:30:00Z",
        publication_time="2099-08-08T01:00:00Z",
    )


def _receipt(
    decision,
    policy,
    generator,
    before_context,
    before_run,
    after_context,
    after_run,
):
    return RealizedInterventionReceipt.from_decision(
        decision,
        actual_input_before_context=before_context,
        actual_input_before_run=before_run,
        actual_input_after_context=after_context,
        actual_input_after_run=after_run,
        action_policy=policy,
        action_generator=generator,
        executor_key_id="review-executor",
        executor_trust_store_digest="0" * 64,
        executor_private_key=Ed25519PrivateKey.generate(),
        executor_sequence_number=1,
        applied_time="2026-08-08T00:23:00Z",
        receipt_time="2026-08-08T00:23:00Z",
    )


class InterventionReviewTests(unittest.TestCase):
    def setUp(self):
        self.frames = torch.zeros((3, 2, 2), dtype=torch.float64)
        self.masks = torch.ones_like(self.frames, dtype=torch.bool)
        (
            self.before_run,
            self.before_context,
            self.plan_digest,
            _,
        ) = _prospective_run_and_context(self.frames, self.masks)

    def test_qc_receipt_rejects_active_observation_std_change(self):
        after_run, after_context, _, _ = _prospective_run_and_context(
            self.frames,
            self.masks,
            quality_weight=0.5 * torch.ones_like(self.frames),
            observation_std_dbz=torch.full_like(self.frames, 9.0),
        )
        generator = InterventionActionGenerator.from_model(
            _DeweightQcAction().eval(),
            self.before_context,
            intervention_type="realized_qc_intervention",
            action_reason="deweight",
        )
        policy = _policy(
            generator,
            self.before_context,
            policy_id="reject-active-std",
        )
        decision = _decision(
            policy,
            generator,
            self.before_context,
            self.before_run,
            self.plan_digest,
            "reject-active-std",
        )
        with self.assertRaisesRegex(
            ValueError, "QC receipt changed observation standard deviation"
        ):
            _receipt(
                decision,
                policy,
                generator,
                self.before_context,
                self.before_run,
                after_context,
                after_run,
            )

    def test_qc_receipt_accepts_masked_cell_std_normalization(self):
        after_masks = self.masks.clone()
        after_masks[0, 0, 0] = False
        after_run, after_context, _, _ = _prospective_run_and_context(
            self.frames,
            after_masks,
            quality_weight=after_masks.to(self.frames),
            observation_std_dbz=torch.full_like(self.frames, 2.0),
        )
        generator = InterventionActionGenerator.from_model(
            _RejectOneQcAction().eval(),
            self.before_context,
            intervention_type="realized_qc_intervention",
            action_reason="reject-one",
        )
        policy = _policy(
            generator,
            self.before_context,
            policy_id="accept-mask-normalization",
            quality_limit=1.0,
        )
        decision = _decision(
            policy,
            generator,
            self.before_context,
            self.before_run,
            self.plan_digest,
            "accept-mask-normalization",
        )
        receipt = _receipt(
            decision,
            policy,
            generator,
            self.before_context,
            self.before_run,
            after_context,
            after_run,
        )
        self.assertIsInstance(receipt, RealizedInterventionReceipt)
        self.assertTrue(
            torch.equal(
                self.before_context.observation_std_dbz[after_masks],
                after_context.observation_std_dbz[after_masks],
            )
        )
        self.assertNotEqual(
            self.before_run.observation_std_dbz_digest,
            after_run.observation_std_dbz_digest,
        )

    def test_qc_receipt_accepts_quality_only_change_with_same_std(self):
        after_run, after_context, _, _ = _prospective_run_and_context(
            self.frames,
            self.masks,
            quality_weight=0.5 * torch.ones_like(self.frames),
            observation_std_dbz=torch.full_like(self.frames, 2.0),
        )
        generator = InterventionActionGenerator.from_model(
            _DeweightQcAction().eval(),
            self.before_context,
            intervention_type="realized_qc_intervention",
            action_reason="deweight",
        )
        policy = _policy(
            generator,
            self.before_context,
            policy_id="accept-quality-only",
        )
        decision = _decision(
            policy,
            generator,
            self.before_context,
            self.before_run,
            self.plan_digest,
            "accept-quality-only",
        )
        receipt = _receipt(
            decision,
            policy,
            generator,
            self.before_context,
            self.before_run,
            after_context,
            after_run,
        )
        self.assertIsInstance(receipt, RealizedInterventionReceipt)
        self.assertEqual(
            self.before_run.observation_std_dbz_digest,
            after_run.observation_std_dbz_digest,
        )


if __name__ == "__main__":
    unittest.main()
