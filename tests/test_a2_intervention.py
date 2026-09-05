import unittest

import torch

from advar.intervention import (
    InterventionActionGenerator,
    ProspectiveInterventionDecision,
    ReusableInterventionPolicyEvidence,
)
from test_ledger import _prospective_run_and_context


class _ZeroDbz(torch.nn.Module):
    def forward(self, context: torch.Tensor):
        return torch.zeros_like(context[0]), torch.tensor(
            True, device=context.device
        )


class _NoOpQc(torch.nn.Module):
    def forward(self, context: torch.Tensor):
        return (
            context[1].to(torch.bool),
            context[2],
            torch.tensor(True, device=context.device),
        )


class _SameValueOverride(torch.nn.Module):
    def forward(self, context: torch.Tensor):
        selected = torch.zeros_like(context[0], dtype=torch.bool)
        selected[0, 0, 0] = True
        return context[0], selected, torch.tensor(True, device=context.device)


class _AddOneOverride(torch.nn.Module):
    def forward(self, context: torch.Tensor):
        replacement = context[0].clone()
        replacement[0, 0, 0] = replacement[0, 0, 0] + 1.0
        selected = torch.zeros_like(context[0], dtype=torch.bool)
        selected[0, 0, 0] = True
        return replacement, selected, torch.tensor(True, device=context.device)


class InterventionA2Tests(unittest.TestCase):
    def setUp(self):
        self.frames = torch.zeros((3, 2, 2), dtype=torch.float64)
        self.masks = torch.ones_like(self.frames, dtype=torch.bool)
        (
            self.run,
            self.context,
            self.plan_digest,
            _,
        ) = _prospective_run_and_context(self.frames, self.masks)

    def _policy(self, generator, intervention_type, policy_id):
        return ReusableInterventionPolicyEvidence(
            policy_id=policy_id,
            action_generator_digest=generator.generator_digest,
            context_schema_digest=self.context.context_schema_digest,
            applicability_region_digest=self.context.applicability_region_digest,
            execution_policy_digest="e" * 64,
            allowed_intervention_types=(intervention_type,),
            maximum_absolute_delta_dbz=2.0,
            maximum_changed_fraction=1.0,
            maximum_global_quality_precision_scale_l2=4.0,
            maximum_tile_quality_precision_scale_l2=4.0,
            validation_evidence_digests=("d" * 64,),
        )

    def _decision(self, policy, generator, intervention_type, decision_id):
        return ProspectiveInterventionDecision.from_policy(
            policy,
            action_generator=generator,
            decision_id=decision_id,
            case_id="case-1",
            radar_id="radar-1",
            intervention_type=intervention_type,
            actual_input_context=self.context,
            actual_input_before_run=self.run,
            input_plan_digest=self.plan_digest,
            decision_basis_digest="d" * 64,
            decision_policy_digest="e" * 64,
            decision_trust_store_digest="f" * 64,
            decided_at="2026-08-08T00:22:00Z",
            observation_valid_time="2026-08-08T00:20:00Z",
            input_available_time="2026-08-08T00:21:00Z",
            decision_deadline="2099-08-08T00:30:00Z",
            publication_time="2099-08-08T01:00:00Z",
        )

    def test_prospective_decision_rejects_all_canonical_noops(self):
        cases = (
            ("realized_sensor_correction", _ZeroDbz(), None),
            ("realized_qc_intervention", _NoOpQc(), "no-op QC"),
            ("operator_override", _SameValueOverride(), "same-value override"),
        )
        for index, (intervention_type, model, reason) in enumerate(cases):
            with self.subTest(intervention_type=intervention_type):
                generator = InterventionActionGenerator.from_model(
                    model.eval(),
                    self.context,
                    intervention_type=intervention_type,
                    action_reason=reason,
                )
                policy = self._policy(
                    generator,
                    intervention_type,
                    f"no-op-policy-{index}",
                )
                with self.assertRaisesRegex(ValueError, "no-op"):
                    self._decision(
                        policy,
                        generator,
                        intervention_type,
                        f"no-op-decision-{index}",
                    )

    def test_selected_override_with_real_change_remains_decidable(self):
        generator = InterventionActionGenerator.from_model(
            _AddOneOverride().eval(),
            self.context,
            intervention_type="operator_override",
            action_reason="operator-confirmed artifact",
        )
        policy = self._policy(
            generator,
            "operator_override",
            "changed-override-policy",
        )
        decision = self._decision(
            policy,
            generator,
            "operator_override",
            "changed-override-decision",
        )
        diagnostics = decision.action_safety_diagnostics_json
        self.assertIn('"changed_pixel_count":1', diagnostics)
        self.assertIn('"global_diagonal_standardized_l2":0.5', diagnostics)


if __name__ == "__main__":
    unittest.main()
