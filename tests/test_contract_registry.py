from dataclasses import fields
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar._contract_registry import (  # noqa: E402
    CONTRACT_CAPABILITIES,
    CURRENT_SEMANTIC_SCORING_REPLAY_CONTRACT,
    CURRENT_VARIATIONAL_FSO_CONTRACT,
    CURRENT_VARIATIONAL_FSOI_CONTRACT,
    CURRENT_VERIFICATION_BUNDLE_CONTRACT,
    current_contract,
    render_contract_capability_table,
)
import advar.promotion as promotion_module  # noqa: E402
from advar.nowcast import RadarMetricDomainEvidence  # noqa: E402
from advar.promotion import (  # noqa: E402
    DeployedNeuralPriorPolicy,
    NeuralPriorPromotionEvidence,
    OperationalDeploymentUnsupportedError,
)


class ContractRegistryTests(unittest.TestCase):
    def test_registry_is_a_small_runtime_generation_authority(self) -> None:
        for capabilities in CONTRACT_CAPABILITIES.values():
            capabilities.validate()

        defaults = (
            (
                RadarMetricDomainEvidence,
                "radar_metric_domain_evidence",
            ),
            (
                NeuralPriorPromotionEvidence,
                "neural_prior_promotion_evidence",
            ),
            (
                DeployedNeuralPriorPolicy,
                "deployed_neural_prior_policy",
            ),
        )
        for artifact_type, family in defaults:
            with self.subTest(family=family):
                contract_default = next(
                    item.default
                    for item in fields(artifact_type)
                    if item.name == "contract"
                )
                self.assertEqual(contract_default, current_contract(family))

        self.assertTrue(
            all(
                not capabilities.operationally_accepted
                for capabilities in CONTRACT_CAPABILITIES.values()
            )
        )

    def test_runtime_current_generations_are_registry_derived(self) -> None:
        self.assertEqual(
            CURRENT_VERIFICATION_BUNDLE_CONTRACT,
            current_contract("verification_bundle"),
        )
        self.assertEqual(
            CURRENT_VARIATIONAL_FSO_CONTRACT,
            current_contract("variational_fso"),
        )
        self.assertEqual(
            CURRENT_VARIATIONAL_FSOI_CONTRACT,
            current_contract("variational_fsoi"),
        )
        self.assertEqual(
            CURRENT_SEMANTIC_SCORING_REPLAY_CONTRACT,
            current_contract("semantic_scoring_replay"),
        )

    def test_readme_capability_table_is_generated_from_registry(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        readme = (repository / "README.md").read_text(encoding="utf-8")
        expected = (
            "<!-- CONTRACT_CAPABILITY_TABLE:START -->\n"
            + render_contract_capability_table()
            + "\n<!-- CONTRACT_CAPABILITY_TABLE:END -->"
        )
        self.assertIn(expected, readme)

    def test_current_package_explicitly_rejects_operational_deployment(
        self,
    ) -> None:
        self.assertFalse(hasattr(promotion_module, "_select_deployed_prior"))
        with self.assertRaisesRegex(
            OperationalDeploymentUnsupportedError,
            "not an operational deployment authorization contract",
        ):
            promotion_module.infer_deployed_neural_prior()


if __name__ == "__main__":
    unittest.main()
