from dataclasses import fields
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar._contract_registry import (  # noqa: E402
    CONTRACT_CAPABILITIES,
    render_contract_capability_table,
)
from advar import promotion as promotion_module  # noqa: E402
from advar.nowcast import RadarMetricDomainEvidence  # noqa: E402
from advar.promotion import (  # noqa: E402
    DeployedNeuralPriorPolicy,
    NeuralPriorPromotionEvidence,
    OperationalDeploymentUnsupportedError,
)


class ContractRegistryTests(unittest.TestCase):
    def test_registry_capabilities_are_constructible_and_fail_closed(self) -> None:
        for capabilities in CONTRACT_CAPABILITIES.values():
            capabilities.validate()

        metric = CONTRACT_CAPABILITIES["radar_metric_domain_evidence"]
        promotion = CONTRACT_CAPABILITIES["neural_prior_promotion_evidence"]
        policy = CONTRACT_CAPABILITIES["deployed_neural_prior_policy"]
        self.assertEqual(
            next(
                item.default
                for item in fields(RadarMetricDomainEvidence)
                if item.name == "contract"
            ),
            metric.current,
        )
        self.assertEqual(
            next(
                item.default
                for item in fields(NeuralPriorPromotionEvidence)
                if item.name == "contract"
            ),
            promotion.current,
        )
        self.assertEqual(
            next(
                item.default
                for item in fields(DeployedNeuralPriorPolicy)
                if item.name == "contract"
            ),
            policy.current,
        )
        self.assertFalse(metric.operationally_accepted)
        self.assertFalse(promotion.operationally_accepted)
        self.assertFalse(policy.operationally_accepted)

    def test_readme_capability_table_is_generated_from_registry(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        readme = (repository / "README.md").read_text(encoding="utf-8")
        expected = (
            "<!-- CONTRACT_CAPABILITY_TABLE:START -->\n"
            + render_contract_capability_table()
            + "\n<!-- CONTRACT_CAPABILITY_TABLE:END -->"
        )
        self.assertIn(expected, readme)

    def test_current_package_exposes_only_explicit_unsupported_deployment(
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
