from collections.abc import Callable
from dataclasses import dataclass, fields
import ast
import importlib.util
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
from advar import promotion as promotion_module  # noqa: E402
from advar.nowcast import RadarMetricDomainEvidence  # noqa: E402
from advar.promotion import (  # noqa: E402
    DeployedNeuralPriorPolicy,
    NeuralPriorPromotionEvidence,
    OperationalDeploymentUnsupportedError,
)


@dataclass(frozen=True)
class LifecycleCase:
    """One executable construct/round-trip/action probe for contract families."""

    families: tuple[str, ...]
    execute: Callable[[], unittest.TestResult]


def _execute_named_lifecycle_probe(probe: str) -> unittest.TestResult:
    path_text, class_name, method_name = probe.split("::")
    path = Path(__file__).resolve().parents[1] / path_text
    module_name = f"_advar_lifecycle_{path.stem}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load lifecycle probe module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    suite = unittest.defaultTestLoader.loadTestsFromName(
        f"{class_name}.{method_name}",
        module,
    )
    result = unittest.TestResult()
    suite.run(result)
    return result


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

    def test_every_declared_lifecycle_probe_is_discoverable_and_not_skipped(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        parsed_files: dict[Path, ast.Module] = {}
        for capabilities in CONTRACT_CAPABILITIES.values():
            path_text, class_name, method_name = (
                capabilities.lifecycle_probe.split("::")
            )
            path = repository / path_text
            tree = parsed_files.setdefault(
                path,
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
            )
            classes = {
                node.name: node
                for node in tree.body
                if isinstance(node, ast.ClassDef)
            }
            self.assertIn(class_name, classes)
            methods = {
                node.name: node
                for node in classes[class_name].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertIn(method_name, methods)
            decorators = {
                ast.unparse(decorator)
                for decorator in methods[method_name].decorator_list
            }
            self.assertFalse(
                any("skip" in decorator.lower() for decorator in decorators),
                capabilities.lifecycle_probe,
            )

    def test_declared_lifecycle_probes_execute_without_skip_or_failure(
        self,
    ) -> None:
        grouped: dict[str, list[str]] = {}
        for family, capabilities in CONTRACT_CAPABILITIES.items():
            grouped.setdefault(capabilities.lifecycle_probe, []).append(family)
        cases = tuple(
            LifecycleCase(
                families=tuple(families),
                execute=(lambda probe=probe: _execute_named_lifecycle_probe(probe)),
            )
            for probe, families in grouped.items()
        )
        for case in cases:
            with self.subTest(families=case.families):
                result = case.execute()
                self.assertEqual(result.testsRun, 1)
                self.assertFalse(result.skipped)
                self.assertFalse(result.failures)
                self.assertFalse(result.errors)

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
