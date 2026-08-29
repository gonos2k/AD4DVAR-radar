from collections.abc import Callable
from dataclasses import dataclass, fields
import ast
from pathlib import Path
import subprocess
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar._contract_registry import (  # noqa: E402
    AUDIT_GENERATION_FIXTURES,
    AuditGenerationFixture,
    CONTRACT_CAPABILITIES,
    CURRENT_SEMANTIC_SCORING_REPLAY_CONTRACT,
    CURRENT_VARIATIONAL_FSO_CONTRACT,
    CURRENT_VARIATIONAL_FSOI_CONTRACT,
    CURRENT_VERIFICATION_BUNDLE_CONTRACT,
    FrozenAuditGeneration,
    current_contract,
    render_contract_capability_table,
)
from advar import promotion as promotion_module  # noqa: E402
import advar as advar_module  # noqa: E402
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
    execute: Callable[[], subprocess.CompletedProcess[str]]


def _execute_named_lifecycle_probe(
    probe: str,
) -> subprocess.CompletedProcess[str]:
    """Run one lifecycle in a fresh interpreter so its fixtures cannot leak."""

    repository = Path(__file__).resolve().parents[1]
    return subprocess.run(
        (
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "-q",
            probe,
        ),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )


def _declared_contract_probes() -> tuple[str, ...]:
    probes = [
        probe
        for capabilities in CONTRACT_CAPABILITIES.values()
        for probe in (
            capabilities.lifecycle_probe,
            *(audit_probe for _, audit_probe in capabilities.audit_generation_probes),
        )
    ]
    probes.extend(fixture.decoder_probe for fixture in AUDIT_GENERATION_FIXTURES)
    return tuple(dict.fromkeys(probes))


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
        for probe in _declared_contract_probes():
            path_text, class_name, method_name = probe.split("::")
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
                probe,
            )

    def test_declared_lifecycle_probes_execute_without_skip_or_failure(
        self,
    ) -> None:
        grouped: dict[str, list[str]] = {}
        for family, capabilities in CONTRACT_CAPABILITIES.items():
            grouped.setdefault(capabilities.lifecycle_probe, []).append(family)
            for contract, probe in capabilities.audit_generation_probes:
                grouped.setdefault(probe, []).append(f"{family}:{contract}")
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
                self.assertEqual(
                    result.returncode,
                    0,
                    result.stdout + result.stderr,
                )
                self.assertIn("1 passed", result.stdout)
                self.assertNotIn("skipped", result.stdout.lower())

    def test_every_audit_only_generation_has_an_executable_cold_probe(
        self,
    ) -> None:
        for family, capabilities in CONTRACT_CAPABILITIES.items():
            declared = {
                contract
                for contract, _ in capabilities.audit_generation_probes
            }
            self.assertEqual(
                declared,
                capabilities.audit_readable - capabilities.issuable,
                family,
            )

    def test_frozen_audit_fixture_matrix_exactly_covers_registry(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        registered = {
            (family, contract)
            for family, capabilities in CONTRACT_CAPABILITIES.items()
            for contract in capabilities.audit_readable
        }
        declared = {
            (fixture.family, fixture.contract) for fixture in AUDIT_GENERATION_FIXTURES
        }
        self.assertEqual(declared, registered)
        self.assertEqual(len(declared), len(AUDIT_GENERATION_FIXTURES))
        fixture_directory = repository / "tests/fixtures/audit_generations"
        self.assertEqual(
            {
                path.relative_to(repository).as_posix()
                for path in fixture_directory.glob("*.json")
            },
            {fixture.fixture_path for fixture in AUDIT_GENERATION_FIXTURES},
        )

        for metadata in AUDIT_GENERATION_FIXTURES:
            with self.subTest(
                family=metadata.family,
                contract=metadata.contract,
            ):
                self.assertIsInstance(metadata, AuditGenerationFixture)
                path = repository / metadata.fixture_path
                frozen = FrozenAuditGeneration.from_bytes(
                    path.read_bytes(),
                    metadata,
                )
                self.assertEqual(frozen.family, metadata.family)
                self.assertEqual(frozen.contract, metadata.contract)
                self.assertEqual(frozen.expected_type, metadata.expected_type)
                self.assertEqual(frozen.decoder_probe, metadata.decoder_probe)
                capabilities = CONTRACT_CAPABILITIES[metadata.family]
                self.assertEqual(
                    frozen.scientific_action_allowed,
                    metadata.contract in capabilities.scientific_eligible,
                )
                self.assertEqual(
                    frozen.operational_action_allowed,
                    metadata.contract in capabilities.operationally_accepted,
                )
                self.assertEqual(
                    frozen.payload["fixture_origin"],
                    "registry-cold-audit-minimum-v1",
                )
                if frozen.scientific_action_allowed:
                    frozen.require_scientific_action()
                else:
                    with self.assertRaisesRegex(ValueError, "scientifically"):
                        frozen.require_scientific_action()
                if frozen.operational_action_allowed:
                    frozen.require_operational_action()
                else:
                    with self.assertRaisesRegex(RuntimeError, "operationally"):
                        frozen.require_operational_action()

                raw = path.read_bytes()
                marker = b"registry-cold-audit-minimum-v1"
                self.assertIn(marker, raw)
                tampered = raw.replace(
                    marker,
                    b"registry-cold-audit-minimum-v2",
                    1,
                )
                with self.assertRaisesRegex(ValueError, "digest"):
                    FrozenAuditGeneration.from_bytes(tampered, metadata)

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

    def test_public_audit_exports_cover_every_supported_recent_wrapper(
        self,
    ) -> None:
        for generation in range(21, 27):
            self.assertTrue(
                hasattr(
                    advar_module,
                    f"LegacyScoringReplayBundleManifestAuditV{generation}",
                )
            )
        for generation in range(32, 37):
            self.assertTrue(
                hasattr(
                    advar_module,
                    f"LegacyNeuralPriorHoldoutPlanV{generation}Audit",
                )
            )


if __name__ == "__main__":
    unittest.main()
