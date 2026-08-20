from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from advar.acceptance import (
    AcceptanceArtifactReference,
    REAL_CASE_ACCEPTANCE_SCENARIOS,
    REAL_CASE_ACCEPTANCE_STAGES,
    RealCaseAcceptanceCase,
    RealCaseAcceptanceManifest,
    verify_real_case_acceptance,
)
from advar.promotion import PromotionSampleSizePreflight


class RealCaseAcceptanceTests(unittest.TestCase):
    def fixture(self, root: Path) -> RealCaseAcceptanceManifest:
        preflight = PromotionSampleSizePreflight(
            family_size=1,
            classifier_family_size=1,
            available_physical_events=len(REAL_CASE_ACCEPTANCE_SCENARIOS),
            minimum_structural_events=len(REAL_CASE_ACCEPTANCE_SCENARIOS),
            minimum_perfect_success_events=0,
            minimum_zero_failure_events=0,
            required_physical_events=len(REAL_CASE_ACCEPTANCE_SCENARIOS),
            metric_cell_event_counts=(),
            issuance_cell_event_counts=(),
            classifier_subset_event_counts=(),
            automatic_inference=True,
            cell_feasible=True,
            classifier_subset_feasible=True,
            feasible=True,
        )
        preflight_data = json.dumps(
            preflight.payload | {"preflight_digest": preflight.preflight_digest},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        preflight_path = root / "sample-size-preflight.json"
        preflight_path.write_bytes(preflight_data)
        preflight_path.chmod(0o600)
        cases = []
        for case_index, scenario in enumerate(
            REAL_CASE_ACCEPTANCE_SCENARIOS,
            start=1,
        ):
            references = []
            for stage_index, stage in enumerate(
                REAL_CASE_ACCEPTANCE_STAGES,
                start=1,
            ):
                relative = f"case-{case_index}/{stage_index:02d}-{stage}.json"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                product_digest = sha256(
                    f"product-{case_index}-{stage}".encode()
                ).hexdigest()
                data = json.dumps(
                    {
                        "artifact_digest": product_digest,
                        "case": case_index,
                        "contract": f"{stage}-v1",
                        "stage": stage,
                    },
                    sort_keys=True,
                ).encode()
                path.write_bytes(data)
                path.chmod(0o600)
                references.append(
                    AcceptanceArtifactReference(
                        stage=stage,
                        relative_path=relative,
                        file_sha256=sha256(data).hexdigest(),
                        product_contract=f"{stage}-v1",
                        product_artifact_digest=product_digest,
                    )
                )
            cases.append(
                RealCaseAcceptanceCase(
                    case_id=f"case-{case_index}",
                    scenario=scenario,
                    physical_event_digest=sha256(
                        f"event-{case_index}".encode()
                    ).hexdigest(),
                    sample_size_preflight_digest=preflight.preflight_digest,
                    artifacts=tuple(references),
                )
            )
        return RealCaseAcceptanceManifest(
            created_at="2026-08-18T00:00:00Z",
            sample_size_preflight_relative_path=preflight_path.name,
            sample_size_preflight_file_sha256=sha256(preflight_data).hexdigest(),
            sample_size_preflight_digest=preflight.preflight_digest,
            required_independent_physical_event_count=len(cases),
            cases=tuple(cases),
        )

    def test_complete_matrix_is_report_only_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = self.fixture(root)
            decoded = RealCaseAcceptanceManifest.from_json(manifest.json)
            report = verify_real_case_acceptance(decoded, artifact_root=root)

            self.assertTrue(report["artifact_index_complete"])
            self.assertFalse(report["observation_error_derivation_replayed"])
            self.assertFalse(report["semantic_e2e_validated"])
            self.assertFalse(report["sample_size_satisfied"])
            self.assertFalse(report["eligible_for_scientific_review"])
            self.assertIsNone(report["independent_physical_event_count"])
            self.assertEqual(
                report["declared_event_label_count"],
                len(REAL_CASE_ACCEPTANCE_SCENARIOS),
            )
            self.assertFalse(report["authorizes_deployment"])
            first = root / decoded.cases[0].artifacts[0].relative_path
            first.chmod(0o600)
            first.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify_real_case_acceptance(decoded, artifact_root=root)

    def test_product_digest_cannot_be_injected_outside_the_stage_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = self.fixture(root)
            reference = manifest.cases[0].artifacts[0]
            forged_reference = replace(
                reference,
                product_artifact_digest="f" * 64,
            )
            forged_case = replace(
                manifest.cases[0],
                artifacts=(forged_reference, *manifest.cases[0].artifacts[1:]),
            )
            forged_manifest = replace(
                manifest,
                cases=(forged_case, *manifest.cases[1:]),
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_real_case_acceptance(
                    forged_manifest,
                    artifact_root=root,
                )

    def test_sample_size_requirement_is_replayed_from_the_typed_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = self.fixture(root)
            forged = replace(
                manifest,
                required_independent_physical_event_count=1,
            )
            with self.assertRaisesRegex(ValueError, "preflight identity mismatch"):
                verify_real_case_acceptance(forged, artifact_root=root)

            preflight_path = root / manifest.sample_size_preflight_relative_path
            preflight_path.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "file digest mismatch"):
                verify_real_case_acceptance(manifest, artifact_root=root)

    def test_manifest_rejects_live_mode_and_reused_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.fixture(Path(directory).resolve())
            with self.assertRaisesRegex(ValueError, "report-only"):
                replace(manifest, mode="LIVE")
            duplicate = replace(
                manifest.cases[1],
                physical_event_digest=manifest.cases[0].physical_event_digest,
            )
            with self.assertRaisesRegex(ValueError, "independent physical events"):
                replace(
                    manifest,
                    cases=(manifest.cases[0], duplicate, *manifest.cases[2:]),
                )

    def test_relabelled_events_never_satisfy_scientific_sample_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = self.fixture(root)
            relabelled = replace(
                manifest,
                cases=tuple(
                    replace(
                        case,
                        physical_event_digest=sha256(
                            f"attacker-label-{index}".encode()
                        ).hexdigest(),
                    )
                    for index, case in enumerate(manifest.cases)
                ),
            )
            report = verify_real_case_acceptance(
                relabelled,
                artifact_root=root,
            )
            self.assertFalse(report["semantic_e2e_validated"])
            self.assertFalse(report["sample_size_satisfied"])
            self.assertIsNone(report["independent_physical_event_count"])


if __name__ == "__main__":
    unittest.main()
