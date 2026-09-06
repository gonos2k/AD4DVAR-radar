"""Focused regressions for the A5 artifact and intervention boundaries."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from advar._digest import tensor_digest
from advar.acceptance import _read_artifact_snapshot
from advar.intervention import (
    DbzCorrectionAction,
    InterventionActionGenerator,
    InterventionInputContext,
    ProspectiveInterventionDecision,
    ReusableInterventionPolicyEvidence,
    _canonical_action_input_state,
    _run_uses_correlated_observation_error,
)
from advar.ledger import (
    EpisodeLedger,
    _artifact_intervention_context,
    _durable_intervention_source_mask,
)
from advar.linearization_artifact import (
    _artifact_digest,
    load_p1_linearization,
    save_p1_linearization,
)
from advar.nowcast import ForecastRunContract
from advar.range_geometry import RangeGeometryContract, resolve_range_geometry

from test_ledger import _prospective_run_and_context
from test_variational import _prior
from advar.variational import AnalysisConfig, variational_nowcast


class _AddOneDbz(torch.nn.Module):
    def forward(self, context: torch.Tensor):
        delta = torch.zeros_like(context[0])
        delta[0, 0, 0] = 0.1
        return delta, torch.tensor(True, device=context.device)


def _source_run(base_run: ForecastRunContract, frames: torch.Tensor, source: torch.Tensor):
    return ForecastRunContract.from_inputs(
        base_run.config,
        frames,
        torch.ones_like(frames, dtype=torch.bool),
        None,
        observation_quality_weight=torch.zeros_like(frames),
        observation_std_dbz=torch.full_like(frames, 2.0),
        source_available_mask=source,
        grid_time_contract=base_run.grid_time_contract,
        analysis_config_json=base_run.analysis_config_json,
        analysis_config_digest=base_run.analysis_config_digest,
        analysis_input_digest=base_run.analysis_input_digest,
        operational_calibration_manifest_json=(
            base_run.operational_calibration_manifest_json
        ),
        operational_calibration_manifest_digest=(
            base_run.operational_calibration_manifest_digest
        ),
        operational_calibration_approval_digest=(
            base_run.operational_calibration_approval_digest
        ),
        operational_data_identity_json=base_run.operational_data_identity_json,
        operational_data_identity_digest=base_run.operational_data_identity_digest,
        input_plan_json=base_run.input_plan_json,
        input_plan_digest=base_run.input_plan_digest,
    )


def _dbz_policy(generator: InterventionActionGenerator, context: InterventionInputContext):
    return ReusableInterventionPolicyEvidence(
        policy_id="a5-source-policy",
        action_generator_digest=generator.generator_digest,
        context_schema_digest=context.context_schema_digest,
        applicability_region_digest=context.applicability_region_digest,
        execution_policy_digest="e" * 64,
        allowed_intervention_types=("realized_sensor_correction",),
        maximum_absolute_delta_dbz=1.0,
        maximum_changed_fraction=1.0,
        validation_evidence_digests=("d" * 64,),
    )


def _decision(policy, generator, context, run, plan_digest):
    return ProspectiveInterventionDecision.from_policy(
        policy,
        action_generator=generator,
        decision_id="a5-source-decision",
        case_id="case-1",
        radar_id="radar-1",
        intervention_type="realized_sensor_correction",
        actual_input_context=context,
        actual_input_before_run=run,
        input_plan_digest=plan_digest,
        decision_basis_digest="d" * 64,
        decision_policy_digest="e" * 64,
        decision_trust_store_digest="f" * 64,
        decided_at="2026-08-08T00:22:00Z",
        observation_valid_time="2026-08-08T00:20:00Z",
        input_available_time="2026-08-08T00:21:00Z",
        decision_deadline="2099-08-08T00:30:00Z",
        publication_time="2099-08-08T01:00:00Z",
    )


def test_source_availability_is_required_and_blocks_zero_effect_dbz_action():
    frames = torch.zeros((3, 2, 2), dtype=torch.float64)
    masks = torch.ones_like(frames, dtype=torch.bool)
    base_run, _, plan_digest, _ = _prospective_run_and_context(frames, masks)
    source = torch.zeros_like(masks)
    run = _source_run(base_run, frames, source)

    with pytest.raises(ValueError, match="source availability is required"):
        InterventionInputContext.from_inputs(
            frames_dbz=frames,
            observation_masks=masks,
            quality_weight=torch.zeros_like(frames),
            observation_std_dbz=torch.full_like(frames, 2.0),
            background_frames_dbz=None,
            radar_id="radar-1",
            applicability_mask=torch.ones_like(masks),
            run=run,
        )

    context = InterventionInputContext.from_inputs(
        frames_dbz=frames,
        observation_masks=masks,
        quality_weight=torch.zeros_like(frames),
        observation_std_dbz=torch.full_like(frames, 2.0),
        background_frames_dbz=None,
        radar_id="radar-1",
        applicability_mask=torch.ones_like(masks),
        run=run,
        source_available_mask=source,
    )
    assert not bool(torch.any(context.effective_observation_mask))
    assert torch.all(context.generator_tensor()[1] == 0.0)
    generator = InterventionActionGenerator.from_model(
        _AddOneDbz().eval(), context, intervention_type="realized_sensor_correction"
    )
    with pytest.raises(ValueError, match="source-unavailable"):
        _decision(_dbz_policy(generator, context), generator, context, run, plan_digest)


def test_source_backed_context_keeps_existing_positive_dbz_decision():
    frames = torch.zeros((3, 2, 2), dtype=torch.float64)
    masks = torch.ones_like(frames, dtype=torch.bool)
    run, context, plan_digest, _ = _prospective_run_and_context(frames, masks)
    source = torch.ones_like(masks)
    context = InterventionInputContext.from_inputs(
        frames_dbz=frames,
        observation_masks=masks,
        quality_weight=masks.to(frames),
        observation_std_dbz=torch.full_like(frames, 2.0),
        background_frames_dbz=None,
        radar_id="radar-1",
        applicability_mask=masks,
        run=run,
        source_available_mask=source,
    )
    generator = InterventionActionGenerator.from_model(
        _AddOneDbz().eval(), context, intervention_type="realized_sensor_correction"
    )
    decision = _decision(
        _dbz_policy(generator, context), generator, context, run, plan_digest
    )
    assert decision.intervention_type == "realized_sensor_correction"


@pytest.mark.parametrize("value", ["1.0", None, True, -1.0])
def test_present_malformed_common_bias_is_rejected(value):
    run = SimpleNamespace(
        analysis_config_json=json.dumps(
            {"observation_common_bias_std_dbz": value},
            allow_nan=True,
        )
    )
    with pytest.raises(ValueError, match="analysis config is invalid"):
        _run_uses_correlated_observation_error(run)


def test_common_bias_guard_accepts_absent_or_zero_and_rejects_positive():
    assert not _run_uses_correlated_observation_error(
        SimpleNamespace(analysis_config_json="{}")
    )
    assert not _run_uses_correlated_observation_error(
        SimpleNamespace(
            analysis_config_json='{"observation_common_bias_std_dbz":0.0}'
        )
    )
    assert _run_uses_correlated_observation_error(
        SimpleNamespace(
            analysis_config_json='{"observation_common_bias_std_dbz":1.0}'
        )
    )


def test_single_site_range_resolver_validates_contract_before_use():
    x = torch.tensor([[0.0, 100.0], [0.0, 100.0]])
    y = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
    contract = RangeGeometryContract(
        radar_site_digest="a" * 64,
        radar_site_location_digest="b" * 64,
        grid_contract_digest="c" * 64,
        radar_x_m=0.0,
        radar_y_m=0.0,
        range_regime_labels=("near", "far"),
        radial_distance_edges_m=(0.0, 120.0, 300.0),
        horizontal_range_rule_digest="d" * 64,
        grid_x_m_digest=tensor_digest(x),
        grid_y_m_digest=tensor_digest(y),
    )
    partition = resolve_range_geometry(contract, grid_x_m=x, grid_y_m=y)
    assert int(partition.mask("near").sum()) == 3
    object.__setattr__(contract, "radar_x_m", 100.0)
    with pytest.raises(ValueError, match="contract digest mismatch"):
        resolve_range_geometry(contract, grid_x_m=x, grid_y_m=y)


def test_p1_loader_rejects_conflicting_duplicate_acceptance_flag(tmp_path):
    frames = torch.full((3, 4, 4), 20.0, dtype=torch.float64)
    application = _prior(frames, 1.0, "candidate")
    config = AnalysisConfig(
        final_linearization_relative_stationarity_tolerance=1.0e9,
        final_robust_relative_stationarity_tolerance=1.0e9,
        final_field_gradient_max_tolerance=1.0e9,
        final_irls_relative_weight_tolerance=1.0e9,
    )
    _, analysis = variational_nowcast(
        frames,
        analysis_config=config,
        neural_prior=application,
    )
    assert analysis.linearization is not None
    path = tmp_path / "p1.npz"
    save_p1_linearization(analysis, path)
    assert load_p1_linearization(path).outer_converged is False
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload = json.loads(str(values["payload_json"].item()))
    payload["state"]["fields"]["final_linearization_stationary"] = False
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    tensors = {
        name: value for name, value in values.items() if name.startswith("tensor_")
    }
    values["payload_json"] = np.asarray(payload_json)
    values["artifact_digest"] = np.asarray(
        _artifact_digest("p1-linearization-v15", payload_json, tensors)
    )
    with path.open("wb") as stream:
        np.savez(stream, **values)
    with pytest.raises(ValueError, match="acceptance flags"):
        load_p1_linearization(path)


def _legacy_artifact_context_manifest() -> dict[str, object]:
    digest = "a" * 64
    return {
        "before_radar_id": "radar-1",
        "before_input_bundle_digest": digest,
        "input_plan_digest": digest,
        "before_input_plan_resolution_digest": digest,
        "before_analysis_input_identity_digest": digest,
        "before_context_schema_digest": digest,
        "before_applicability_region_digest": digest,
        "before_applicability_mask_digest": digest,
        "before_canonicalization_contract_digest": digest,
        "minimum_dbz": 0.0,
        "maximum_dbz": 80.0,
        "missing_fill_dbz": 0.0,
        "before_context_digest": digest,
        "before_source_available_mask_digest": None,
    }


def _legacy_artifact_context_tensors() -> dict[str, torch.Tensor]:
    shape = (2, 2, 2)
    return {
        "before_frames": torch.full(shape, 20.0),
        "before_masks": torch.ones(shape, dtype=torch.bool),
        "before_quality": torch.ones(shape),
        "before_std": torch.ones(shape),
        "before_applicability": torch.ones(shape, dtype=torch.bool),
    }


def test_durable_source_mask_rebuild_rejects_ambiguous_current_archive():
    tensors = _legacy_artifact_context_tensors()
    source = torch.zeros_like(tensors["before_masks"])
    manifest = {
        "before_source_available_mask_digest": tensor_digest(source),
    }
    with pytest.raises(ValueError, match="source availability is not retained"):
        _durable_intervention_source_mask(manifest, tensors, prefix="before")
    with pytest.raises(ValueError, match="source availability is not retained"):
        _artifact_intervention_context(manifest, tensors, None)


def test_legacy_archive_rebuilds_all_source_mask_and_action_state():
    tensors = _legacy_artifact_context_tensors()
    manifest = _legacy_artifact_context_manifest()
    context = _artifact_intervention_context(
        manifest,
        tensors,
        None,
    )
    assert torch.all(context.source_available_mask)
    frames, masks, quality = _canonical_action_input_state(
        DbzCorrectionAction(torch.full_like(tensors["before_frames"], 0.1)),
        context,
    )
    assert torch.all(masks)
    assert torch.equal(quality, tensors["before_quality"])
    assert torch.allclose(frames, tensors["before_frames"] + 0.1)
    manifest["before_source_available_mask_digest"] = tensor_digest(
        tensors["before_masks"]
    )
    all_source_context = _artifact_intervention_context(manifest, tensors, None)
    assert torch.all(all_source_context.source_available_mask)


def test_durable_append_rejects_nontrivial_source_before_publishing(tmp_path):
    frames = torch.zeros((3, 2, 2), dtype=torch.float64)
    masks = torch.ones_like(frames, dtype=torch.bool)
    base_run, _, _, _ = _prospective_run_and_context(frames, masks)
    source = torch.zeros_like(masks)
    run = _source_run(base_run, frames, source)
    context = InterventionInputContext.from_inputs(
        frames_dbz=frames,
        observation_masks=masks,
        quality_weight=torch.zeros_like(frames),
        observation_std_dbz=torch.full_like(frames, 2.0),
        background_frames_dbz=None,
        radar_id="radar-1",
        applicability_mask=masks,
        run=run,
        source_available_mask=source,
    )
    ledger = SimpleNamespace(interventions_dir=tmp_path)
    receipt = SimpleNamespace(receipt_digest="b" * 64)
    with pytest.raises(ValueError, match="source availability is not retained"):
        EpisodeLedger._write_intervention_action_artifact(
            ledger,
            None,
            receipt,
            None,
            None,
            context,
            run,
            context,
            run,
        )
    assert tuple(tmp_path.iterdir()) == ()


def test_acceptance_non_directory_root_normalizes_open_error(tmp_path):
    root = tmp_path / "artifact-root"
    root.write_bytes(b"not a directory")
    root.chmod(0o600)
    with pytest.raises(ValueError, match="cannot be opened safely"):
        _read_artifact_snapshot(root, "artifact.json", maximum_bytes=100)


def test_acceptance_fifo_final_component_is_rejected_without_blocking(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO is unavailable on this platform")
    root = tmp_path / "artifact-root"
    root.mkdir()
    os.mkfifo(root / "artifact.json")
    with pytest.raises(ValueError, match="regular and non-writable"):
        _read_artifact_snapshot(root, "artifact.json", maximum_bytes=100)
