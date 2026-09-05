from dataclasses import replace

import pytest
import torch

import advar.variational as variational
from advar.physics import RemapCell, remap
from advar.range_geometry import (
    RangeGeometryContract,
    RangePartitionEvidence,
    resolve_range_geometry,
)
from advar._digest import tensor_digest


def test_extreme_finite_remap_is_zero_and_has_finite_ad_paths() -> None:
    for dtype in (torch.float32, torch.float64):
        echo = torch.arange(4, dtype=dtype).reshape(2, 2)
        displacement = torch.tensor((1.0e20, -1.0e20), dtype=dtype)
        tangent_echo = torch.ones_like(echo)
        tangent_displacement = torch.ones_like(displacement)

        def function(field: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
            return remap(field, motion)

        output, tangent = torch.func.jvp(
            function,
            (echo, displacement),
            (tangent_echo, tangent_displacement),
        )
        _, pullback = torch.func.vjp(function, echo, displacement)
        echo_gradient, displacement_gradient = pullback(torch.ones_like(output))
        echo_for_grad = echo.clone().requires_grad_()
        displacement_for_grad = displacement.clone().requires_grad_()
        function(echo_for_grad, displacement_for_grad).sum().backward()

        torch.testing.assert_close(output, torch.zeros_like(output))
        torch.testing.assert_close(tangent, torch.zeros_like(tangent))
        torch.testing.assert_close(echo_gradient, torch.zeros_like(echo))
        torch.testing.assert_close(
            displacement_gradient,
            torch.zeros_like(displacement),
        )
        torch.testing.assert_close(echo_for_grad.grad, torch.zeros_like(echo))
        torch.testing.assert_close(
            displacement_for_grad.grad,
            torch.zeros_like(displacement),
        )


def test_canonical_observations_are_finite_without_binding_frozen_state() -> None:
    frames = torch.full((3, 3, 3), 20.0, dtype=torch.float64)
    frames[0, 0, 0] = torch.nan
    frames[1, 0, 1] = torch.inf
    observations, frozen = variational.prepare_analysis(frames)
    assert bool(torch.all(torch.isfinite(observations.dbz)))

    control = variational.initial_control(frozen)
    changed = replace(observations, dbz=observations.dbz + 0.25)
    residual = variational.observation_residual_dbz(control, changed, frozen)
    assert bool(torch.all(torch.isfinite(residual)))

    malformed = replace(observations, dbz=observations.dbz.clone())
    malformed.dbz[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="dbz must be finite"):
        variational.observation_residual_dbz(control, malformed, frozen)


def test_retained_linearization_rejects_control_inconsistent_remap_cells() -> None:
    frames = torch.stack(
        [torch.full((3, 3), value, dtype=torch.float64) for value in (20.0, 19.0, 18.0)]
    )
    config = variational.AnalysisConfig(
        censored_background_policy="detection_limit",
        maximum_outer_iterations=8,
        maximum_pcg_iterations=100,
        pcg_relative_tolerance=1.0e-8,
    )
    observations, frozen = variational.prepare_analysis(
        frames,
        analysis_config=config,
    )
    result = variational.solve_analysis(observations, frozen)
    assert result.linearization is not None
    linearization = result.linearization
    variational.validate_analysis_linearization_content(
        result.control,
        linearization,
        require_current_environment=False,
    )

    retained_cells = linearization.frozen.analysis_remap_cells
    wrong_frozen = replace(
        linearization.frozen,
        analysis_remap_cells=(
            retained_cells[0],
            RemapCell(retained_cells[1].y - 1, retained_cells[1].x),
        ),
    )
    forged = variational._content_address_linearization(
        result.control,
        replace(linearization, frozen=wrong_frozen),
    )
    with pytest.raises(ValueError, match="remap cells disagree"):
        variational.validate_analysis_linearization_content(
            result.control,
            forged,
            require_current_environment=False,
        )


def test_range_partition_evidence_requires_nonempty_unique_labels() -> None:
    digest = "a" * 64
    grid_x = torch.tensor([[0.0, 3.0, 5.0]], dtype=torch.float64)
    grid_y = torch.zeros_like(grid_x)
    contract = RangeGeometryContract(
        radar_site_digest=digest,
        radar_site_location_digest="b" * 64,
        grid_contract_digest="c" * 64,
        radar_x_m=0.0,
        radar_y_m=0.0,
        range_regime_labels=("near", "far"),
        radial_distance_edges_m=(0.0, 3.0, 5.0),
        horizontal_range_rule_digest="d" * 64,
        grid_x_m_digest=tensor_digest(grid_x),
        grid_y_m_digest=tensor_digest(grid_y),
    )
    partition = resolve_range_geometry(
        contract,
        grid_x_m=grid_x,
        grid_y_m=grid_y,
    )

    for labels in (("same", "same"), ("", "far")):
        with pytest.raises(ValueError, match="range partition evidence is invalid"):
            RangePartitionEvidence(
                range_geometry_contract_digest=contract.contract_digest,
                grid_contract_digest=contract.grid_contract_digest,
                range_regime_labels=labels,
                masks=partition.masks,
                valid_range_domain_mask=partition.valid_range_domain_mask,
                range_band_mask_digests=partition.range_band_mask_digests,
                valid_range_domain_mask_digest=partition.valid_range_domain_mask_digest,
                active_range_regimes=tuple(
                    label
                    for label, mask in zip(labels, partition.masks, strict=True)
                    if bool(torch.any(mask))
                ),
            )
