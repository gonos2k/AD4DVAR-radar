"""Bounded central check for the saved 240x240 FV observation response.

This probe reuses the nominal exact response and the matched positive h=.0005
checkpoint.  It refines only the reflected negative perturbation and fails
closed on stale source, response, or positive-checkpoint artifacts.
"""

from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import time

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "weather_scenarios"))
import fv_rotation_demo as demo  # noqa: E402

from advar.fv_sensitivity import face_branch_margin, refine_fv_stationarity  # noqa: E402
from advar.physics import echo_to_dbz  # noqa: E402
from advar.variational import forecast_fv_analysis, robust_objective  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "graphify-out/fv-root-cause-20260919"
LEADS = 18
STEP = 0.0005
REPORT = HERE / "rotation240_stable_response_18.json"
RESPONSE = HERE / "rotation240_stable_response_18.pt"
REFINED = HERE / "rotation240_stable_refined.pt"
POSITIVE = HERE / "rotation240_stable_impact_0.0005.pt"
NEGATIVE = HERE / "rotation240_stable_central_negative_0.0005.pt"
OUTPUT = HERE / "rotation240_stable_central_0.0005.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "src/advar/fv_sensitivity.py",
        ROOT / "src/advar/variational.py",
        ROOT / "src/advar/transport.py",
        ROOT / "examples/weather_scenarios/fv_large_response_probe.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _load_and_validate_inputs() -> tuple[dict, dict, dict, dict, dict]:
    report = json.loads(REPORT.read_text())
    if report.get("leads") != LEADS or report.get("curvature") != "exact_robust_hessian":
        raise ValueError("saved response report has the wrong lead or curvature contract")
    expected_sources = report.get("impact_source_hashes")
    current_sources = _source_hashes()
    if expected_sources != current_sources:
        raise ValueError("current solver/probe sources do not match saved response")
    if _sha256(RESPONSE) != report.get("response_tensor_sha256"):
        raise ValueError("response tensor hash does not match saved report")
    if _sha256(REFINED) != report.get("refined_checkpoint_sha256"):
        raise ValueError("refined checkpoint hash does not match saved report")
    if not POSITIVE.exists():
        raise FileNotFoundError("matched positive h=.0005 checkpoint is not available")

    nominal = torch.load(REFINED, weights_only=False)
    response = torch.load(RESPONSE, weights_only=True)
    positive = torch.load(POSITIVE, weights_only=True)
    if not torch.equal(nominal["control"], response["control"]):
        raise ValueError("response cache does not belong to nominal control")
    if positive.get("source_hashes") != expected_sources:
        raise ValueError("positive checkpoint has stale source hashes")
    if not torch.equal(positive["base_control"], nominal["control"]):
        raise ValueError("positive checkpoint has a different base control")
    return report, nominal, response, positive, current_sources


def _direction(observations: object) -> torch.Tensor:
    dbz = observations.dbz
    height, width = dbz.shape[-2:]
    yy = torch.arange(height, dtype=dbz.dtype)[:, None] / height
    xx = torch.arange(width, dtype=dbz.dtype)[None, :] / width
    spatial = 0.5 + 0.5 * torch.cos(2 * torch.pi * xx) * torch.cos(2 * torch.pi * yy)
    return torch.tensor((0.5, 0.75, 1.0), dtype=dbz.dtype)[:, None, None] * spatial


def _validate_observation_contract(observations: object, frozen: object, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError("negative observation perturbation is nonfinite")
    detected = observations.valid_mask & observations.detected_mask
    config = frozen.analysis_config
    nowcast = frozen.nowcast_config
    interior = (
        (value > config.detection_limit_dbz)
        & (value > nowcast.min_dbz)
        & (value < nowcast.max_dbz)
    )
    if bool(detected.any()) and not bool(interior[detected].all()):
        raise ValueError("negative perturbation leaves detected-observation bounds")
    if frozen.initial_support_mask.dtype is not torch.bool:
        raise ValueError("negative perturbation has a non-boolean support contract")


def _score(control: torch.Tensor, frozen: object, observations: object, boundary: object, support: object, truth: torch.Tensor, weights: torch.Tensor) -> float:
    with torch.no_grad():
        trajectory = forecast_fv_analysis(
            control,
            frozen,
            leads=LEADS,
            boundary_start_interval=2,
            boundary_echo=boundary,
            boundary_support=support,
        )
        prediction = echo_to_dbz(trajectory.frames_linear[1:], min_dbz=demo.MIN_DBZ)
        return float((weights * (prediction - truth).square()).sum() / weights.sum())


def run() -> dict:
    started = time.perf_counter()
    report, nominal, response, positive, source_hashes = _load_and_validate_inputs()
    control = nominal["control"]
    observations = nominal["observations"]
    frozen = nominal["frozen"]
    boundary = demo._boundary_schedule(0.0, LEADS)
    support = demo._support_schedule(LEADS)
    truth = echo_to_dbz(
        torch.stack([demo._q_field((i + 1) * 600.0) for i in range(LEADS)]),
        min_dbz=demo.MIN_DBZ,
    )
    weights = torch.ones_like(truth)
    direction = _direction(observations)
    slope = float((response["sensitivity"] * direction).sum())
    y_plus = observations.dbz + STEP * direction
    y_minus = observations.dbz - STEP * direction
    if not torch.equal(positive["observations"], y_plus):
        raise ValueError("positive checkpoint is not the requested h=.0005 perturbation")
    _validate_observation_contract(observations, frozen, y_minus)

    def contract(value: torch.Tensor) -> object:
        return replace(frozen, initial_background_dbz=value[0], input_frames_dbz=value)

    positive_frozen = contract(y_plus)
    positive_control = positive["control"]
    positive_margin = face_branch_margin(control, positive_control, frozen)
    positive_changed = replace(observations, dbz=y_plus)
    positive_gradient = torch.func.grad(robust_objective)(
        positive_control, positive_changed, positive_frozen
    )
    if (
        not bool(torch.isfinite(positive_gradient).all())
        or float(positive_gradient.abs().max()) > 1.0e-8
    ):
        raise ValueError("positive checkpoint is not a stationarity-gated final control")

    negative_start = 2.0 * control - positive_control
    negative_changed = replace(observations, dbz=y_minus)
    negative_frozen = contract(y_minus)

    def save_step(value: torch.Tensor, record: dict[str, float | int]) -> None:
        torch.save(
            {
                "control": value.detach().clone(),
                "base_control": control,
                "observations": y_minus,
                "source_hashes": source_hashes,
                "last_step": record,
            },
            NEGATIVE,
        )

    resumed = False
    if NEGATIVE.exists():
        previous = torch.load(NEGATIVE, weights_only=True)
        if (
            previous.get("source_hashes") != source_hashes
            or not torch.equal(previous["base_control"], control)
            or not torch.equal(previous["observations"], y_minus)
        ):
            raise ValueError("negative checkpoint belongs to a different problem")
        negative_start = previous["control"]
        resumed = True
    negative_margin = face_branch_margin(control, negative_start, frozen)
    negative_started = time.perf_counter()
    negative_control, records = refine_fv_stationarity(
        negative_start,
        negative_changed,
        negative_frozen,
        gradient_tolerance=1.0e-8,
        maximum_iterations=4,
        maximum_normal_products=64,
        on_step=save_step,
    )
    save_step(
        negative_control,
        records[-1] if records else {"initial": True},
    )
    negative_seconds = time.perf_counter() - negative_started

    nominal_score = _score(control, frozen, observations, boundary, support, truth, weights)
    plus_score = _score(positive_control, positive_frozen, positive_changed, boundary, support, truth, weights)
    minus_score = _score(negative_control, negative_frozen, negative_changed, boundary, support, truth, weights)
    plus_change = plus_score - nominal_score
    minus_change = minus_score - nominal_score
    negative_gradient = torch.func.grad(robust_objective)(
        negative_control, negative_changed, negative_frozen
    )
    central = (plus_change - minus_change) / (2.0 * STEP)
    result = {
        "scope": "240x240 exact local central response check; fixed analytic boundaries and verification",
        "step": STEP,
        "leads": LEADS,
        "directional_slope": slope,
        "nominal_score": nominal_score,
        "positive": {
            "actual_change": plus_change,
            "linear_prediction": STEP * slope,
            "taylor_error": abs(plus_change - STEP * slope),
            "gradient_max": float(positive_gradient.abs().max()),
            "gradient_norm": float(torch.linalg.vector_norm(positive_gradient)),
            "face_margin": positive_margin,
        },
        "negative": {
            "actual_change": minus_change,
            "linear_prediction": -STEP * slope,
            "taylor_error": abs(minus_change + STEP * slope),
            "gradient_max": float(negative_gradient.abs().max()),
            "gradient_norm": float(torch.linalg.vector_norm(negative_gradient)),
            "face_margin": face_branch_margin(control, negative_control, frozen),
            "start_face_margin": negative_margin,
            "resumed": resumed,
            "refinement": records,
        },
        "central_slope": central,
        "central_slope_error": abs(central - slope),
        "response_tensor_sha256": _sha256(RESPONSE),
        "refined_checkpoint_sha256": _sha256(REFINED),
        "positive_checkpoint_sha256": _sha256(POSITIVE),
        "negative_checkpoint_sha256": _sha256(NEGATIVE),
        "central_probe_sha256": _sha256(Path(__file__)),
        "source_hashes": source_hashes,
        "negative_refinement_seconds": negative_seconds,
        "wall_seconds": time.perf_counter() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), indent=2))
