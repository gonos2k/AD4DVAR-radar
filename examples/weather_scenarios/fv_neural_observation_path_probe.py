"""Isolate the NN observation-error path in the fixed FV demonstration."""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path

import torch

from advar import variational as v
import fv_neural_learning_probe as neural
import fv_sensitivity_probe as oracle


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "graphify-out/fv-root-cause-20260919/fv_neural_std.pt"
PRIOR = ROOT / "graphify-out/fv-root-cause-20260919/fv_neural_observation.json"


def run_probe(model_path: str | Path = MODEL) -> dict[str, object]:
    obs, frozen, truth, _ho, _hf, _ht, future, support = neural._windows()
    model = neural.TinyObservationStd()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    params = neural._flat(model).detach()
    y0 = obs.dbz
    features0 = neural._features(obs)
    weights = torch.ones_like(truth)

    def std_observations(features):
        # Raw dBZ, masks, and all frozen analysis data remain nominal.
        return neural._std_observations(obs, params, model, features)

    def objective(control, y):
        features = neural._features(replace(obs, dbz=y))
        return v.robust_objective(control, std_observations(features), frozen)

    def score(control, _y):
        return neural.base._score(control, frozen, future, support, truth, weights)

    nominal = v.solve_analysis(std_observations(features0), frozen)
    control = oracle.polish(
        objective, nominal.control, y0,
        check_step=lambda start, stop: oracle.face_branch_margin(start, stop, frozen),
    )
    response, _direct, adjoint = oracle.stationary_sensitivity(objective, score, control, y0)
    direction = torch.sin(torch.arange(y0.numel(), dtype=y0.dtype)).reshape_as(y0)
    direction /= torch.linalg.vector_norm(direction)
    directional = float((response * direction).sum())
    nominal_faces = oracle.face_signs(control, frozen)
    fd_slopes, fd_errors, margins, endpoint_gradients = [], [], [], []
    same_faces = True
    for step in (0.1, 0.05):
        yp, ym = y0 + step * direction, y0 - step * direction
        rp = v.solve_analysis(std_observations(neural._features(replace(obs, dbz=yp))), frozen, control=control)
        rm = v.solve_analysis(std_observations(neural._features(replace(obs, dbz=ym))), frozen, control=control)
        cp = oracle.polish(objective, rp.control, yp, check_step=lambda a, b: oracle.face_branch_margin(a, b, frozen))
        cm = oracle.polish(objective, rm.control, ym, check_step=lambda a, b: oracle.face_branch_margin(a, b, frozen))
        plus = score(cp, yp); minus = score(cm, ym)
        slope = float((plus - minus) / (2.0 * step))
        fd_slopes.append(slope)
        fd_errors.append(abs(slope - directional))
        margins.extend([oracle.face_branch_margin(control, cp, frozen), oracle.face_branch_margin(control, cm, frozen)])
        endpoint_gradients.extend([
            float(torch.func.grad(objective, argnums=0)(cp, yp).abs().max()),
            float(torch.func.grad(objective, argnums=0)(cm, ym).abs().max()),
        ])
        same_faces &= bool(torch.equal(nominal_faces, oracle.face_signs(cp, frozen)))
        same_faces &= bool(torch.equal(nominal_faces, oracle.face_signs(cm, frozen)))
    prior = json.loads(PRIOR.read_text())["frozen_stats_directional_difference"]
    hessian = oracle.dense_hessian(torch.func.grad(objective, argnums=0), control, y0)
    return {
        "scope": "isolated NN y-to-features-to-std path; raw observations/B/score fixed; research-only",
        "production_learning_eligible": False, "neural_prior_eligible": False,
        "parameter_retuning": False, "control_count": int(control.numel()), "parameter_count": int(params.numel()),
        "stationarity_max": float(torch.func.grad(objective, argnums=0)(control, y0).abs().max()),
        "endpoint_stationarity_max": max(endpoint_gradients),
        "hessian_min_eigenvalue": float(torch.linalg.eigvalsh(hessian).min()),
        "adjoint_relative_residual": float(adjoint), "directional_isolated": directional,
        "finite_difference_steps": [0.1, 0.05], "finite_difference_slopes": fd_slopes,
        "finite_difference_errors": fd_errors, "prior_total_minus_frozen": prior,
        "isolated_minus_prior": directional - prior, "face_branch_margins": margins,
        "same_face_signs": same_faces, "model_path": str(model_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(run_probe(args.model), indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
