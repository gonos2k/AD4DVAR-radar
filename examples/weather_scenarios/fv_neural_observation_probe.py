"""Total observation-input response of the fixed tiny neural std model.

Research evidence only: the trained NN parameters are loaded and held fixed;
this probe does not retrain the model or change FV neural-prior eligibility.
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path

import torch

from advar import variational as v
from advar.physics import echo_to_dbz
import fv_learning_probe as base
import fv_neural_learning_probe as neural
import fv_sensitivity_probe as oracle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "graphify-out/fv-root-cause-20260919/fv_neural_std.pt"


def run_probe(model_path: str | Path = DEFAULT_MODEL) -> dict[str, object]:
    train_obs, train_frozen, truth, _hold_obs, _hold_frozen, _hold_truth, future, support = neural._windows()
    model = neural.TinyObservationStd()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    params = neural._flat(model).detach()
    y0 = train_obs.dbz
    frozen_features = neural._features(train_obs)
    weights = torch.ones_like(truth)

    def contract(y):
        return replace(train_frozen, input_frames_dbz=y, initial_background_dbz=y[0])

    def observations(y, features):
        return neural._std_observations(replace(train_obs, dbz=y), params, model, features)

    def objective(c, y, features=frozen_features):
        return v.robust_objective(c, observations(y, features), contract(y))

    def forecast(c, y):
        trajectory = v.forecast_fv_analysis(
            c, contract(y), leads=2, boundary_start_interval=2,
            boundary_echo=future, boundary_support=support,
        )
        return echo_to_dbz(trajectory.frames_linear[1:], min_dbz=-10.0)

    def score(c, y):
        prediction = forecast(c, y)
        return (weights * (prediction - truth).square()).sum() / weights.sum()

    # One public solve supplies the nominal control; all endpoint solves are
    # ordinary public reanalyses warm-started from that fixed nominal state.
    nominal = v.solve_analysis(observations(y0, neural._features(train_obs)), train_frozen)
    control = oracle.polish(
        lambda c, y: objective(c, y), nominal.control, y0,
        check_step=lambda start, stop: oracle.face_branch_margin(start, stop, train_frozen),
    )
    total_objective = lambda c, y: objective(c, y, neural._features(replace(train_obs, dbz=y)))
    frozen_objective = lambda c, y: objective(c, y, frozen_features)
    response, _direct, adjoint = oracle.stationary_sensitivity(total_objective, score, control, y0)
    frozen_response, _frozen_direct, frozen_adjoint = oracle.stationary_sensitivity(
        frozen_objective, score, control, y0
    )
    direction = torch.sin(torch.arange(y0.numel(), dtype=y0.dtype)).reshape_as(y0)
    direction /= torch.linalg.vector_norm(direction)
    directional = float((response * direction).sum())
    frozen_directional = float((frozen_response * direction).sum())
    finite_difference = []
    taylor_error = []
    endpoint_gradients = []
    same_face = True
    for step in (1.0e-3, 5.0e-4):
        yp, ym = y0 + step * direction, y0 - step * direction
        rp = v.solve_analysis(observations(yp, neural._features(replace(train_obs, dbz=yp))), contract(yp), control=control)
        rm = v.solve_analysis(observations(ym, neural._features(replace(train_obs, dbz=ym))), contract(ym), control=control)
        cp = oracle.polish(lambda c, y: total_objective(c, y), rp.control, yp, check_step=lambda a, b: oracle.face_branch_margin(a, b, train_frozen))
        cm = oracle.polish(lambda c, y: total_objective(c, y), rm.control, ym, check_step=lambda a, b: oracle.face_branch_margin(a, b, train_frozen))
        nominal_faces = oracle.face_signs(control, train_frozen)
        same_face &= bool(torch.equal(nominal_faces, oracle.face_signs(cp, train_frozen)))
        same_face &= bool(torch.equal(nominal_faces, oracle.face_signs(cm, train_frozen)))
        sp, sm, s0 = score(cp, yp), score(cm, ym), score(control, y0)
        finite_difference.append(float((sp - sm) / (2.0 * step)))
        taylor_error.append(float((sp - s0 - step * directional).abs()))
        endpoint_gradients.extend([
            float(torch.func.grad(total_objective, argnums=0)(cp, yp).abs().max()),
            float(torch.func.grad(total_objective, argnums=0)(cm, ym).abs().max()),
        ])
    hessian = oracle.dense_hessian(torch.func.grad(total_objective, argnums=0), control, y0)
    return {
        "scope": "4x5 CPU FP64 total NN observation-input response; fixed state_dict; research-only",
        "production_learning_eligible": False,
        "neural_prior_eligible": False,
        "parameter_retuning": False,
        "control_count": int(control.numel()), "parameter_count": int(params.numel()),
        "stationarity_max": float(torch.func.grad(total_objective, argnums=0)(control, y0).abs().max()),
        "hessian_min_eigenvalue": float(torch.linalg.eigvalsh(hessian).min()),
        "adjoint_relative_residual": float(adjoint), "frozen_adjoint_relative_residual": float(frozen_adjoint),
        "directional_total": directional, "directional_frozen_stats": frozen_directional,
        "frozen_stats_directional_difference": directional - frozen_directional,
        "nn_effect_resolved": abs(directional - frozen_directional) > abs(finite_difference[-1] - directional),
        "nn_effect_resolution_scope": "Comparison with finest central-FD error; not a rigorous uncertainty bound",
        "finite_difference_slopes": finite_difference,
        "finite_difference_errors": [abs(value - directional) for value in finite_difference],
        "taylor_plus_errors": taylor_error, "endpoint_stationarity_max": max(endpoint_gradients),
        "same_face_signs": same_face,
        "model_path": str(model_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(run_probe(args.model), indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
