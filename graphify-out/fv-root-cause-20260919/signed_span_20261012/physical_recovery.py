"""Compare known generating controls to the accepted analysis without FV calls."""
from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from advar.variational import _control_prior_residual
from examples.weather_scenarios.fv_point_3h_netflux_current_cycle import load_fixed_case
from examples.weather_scenarios.fv_point_3h_current_gradient_endpoint import FACE_WEIGHTS
from examples.weather_scenarios.fv_point_3h_tangent_continuation import _tensor_sha

case = load_fixed_case()
problem, generating = case["problem"], case["original"]
raw = json.loads((WORK / "attempt1/step.json").read_text())
candidate = torch.tensor(raw["current_control"], dtype=torch.float64)
if _tensor_sha(candidate) != raw["current_control_sha256"]:
    raise ValueError("candidate control digest differs from confirmed receipt")
contract = problem.contract(case["parameters"])
if not all(torch.equal(_control_prior_residual(control, contract), control)
           for control in (generating, candidate)):
    raise ValueError("this proof requires the unchanged identity control prior")
J = next(arm["repeat"]["objective"] for arm in raw["arms"]
         if arm["name"] == raw["selected_direction"])
generator_prior = 0.5 * float(generating @ generating)
candidate_prior = 0.5 * float(candidate @ candidate)
if not generator_prior > J:
    raise ValueError("the generating-control lower-bound separation is absent")
spec = problem.frozen.fv_transport
true_coeff = spec.coefficient_limits * torch.tanh(generating[20:25])
estimated_coeff = spec.coefficient_limits * torch.tanh(candidate[20:25])

def face_flux(control_coeff):
    psi = (control_coeff[:, None, None] * spec.psi_basis).sum(0)
    return torch.cat(((psi[1:, :] - psi[:-1, :]).flatten(),
                      (-(psi[:, 1:] - psi[:, :-1])).flatten()))

truth_flux, estimated_flux = face_flux(true_coeff), face_flux(estimated_coeff)
growth = lambda control: problem.frozen.nowcast_config.max_log_growth_per_step * float(torch.tanh(control[25]))
generator_q = float(FACE_WEIGHTS @ torch.tanh(generating[20:25]))
result = {
    "scope": "known generator vs confirmed analysis; cached J and algebra only; no FV/HVP/score calls",
    "candidate_control": _tensor_sha(candidate), "candidate_J": J,
    "candidate_prior_cost": candidate_prior,
    "candidate_nonnegative_data_cost_J_minus_prior": J - candidate_prior,
    "generator_prior_lower_bound_on_J": generator_prior,
    "generator_J_lower_bound_over_candidate_J": generator_prior / J,
    "generator_cannot_minimize_this_J": True,
    "generating_flow_fractions": torch.tanh(generating[20:25]).tolist(),
    "estimated_flow_fractions": torch.tanh(candidate[20:25]).tolist(),
    "relative_staggered_volume_flux_error_norm": float((estimated_flux - truth_flux).norm() / truth_flux.norm()),
    "generating_growth_per_10min": growth(generating),
    "estimated_growth_per_10min": growth(candidate),
    "generator_Q_selected_face": generator_q,
    "candidate_Q_selected_face": float(FACE_WEIGHTS @ torch.tanh(candidate[20:25])),
    "generator_not_on_selected_Q0_face": generator_q != 0.0,
    "interpretation": "J(generator) is bounded below by its prior cost, already above observed candidate J. Exact generator recovery and minimizing this fixed J are incompatible. The generating state is also outside the selected conditional Q0 face. Neither fact proves all J minima have poor forecasts. Echo log growth and synthetic volume flux are not observed wind/rainwater retrieval.",
}
(WORK / "PHYSICAL_RECOVERY.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
