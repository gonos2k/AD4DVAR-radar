"""Small analytic checks for the research-only objective basin search."""

import pytest
import torch

from examples.weather_scenarios.fv_point_basin_search import search_basin
from examples.weather_scenarios import fv_point_basin_probe as probe


def _constant_branch(_control, _parameters):
    return {"choices": [], "face_signs": []}, "smooth analytic problem"


def test_spd_quadratic_reaches_a_small_gradient_with_true_objective_decrease():
    diagonal = torch.tensor([2.0, 5.0], dtype=torch.float64)
    target = torch.tensor([0.3, -0.2], dtype=torch.float64)

    def objective(control, _parameters):
        delta = control - target
        return 0.5 * torch.dot(delta, diagonal * delta)

    start = torch.tensor([2.0, -2.0], dtype=torch.float64)
    result = search_basin(objective, start, torch.zeros(1, dtype=start.dtype),
                          _constant_branch, max_steps=50, handoff_gradient_max=1e-8)
    assert result.status == "candidate"
    assert result.accepted_steps > 0
    assert result.branch_changes == 0
    assert result.objective < float(objective(start, torch.zeros(1, dtype=start.dtype)))
    torch.testing.assert_close(result.control, target, rtol=0, atol=1e-8)
    assert all(record["reduction"] > 0 for record in result.history
               if record["status"] == "accepted")
    accepted = [record for record in result.history if record["status"] == "accepted"]
    assert all(isinstance(record["control"], list) and len(record["control"]) == 2
               for record in accepted)
    assert all(isinstance(record["gradient"], list) and len(record["gradient"]) == 2
               for record in accepted)
    assert accepted[-1]["control"] == result.control.tolist()


def test_nonconvex_negative_curvature_start_can_reach_a_local_basin():
    def objective(control, _parameters):
        x, y = control.unbind()
        return 0.5 * x.square() - 0.5 * y.square() + 0.25 * y.pow(4)

    start = torch.tensor([0.2, 0.2], dtype=torch.float64)
    result = search_basin(objective, start, torch.zeros(1, dtype=start.dtype),
                          _constant_branch, max_steps=80, handoff_gradient_max=1e-5)
    assert result.status == "candidate"
    assert result.objective < float(objective(start, torch.zeros(1, dtype=start.dtype)))
    assert result.control[1] > 0.9
    assert result.gradient_max <= 1e-5


def test_accepted_sector_change_clears_local_curvature_history():
    def objective(control, _parameters):
        return 0.5 * (control[0] - 0.8).square()

    def branch(control, _parameters):
        value = float(control[0])
        if abs(value) <= 0.05:
            raise ValueError("nonstrict branch margin")
        sector = -1 if value < 0 else 1
        return {"choices": [sector], "face_signs": [sector]}, "strict sector"

    start = torch.tensor([-0.2], dtype=torch.float64)
    result = search_basin(objective, start, torch.zeros(1, dtype=start.dtype),
                          branch, max_steps=20, handoff_gradient_max=1e-8)
    assert result.status == "candidate"
    assert result.branch_changes >= 1
    assert any(record.get("branch_changed") for record in result.history)
    torch.testing.assert_close(result.control, torch.tensor([0.8], dtype=start.dtype),
                               rtol=0, atol=1e-8)


def test_no_strict_trial_is_a_refusal_not_a_stationary_result():
    def objective(control, _parameters):
        return 0.5 * (control[0] - 1).square()

    start = torch.tensor([0.0], dtype=torch.float64)

    def branch(control, _parameters):
        if not torch.equal(control, start):
            raise ValueError("candidate outside declared smooth sector")
        return {"choices": [0], "face_signs": [1]}, "pinned start only"

    result = search_basin(objective, start, torch.zeros(1, dtype=start.dtype),
                          branch, max_backtracks=4)
    assert result.status == "line_search_refused"
    assert result.accepted_steps == 0 and result.trial_evaluations == 4
    assert all(record["status"] == "branch_refused" for record in result.history)
    assert all(len(record["control_sha256"]) == 64 for record in result.history)


def test_finite_gradient_components_with_overflowed_norm_are_rejected():
    def objective(control, _parameters):
        return 1e308 * (control[0] + control[1])

    with pytest.raises(ValueError, match="gradient norm is nonfinite"):
        search_basin(
            objective, torch.zeros(2, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64), _constant_branch,
        )


def test_phase_budget_includes_initial_branch_and_gradient_evaluation():
    def objective(control, _parameters):
        return 0.5 * (control[0] - 1).square()

    result = search_basin(
        objective, torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64), _constant_branch,
        phase_seconds=1e-12,
    )
    assert result.status == "time_budget"
    assert result.accepted_steps == 0


def test_explicit_trial_budget_stops_a_nonaccepting_search():
    def objective(control, _parameters):
        return 0.5 * (control[0] - 1).square()

    start = torch.zeros(1, dtype=torch.float64)

    def branch(control, _parameters):
        if not torch.equal(control, start):
            raise ValueError("trial refused")
        return {"choices": [0], "face_signs": [1]}, "start only"

    result = search_basin(objective, start, torch.zeros(1, dtype=start.dtype),
                          branch, max_trials=2)
    assert result.status == "trial_budget" and result.trial_evaluations == 2


@pytest.mark.parametrize("stages,slope,face,accepted", [
    (54, 2e-4, 3e-4, True),
    (53, 2e-4, 3e-4, False),
    (54, 5e-5, 3e-4, False),
    (54, 2e-4, 5e-5, False),
    (54, float("nan"), 3e-4, False),
    (54, 2e-4, float("nan"), False),
])
def test_point_adapter_enforces_stage_and_both_margin_gates(
    monkeypatch, stages, slope, face, accepted,
):
    def fake_branch(_problem, _control, _parameters):
        return {"euler_stages": stages, "minimum_scaled_slope_margin": slope,
                "choices": [], "face_signs": []}, "fixed point scope", face

    monkeypatch.setattr(probe.preflight, "_branch", fake_branch)
    control = torch.zeros(1, dtype=torch.float64)
    if accepted:
        signature, scope = probe._strict_branch(None, control, control)
        assert signature["minimum_scaled_face_flux_margin"] == face
        assert scope == "fixed point scope"
    else:
        with pytest.raises(ValueError, match="margin refused"):
            probe._strict_branch(None, control, control)


@pytest.mark.parametrize("diagonal,accepted", [
    ((2.0, 3.0), True),
    ((1.0, -1.0), False),
    ((1.0, 1e-12), False),
])
def test_exact_hessian_gate_distinguishes_spd_and_weak_curvature(diagonal, accepted):
    matrix = torch.diag(torch.tensor(diagonal, dtype=torch.float64))

    class Problem:
        def objective(self, control, _parameters):
            return 0.5 * torch.dot(control, matrix @ control)

    control = torch.zeros(2, dtype=torch.float64)
    parameters = torch.zeros(1, dtype=torch.float64)
    if accepted:
        diagnostics = probe._hessian_audit(Problem(), control, parameters)
        assert diagnostics["lambda_min"] == 2.0
        assert diagnostics["lambda_max"] == 3.0
        assert diagnostics["symmetry_relative"] == 0.0
    else:
        with pytest.raises(ValueError, match="sufficiently SPD"):
            probe._hessian_audit(Problem(), control, parameters)


def test_warm_entry_reuses_finite_margin_gate(monkeypatch):
    def fake_branch(_problem, _control, _parameters):
        return {"euler_stages": 54, "minimum_scaled_slope_margin": float("nan"),
                "choices": [], "face_signs": []}, "warm scope", 0.01

    monkeypatch.setattr(probe.preflight, "_branch", fake_branch)
    control = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="margin refused"):
        probe._warm_branch(None, control, control,
                           {"choices": [], "face_signs": []})


@pytest.mark.parametrize("slope,face,accepted", [
    (5e-5, 5e-5, True),
    (float("nan"), 5e-5, False),
    (5e-5, float("nan"), False),
    (0.0, 5e-5, False),
])
def test_exploration_gate_is_pointwise_but_final_gate_keeps_robust_margin(
    monkeypatch, slope, face, accepted,
):
    def fake_branch(_problem, _control, _parameters):
        return {"euler_stages": 54, "minimum_scaled_slope_margin": slope,
                "choices": [], "face_signs": []}, "exploration", face

    monkeypatch.setattr(probe.preflight, "_branch", fake_branch)
    control = torch.zeros(1, dtype=torch.float64)
    if accepted:
        signature, _ = probe._exploration_branch(None, control, control)
        assert signature["minimum_scaled_slope_margin"] == 5e-5
        with pytest.raises(ValueError, match="margin refused"):
            probe._strict_branch(None, control, control)
    else:
        with pytest.raises(ValueError, match="pointwise smoothness refused"):
            probe._exploration_branch(None, control, control)
