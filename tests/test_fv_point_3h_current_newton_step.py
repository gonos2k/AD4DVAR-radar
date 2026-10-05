"""Saved-matrix and analytic model checks; no FV trajectory execution."""
import json

import torch
from types import SimpleNamespace

from examples.weather_scenarios import fv_point_3h_current_newton_step as probe


def test_unshifted_direction_solves_original_f82c_system():
    base = json.loads(probe.BASE.read_text())
    h = torch.tensor(base["curvature"]["hessian"], dtype=torch.float64)
    g = torch.tensor(base["gradient"], dtype=torch.float64)
    s, audit = probe.search.exact_shifted_direction(h, g, mu=0.)
    torch.testing.assert_close(h @ s, -g, rtol=1e-10, atol=1e-12)
    assert audit["mu"] == 0 and audit["g_dot_s"] < 0
    torch.testing.assert_close(g @ (h @ s), -(g @ g), rtol=1e-10, atol=1e-12)
    shifted, _ = probe.search.exact_shifted_direction(h, g, mu=40.)
    assert (shifted-s).norm() > .1
    weak = torch.tensor(json.loads(probe.CHECKPOINT.read_text())["audit_hvp"]["direction"], dtype=torch.float64)
    assert .59 < float((weak @ s).square()/(s @ s)) < .61


def test_model_agrees_exactly_for_quadratic_objective():
    h = 2*torch.eye(26, dtype=torch.float64)
    c = torch.linspace(-.3, .3, 26, dtype=torch.float64)
    delta = -.2*c
    j0, j1 = float(c @ c), float((c+delta) @ (c+delta))
    result = probe.model_diagnostics(j0, 2*c, h, delta, j1, 2*(c+delta))
    assert abs(result["predicted_J"]-j1) < 1e-14
    assert abs(result["actual_over_predicted_J_reduction"]-1) < 1e-14
    assert result["gradient_linearization_error_l2"] < 1e-14
    assert abs(result["directional_secant"]-result["old_point_directional_curvature"]) < 1e-14


def test_nonlinear_change_is_reported_not_used_as_new_hessian():
    h = torch.eye(26, dtype=torch.float64)
    g = torch.ones(26, dtype=torch.float64)
    delta = -.01*g
    actual_g = g+h@delta
    actual_g[-1] += 1
    result = probe.model_diagnostics(1., g, h, delta, .9, actual_g)
    assert result["gradient_linearization_error_l2"] > .99
    assert result["actual_phi"] > result["linear_phi"]
    assert "new_hessian" not in result


def test_deadline_after_candidate_prevents_additional_branch_capture(tmp_path, monkeypatch):
    base = json.loads(probe.BASE.read_text())
    checkpoint = json.loads(probe.CHECKPOINT.read_text())
    c = torch.tensor(base["control"], dtype=torch.float64)
    p = torch.tensor(base["parameters"], dtype=torch.float64)
    g = torch.tensor(base["gradient"], dtype=torch.float64)
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(probe, "load_base", lambda *args:
                        ({"source_files": {}, "archive_files": {}, "policy": {}}, base, checkpoint))

    def forbidden(*args):
        raise AssertionError("expired accepted candidate cannot start another branch capture")

    problem = SimpleNamespace(objective=lambda x, pp: x.sum(), branch_check=forbidden)
    identity = base["input_after"]
    monkeypatch.setattr(probe.curvature.seed, "_prepare_fixed_seed", lambda:
                        (problem, object(), None, p, object(), {}))
    monkeypatch.setattr(probe.curvature.seed, "_input_identity", lambda *args: identity)
    monkeypatch.setattr(probe.curvature.tail, "_full_current_branch", lambda *args:
                        (base["endpoint_branch"], base["endpoint_margins"]))
    monkeypatch.setattr(probe.curvature.tail, "_fresh_merit", lambda *args:
                        (torch.tensor(base["objective"], dtype=torch.float64), g,
                         torch.tensor(base["phi"], dtype=torch.float64)))
    clock = {"now": 0.}
    monkeypatch.setattr(probe.search.time, "monotonic", lambda: clock["now"])

    def candidate(*args, **kwargs):
        row = {"status": "accepted"}
        kwargs["on_trial"](row)
        clock["now"] = 241.
        return c+.001, [row], "one_original_J_step_accepted"

    monkeypatch.setattr(probe.search, "bounded_original_j_search", candidate)
    result = probe.run(plan, probe.curvature.sha(plan), tmp_path / "step.json")
    assert result["numerical_status"] == "budget_refusal"
    assert result["optimizer_steps_applied"] == 0
    assert result["trials"][-1]["status"] == "candidate_not_committed"
