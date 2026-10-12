from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_netflux_subspace_trial as trial


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path, *, bad_subspace_repeat: bool = False,
         gn_timeout: bool = False):
    dtype = torch.float64
    H = torch.eye(26, dtype=dtype)
    c = torch.zeros(26, dtype=dtype)
    c[0], c[4] = 1e-3, -3e-4
    normal = torch.zeros_like(c)
    normal[0] = 1.0
    counts = {"HVP": 0, "row": 0, "repeat": 0, "DG_linearity": 0}

    def observe(control: torch.Tensor) -> dict:
        gm = H @ control
        gp = gm.clone()
        gm[0], gp[0] = -1.0, 1.0
        objective = 0.5 * torch.dot(control, H @ control)
        trace = {"signature_sha256": trial.tangent._tensor_sha(control)}
        return {"native_j": objective, "q": torch.tensor(0.0, dtype=dtype),
            "side": {-1: (objective, gm), 1: (objective, gp)},
            "traces": {-1: trace, 1: trace}, "pair": {"passed": True},
            "face_ok": True, "side_objectives_match_native": True,
            "side_gradients_finite": True}

    def repeat_at(control: torch.Tensor) -> dict:
        obs = observe(control)
        G = torch.cat((H @ control, torch.zeros(1, dtype=dtype)))
        return {"control": control.tolist(), "theta": 0.5,
            "objective": float(obs["native_j"]), "F_squared": float(torch.dot(G, G)),
            "side_gradients": {str(side): obs["side"][side][1].tolist() for side in (-1, 1)},
            "branch_trace": {str(side): obs["traces"][side] for side in (-1, 1)}}

    base_repeat = repeat_at(c)
    base = {"control": c, "theta": 0.5,
        "full_G": torch.cat((H @ c, torch.zeros(1, dtype=dtype))),
        "repeat": base_repeat, "input_after": {"fixed": True}, "runtime_after": {"fixed": True}}

    class Backend:
        def __init__(self, *_args):
            self.control = c.clone()
            self.weights = normal
            self.parameters = torch.zeros(13, dtype=dtype)
            self.source_anchor = {"source": "fixed"}
            self.input_anchor = base["input_after"]
            self.runtime_anchor = base["runtime_after"]
            self.problem = SimpleNamespace(objective=lambda x, _p: 0.5 * torch.dot(x, H @ x))
            self.original = self.truth = object()
            self.timings = {"pure_hvp_seconds": 0.0, "row_solve_seconds_inclusive": 0.0}

        def _check(self, *_args):
            return None

        def chart(self, _bundle, control, direction, alpha):
            return control + alpha * direction

        def evaluate(self, _bundle, candidate, alpha):
            G = torch.cat((H @ candidate, torch.zeros(1, dtype=dtype)))
            J, R = 0.5 * torch.dot(candidate, H @ candidate), torch.dot(G, G)
            trace = {str(side): {"signature_sha256": trial.tangent._tensor_sha(candidate)}
                for side in (-1, 1)}
            return {"alpha": alpha, "control": candidate.tolist(), "G": G.tolist(),
                "objective": float(J), "F_squared": float(R), "theta": 0.5,
                "mixing_minimum_valid": True, "J_armijo_passed": True,
                "F_squared_armijo_passed": True, "branch_pair_passed": True,
                "face_audit_passed": True, "side_objectives_match_native": True,
                "side_gradients_finite": True, "finite": True,
                "branch_trace": trace, "branch_signature_matches_base": False,
                "model_error_norm": 0.0}

        def repeat(self, _bundle, selected):
            counts["repeat"] += 1
            x = torch.as_tensor(selected["control"], dtype=dtype)
            actual = repeat_at(x)
            if bad_subspace_repeat and counts["repeat"] == 2:
                actual["F_squared"] = 10.0
            return ({**selected, "control_sha256": trial.tangent._tensor_sha(x),
                "J_armijo_bound": base_repeat["objective"] * 1.01,
                "F_squared_armijo_bound": base_repeat["F_squared"] * 1.01,
                "conditional_Q0": True, "J_armijo_passed": True,
                "F_squared_armijo_passed": True}, actual)

    def factory(context):
        def build(control, _theta, gm, _gp, _normal, _chart, _pivot):
            op = context["record"]
            op.update(jacobian_rows_started=24, jacobian_rows_completed=24,
                dense_solves_started=1, dense_solves_completed=1)
            counts["row"] += 24
            if gn_timeout:
                raise TimeoutError("GN preparation reached the shared deadline")
            gradient = -gm
            curvature = -2.0 * gm
            return [{"name": "baseline_tangent", "direction": gradient},
                {"name": "robust_gn_coupled", "direction": curvature}]
        return build

    def model(gm, gp, hm, hp, _normal, _chart, theta, _q, _pivot, _scale, *, direction_override):
        G = torch.cat((0.5 * (gm + gp), torch.zeros(1, dtype=dtype)))
        mixed_hvp = 0.5 * (hm + hp)
        r = torch.cat((mixed_hvp, torch.zeros(1, dtype=dtype)))
        return SimpleNamespace(direction=direction_override, residual=G,
            residual_direction=r, gates={key: True for key in (
                "finite", "both_side_gradients_descend", "merit_descends",
                "envelope_slope_matches_residual_dot")},
            merit_product=float(torch.dot(G, r)), theta=theta,
            side_products=(float(torch.dot(gm, direction_override)),
                float(torch.dot(gp, direction_override))))

    def full_g(repeat_data, _expected, _weights):
        x = torch.as_tensor(repeat_data["control"], dtype=dtype)
        G = torch.cat((H @ x, torch.zeros(1, dtype=dtype)))
        return {"G": G.tolist(), "passed": True}

    original_jvp = torch.func.jvp

    def jvp(*args, **kwargs):
        counts["HVP"] += 1
        return original_jvp(*args, **kwargs)

    old_match = trial.finite._full_G_match

    def match(a, b):
        counts["DG_linearity"] += 1
        return old_match(a, b)

    monkeypatch.setattr(trial, "ROOT", tmp_path)
    monkeypatch.setattr(trial, "ATTEMPT", tmp_path / "attempt")
    monkeypatch.setattr(trial.comparison, "_sha", lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(trial, "load_plan", lambda *_args: {})
    monkeypatch.setattr(trial.endpoint, "load_endpoint", lambda: base)
    monkeypatch.setattr(trial.comparison, "CurrentBackend", Backend)
    monkeypatch.setattr(trial.tangent, "_observe", lambda _s, _p, x, *_args: observe(x))
    monkeypatch.setattr(trial.tangent, "minimum_mixture_weight", lambda *_args: torch.tensor(0.5, dtype=dtype))
    monkeypatch.setattr(trial.tangent, "_validate_direction_override", lambda *_args: None)
    monkeypatch.setattr(trial.geometry, "_face_normal", lambda *_args: normal)
    monkeypatch.setattr(trial.geometry, "_pivot", lambda *_args: 0)
    monkeypatch.setattr(trial.geometry, "_chart_jacobian",
        lambda *_args: (None, torch.eye(26, dtype=dtype), None))
    monkeypatch.setattr(trial.previous, "match_previous_P2", lambda *_args: {"passed": True})
    monkeypatch.setattr(trial.previous, "native_face_release", lambda *_args: {"release_required": False})
    monkeypatch.setattr(trial.comparison.coupled, "_direction_model_factory", factory)
    monkeypatch.setattr(trial.torch.func, "jvp", jvp)
    monkeypatch.setattr(trial.prepared.mixing, "minimum_tangent_model", model)
    monkeypatch.setattr(trial.finite, "_base_model_closure", lambda *_args: {"passed": True})
    monkeypatch.setattr(trial.finite, "_full_G_match", match)
    monkeypatch.setattr(trial.previous, "fresh_full_G", full_g)
    monkeypatch.setattr(trial.tangent, "fresh_final_closure", lambda *_args: True)
    monkeypatch.setattr(trial.shared, "_input_identity", lambda *_args: base["input_after"])
    monkeypatch.setattr(trial.shared, "_source_hashes", lambda *_args: {"source": "fixed"})
    monkeypatch.setattr(trial.shared, "_runtime", lambda: base["runtime_after"])
    monkeypatch.setattr(trial.geometry.model, "_fixed_input", lambda *_args: True)
    result = trial._run_child(tmp_path / "PLAN.json", "synthetic", tmp_path / "step.json")
    return result, counts


def test_two_direction_trial_shares_hvps_rows_repeats_and_DG_linearity(monkeypatch, tmp_path):
    raw, counts = _run(monkeypatch, tmp_path)
    assert raw["execution_status"] == "completed", raw.get("termination_reason")
    assert raw["optimizer_steps_applied"] == 1 and raw["candidate_committed"] is True
    assert counts["HVP"] == 4 and raw["hvp_completed"] == 4
    assert counts["row"] == 24 and raw["GN_operator"]["jacobian_rows_completed"] == 24
    assert raw["GN_operator"]["dense_solves_completed"] == 1
    assert counts["repeat"] == 2
    assert counts["DG_linearity"] >= 1
    assert raw["comparison_complete"] is True


def test_finite_candidate_with_corrupt_subspace_repeat_cannot_fallback_to_gradient(monkeypatch, tmp_path):
    raw, counts = _run(monkeypatch, tmp_path, bad_subspace_repeat=True)
    assert raw["execution_status"] == "failed"
    assert raw["optimizer_steps_applied"] == 0 and raw["candidate_committed"] is False
    assert raw["current_control"] == raw["last_confirmed_control"]
    assert counts["repeat"] == 2
    assert raw["arms"][0]["trials"][0]["finite"] is True
    assert raw["arms"][1]["trials"][0]["finite"] is True


def test_GN_timeout_keeps_d19e_base_without_single_arm_commit(monkeypatch, tmp_path):
    raw, counts = _run(monkeypatch, tmp_path, gn_timeout=True)
    assert raw["numerical_status"] == "budget_exhausted_no_commit"
    assert raw["base_control_sha256"] == trial.BASE_SHA
    assert raw["current_control"] == raw["last_confirmed_control"]
    assert raw["optimizer_steps_applied"] == 0 and raw["candidate_committed"] is False
    assert counts["HVP"] == 0 and counts["row"] == 24 and counts["repeat"] == 0


def test_curved_chart_contracts_actual_displacement_and_model_by_same_alpha():
    c = torch.zeros(2,dtype=torch.float64)
    direction = torch.tensor([1.,0.],dtype=c.dtype)
    class CurvedChart:
        def _check(self, *_args): pass
        def chart(self, bundle, current, d, alpha):
            return current + alpha*d + torch.tensor([0.,alpha*alpha],dtype=c.dtype)
    candidate, alpha, calls = trial.chart_within_radius(CurvedChart(),{'direction':direction},c,1.)
    assert calls == 33 and 0 < alpha < 1.
    assert float(candidate.norm()) <= 1.
    expected_alpha = ((5.**.5-1)/2)**.5
    assert alpha == pytest.approx(expected_alpha,abs=1e-9)
    G = torch.tensor([1.,1.],dtype=c.dtype)
    r = torch.tensor([-0.1,-0.1],dtype=c.dtype)
    fidelity = trial.model_fidelity(G,alpha*r,G+alpha*r)
    assert fidelity['rho_R'] == pytest.approx(1.)
    assert fidelity['eta_vec'] == pytest.approx(0.,abs=1e-14)


def test_good_scalar_decrease_cannot_hold_radius_with_poor_vector_prediction():
    G=torch.tensor([1.,0.],dtype=torch.float64)
    r=torch.tensor([-0.001,0.],dtype=G.dtype)
    actual=torch.tensor([(0.999**2-0.02**2)**.5,0.02],dtype=G.dtype)
    fidelity=trial.model_fidelity(G,r,actual)
    assert fidelity['rho_R'] == pytest.approx(1.)
    assert fidelity['eta_vec'] > 0.5
    assert trial.radius_factor(True,fidelity['rho_R'],fidelity['eta_vec']) == 0.5
    assert trial.radius_factor(False,1.,0.) == 0.5
