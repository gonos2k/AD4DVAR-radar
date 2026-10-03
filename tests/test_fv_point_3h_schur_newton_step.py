"""Pure coupled-block and Newton/Armijo wiring checks; no FV model run."""
import json
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_schur_newton_step as probe


def test_source_pins_and_frozen_hessian_prerequisite():
    probe.require_pins()
    assert probe.sha(probe.DIAGNOSTIC) == probe.DIAGNOSTIC_SHA
    control = probe.current_endpoint_control()
    assert probe.tensor_sha(control) == probe.frozen.CONTROL_SHA
    assert probe.tensor_sha(control) != probe.seed.SHIFTED_SEED_CONTROL_SHA256
    frozen = json.loads(probe.DIAGNOSTIC.read_text())
    assert frozen["hessian_status"] == "schur_evaluated"
    assert frozen["score_computed"] is False and frozen["response_computed"] is False


def _coupled_spd():
    generator = torch.Generator().manual_seed(23)
    q = torch.randn((26, 26), generator=generator, dtype=torch.float64)
    return q.T @ q + 2.0 * torch.eye(26, dtype=torch.float64)


def test_block_inverse_matches_full_coupled_solve_and_is_positive():
    h = _coupled_spd()
    apply, audit = probe.block_inverse_preconditioner(h)
    rhs = torch.linspace(-1.0, 2.0, 26, dtype=torch.float64)
    actual = apply(rhs)
    expected = torch.linalg.solve(h, rhs)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert float(rhs @ actual) > 0
    assert audit["full"]["lambda_min"] > 0 and audit["hff"]["lambda_min"] > 0
    assert audit["schur"]["lambda_min"] > 0


def test_newton_pcg_uses_original_operator_and_fresh_true_residual():
    frozen_h = _coupled_spd()
    h = frozen_h + 0.125 * torch.eye(26, dtype=torch.float64)
    preconditioner, _ = probe.block_inverse_preconditioner(frozen_h)
    gradient = torch.linspace(-0.5, 1.5, 26, dtype=torch.float64)
    calls = []

    def original_hvp(value):
        calls.append(value.clone())
        return h @ value

    step, audit = probe.solve_newton_direction(original_hvp, gradient, preconditioner)
    assert float(torch.linalg.vector_norm(h @ step + gradient) / gradient.norm()) <= 1e-10
    assert calls and audit["converged"] and audit["true_relative_residual"] <= 1e-10
    assert audit["g_dot_s"] < -audit["g_dot_s_rounding_budget"]
    assert audit["max_iterations"] == 104


def test_line_search_accepts_armijo_even_without_a_phi_decrease(monkeypatch):
    monkeypatch.setattr(probe.seed, "_valid_margins", lambda *_args, **_kwargs: True)
    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    step = torch.zeros_like(control)
    step[0] = 1.0
    branch = {"status": "passed_strict_branch", "euler_stages": 3600,
              "choice_stage_count": 3600, "face_sign_stage_count": 3600,
              "signature_sha256": "changed"}
    candidate, row = probe.evaluate_trial(control, parameters, step, 1.0, 0.0, -1.0,
        lambda c, _p: -c[0], lambda _c, _p: torch.full((26,), 1000.0, dtype=torch.float64),
        lambda *_args: (branch, {}))
    assert row["status"] == "accepted" and row["phi"] > 0
    assert row["branch"]["signature_sha256"] == "changed"
    assert "branch_signature_changed" not in row  # caller compares it with the pinned base signature
    assert candidate[0] == 1.0
    assert probe.ALPHAS == tuple(2.0**-i for i in range(16))


def test_armijo_rejection_skips_branch_and_known_trials_are_json_safe():
    control, parameters, step = (torch.zeros(26, dtype=torch.float64),
                                torch.zeros(13, dtype=torch.float64),
                                torch.ones(26, dtype=torch.float64))
    def unexpected(*_args):
        raise AssertionError("branch must not run before objective Armijo passes")
    _, row = probe.evaluate_trial(control, parameters, step, 1.0, 0.0, -1.0,
        lambda _c, _p: torch.tensor(1.0, dtype=torch.float64), lambda _c, _p: torch.zeros(26), unexpected)
    assert row["status"] == "armijo_rejected"
    json.dumps(row, allow_nan=False)


def test_unclassified_objective_exception_propagates():
    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="programming defect"):
        probe.evaluate_trial(control, parameters, control, 1.0, 0.0, -1.0,
            lambda *_args: (_ for _ in ()).throw(RuntimeError("programming defect")),
            lambda *_args: torch.zeros(26), lambda *_args: ({}, {}))


def test_wrong_callback_shapes_raise_as_contract_errors(monkeypatch):
    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    step = torch.ones_like(control)
    with pytest.raises(ValueError, match="objective must return"):
        probe.evaluate_trial(control, parameters, step, 1.0, 0.0, -1.0,
            lambda *_args: torch.zeros(2), lambda *_args: torch.zeros(26), lambda *_args: ({}, {}))
    monkeypatch.setattr(probe.seed, "_valid_margins", lambda *_args, **_kwargs: True)
    branch = {"status": "passed_strict_branch", "euler_stages": 3600,
              "choice_stage_count": 3600, "face_sign_stage_count": 3600}
    with pytest.raises(ValueError, match="gradient must be CPU FP64 length 26"):
        probe.evaluate_trial(control, parameters, step, 1.0, 0.0, -1.0,
            lambda c, _p: -c[0], lambda *_args: torch.zeros(27), lambda *_args: (branch, {}))


def test_unresolved_indefinite_block_refuses_preconditioner():
    h = _coupled_spd()
    h[0, 0] = -100.0
    with pytest.raises(probe.StepRefusal, match="positive definite"):
        probe.block_inverse_preconditioner(h)


def test_known_initial_preconditioner_refusal_writes_terminal_identity_audited_report(monkeypatch, tmp_path):
    diagnostic = json.loads(probe.DIAGNOSTIC.read_text())
    control = probe.current_endpoint_control()
    parameters = torch.zeros(13, dtype=torch.float64)
    original, truth = torch.zeros(26, dtype=torch.float64), torch.zeros((4, 5), dtype=torch.float64)
    problem = SimpleNamespace(objective=object())
    identity = diagnostic["input_before"]["identity"]
    monkeypatch.setattr(probe.seed, "_prepare_fixed_seed",
        lambda: (problem, original, torch.zeros_like(control), parameters, truth, {}))
    monkeypatch.setattr(probe.seed, "_input_identity", lambda *_args: identity)
    real_hash = probe.tensor_sha
    monkeypatch.setattr(probe, "tensor_sha", lambda value:
        diagnostic["endpoint_control_sha256"] if value.shape == control.shape and torch.equal(value, control)
        else diagnostic["parameters_sha256"] if value.shape == parameters.shape else real_hash(value))
    branch = dict(diagnostic["branch"])
    margins = diagnostic["branch_margins"]
    monkeypatch.setattr(probe.tail, "_full_current_branch", lambda *_args: (branch, margins))
    monkeypatch.setattr(probe.seed, "_valid_margins", lambda *_args, **_kwargs: True)
    gradient = torch.tensor(diagnostic["gradient"], dtype=torch.float64)
    monkeypatch.setattr(torch.func, "grad", lambda *_args, **_kwargs: lambda *_args: gradient)
    monkeypatch.setattr(probe.tail, "_fresh_merit", lambda *_args: (
        torch.tensor(diagnostic["objective"], dtype=torch.float64), gradient,
        torch.tensor(diagnostic["phi"], dtype=torch.float64)))
    monkeypatch.setattr(probe, "block_inverse_preconditioner",
        lambda *_args: (_ for _ in ()).throw(probe.StepRefusal("synthetic preconditioner refusal")))
    monkeypatch.setattr(probe.matrix_free, "pcg",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("PCG must not run after preconditioner refusal")))
    output = tmp_path / "step.json"
    result = probe.run(output)
    assert result["phase"] == "finished" and result["numerical_status"] == "step_refused"
    assert result["refusal"] == "synthetic preconditioner refusal" and result["optimizer_steps"] == 0
    assert result["source_unchanged"] and result["input_unchanged"] and result["runtime_after"] == result["runtime"]
    assert json.loads(output.read_text()) == result
