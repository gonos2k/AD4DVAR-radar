"""Pure S4 block/Schur algebra tests; no FV problem construction."""
import json
from pathlib import Path

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_terminal_block_schur as probe


def test_frozen_source_and_plan_pins_match():
    probe.require_source_pins()


def _case():
    h = 3.0 * torch.eye(26, dtype=torch.float64)
    h[:20, :20] = 2.0 * torch.eye(20, dtype=torch.float64)
    h[0, 20] = h[20, 0] = 0.5
    g = torch.zeros(26, dtype=torch.float64)
    g[0], g[20] = 2.0, 1.0
    return h, g


def test_gradient_blocks_keep_field_flow_growth_split():
    g = torch.arange(1, 27, dtype=torch.float64)
    result = probe.gradient_blocks(g)
    assert result["field"]["l2"] == float(g[:20].norm())
    assert result["flow"]["l2"] == float(g[20:25].norm())
    assert result["growth"]["value"] == 26.0
    assert result["dynamics"]["l2"] == float(g[20:].norm())


def test_schur_and_linearized_eliminated_residual_are_algebraically_correct():
    result = probe.schur_diagnostic(*_case())
    assert result["status"] == "schur_evaluated"
    assert result["schur_eigenvalues"][0] == pytest.approx(2.875)
    assert result["linearized_eliminated_dynamics_residual"][0] == pytest.approx(0.5)
    assert max(result["field_solve_residuals"].values()) < 1e-12


def test_hff_indefinite_or_ill_conditioned_refuses_schur():
    h, g = _case()
    h[0, 0] = -1.0
    result = probe.schur_diagnostic(h, g)
    assert result["status"] == "hff_schur_refused" and result["hff_condition"] is None
    json.dumps(result, allow_nan=False)
    h, g = _case()
    h[0, 0] = 1e-9
    result = probe.schur_diagnostic(h, g)
    assert result["status"] == "hff_schur_refused"
    assert result["hff_condition"] > result["hff_condition_limit"]
    json.dumps(result, allow_nan=False)


def test_hdd_scale_does_not_raise_hff_roundoff_floor():
    h, g = _case()
    h[20:, 20:] *= 1e12
    result = probe.schur_diagnostic(h, g)
    assert result["hff_positive_floor"] < 1e-6
    assert result["status"] == "schur_evaluated"


def test_asymmetry_and_bad_schur_solve_fail_closed(monkeypatch):
    h, g = _case()
    h[0, 20] += 0.1
    with pytest.raises(probe.DiagnosticRefusal, match="symmetry"):
        probe.schur_diagnostic(h, g)
    h, g = _case()
    monkeypatch.setattr(torch.linalg, "solve", lambda a, b: torch.zeros_like(b))
    assert probe.schur_diagnostic(h, g)["refusal"] == "Hff solve residual failed"


def test_bad_eigenpair_residual_is_a_json_safe_refusal_record():
    result = probe.eigenpair_audit(torch.zeros(26, dtype=torch.float64),
                                   torch.tensor(1.0), torch.ones(26, dtype=torch.float64),
                                   torch.tensor(1.0))
    assert result["passed"] is False
    assert result["relative_residual"] > result["tolerance"]
    json.dumps(result, allow_nan=False)
    extreme = probe.eigenpair_audit(torch.zeros(26, dtype=torch.float64),
                                    torch.tensor(1.0), torch.ones(26, dtype=torch.float64),
                                    torch.tensor(1e-308))
    assert extreme["passed"] is False and extreme["relative_residual"] is None
    json.dumps(extreme, allow_nan=False)


def test_metric_budget_and_tail_control_binding_are_pure():
    check = probe.metric_check(1.0, 1.0)
    assert check["passed"] and check["difference"] == 0.0
    raw = json.loads(probe.TAIL_RAW.read_text())
    control = probe.endpoint_control(raw)
    assert tuple(control.shape) == (26,)
    assert probe.tensor_sha(control) == probe.CONTROL_SHA


def test_known_initial_branch_refusal_finishes_record_without_hvp(monkeypatch, tmp_path):
    tail = json.loads(probe.TAIL_RAW.read_text())
    p = torch.zeros(13, dtype=torch.float64)
    original = torch.zeros(26, dtype=torch.float64)
    truth = torch.zeros((4, 5), dtype=torch.float64)
    identity = tail["input_after"]
    problem = object()
    base_identity = {"archived_input": identity["archived_input"]}
    monkeypatch.setattr(probe.seed_probe, "_prepare_fixed_seed",
                        lambda: (problem, original, original, p, truth, base_identity))
    monkeypatch.setattr(probe.seed_probe, "_input_identity", lambda *_args: identity)
    real_sha = probe.tensor_sha
    monkeypatch.setattr(probe, "tensor_sha", lambda value: probe.PARAMETERS_SHA if value is p else real_sha(value))
    bad_branch = {"status": "branch_or_margin_refused", "signature_sha256": None}
    bad_margins = {"complete": False}
    monkeypatch.setattr(probe.tail_probe, "_full_current_branch",
                        lambda *_args: (dict(bad_branch), dict(bad_margins)))
    monkeypatch.setattr(probe.tail_probe, "_fresh_merit",
                        lambda *_args: (_ for _ in ()).throw(AssertionError("no objective/HVP after refusal")))
    target = tmp_path / "refusal.json"
    result = probe.run(target)
    assert result["phase"] == "finished"
    assert result["numerical_status"] == "diagnostic_refusal"
    assert result["reason"] == "current endpoint no longer passes the pinned strict branch/margin gate"
    assert result["hvp_columns_completed"] == 0
    assert result["source_unchanged"] is True and result["input_unchanged"] is True
    assert json.loads(target.read_text()) == result
