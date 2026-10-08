from __future__ import annotations

from typing import Any

import torch
import pytest

from examples.weather_scenarios import fv_point_3h_sample_common_search as search


def test_sample_direction_uses_fixed_inner_gradient_pair() -> None:
    minus = torch.zeros(26, dtype=torch.float64)
    plus = torch.zeros(26, dtype=torch.float64)
    minus[0], minus[1] = -2.0, 1.0
    plus[0], plus[1] = 2.0, 1.0
    diagnostic = {"base_control_sha256": search.CONTROL_SHA, "samples": [
        {"requested_eta": search.ETA_MINUS, "gradient": minus.tolist()},
        {"requested_eta": search.ETA_PLUS, "gradient": plus.tolist()}]}

    direction, audit = search.sample_direction(diagnostic)

    assert torch.equal(direction, torch.tensor([0.0, -1.0] + [0.0] * 24, dtype=torch.float64))
    assert audit["theta"] == 0.5
    assert audit["minus_slope"] == audit["plus_slope"] == -1.0
    assert "no root claim" in audit["scope"]


def test_sample_direction_rejects_other_base_lineage() -> None:
    diagnostic = {"base_control_sha256": "not-the-current-base", "samples": []}

    with pytest.raises(ValueError, match="current model control"):
        search.sample_direction(diagnostic)


def test_initial_scale_enforces_radius_and_uses_resolved_positive_curvature() -> None:
    direction = torch.ones(26, dtype=torch.float64)
    gradient = -direction.clone()
    h_direction = 2 * direction

    alpha, audit = search.initial_scale(gradient, direction, h_direction)

    assert alpha == min(1.0, 0.05 / (26**0.5), 0.5)
    assert audit["curvature_cap_used"] is True
    assert audit["initial_step_norm"] <= search.RADIUS


def test_unresolved_curvature_keeps_radius_cap() -> None:
    direction = torch.ones(26, dtype=torch.float64)
    gradient = -direction.clone()
    h_direction = torch.zeros_like(direction)

    alpha, audit = search.initial_scale(gradient, direction, h_direction)

    assert alpha == search.RADIUS / (26**0.5)
    assert audit["curvature_cap_used"] is False


def test_positive_phi_slope_is_diagnostic_and_does_not_refuse_cost_search() -> None:
    g=torch.zeros(26,dtype=torch.float64);d=g.clone();hd=g.clone()
    g[0],g[1]=-1.,10.
    d[0]=1.;hd[0],hd[1]=2.,1.
    alpha,info=search.initial_scale(g,d,hd)
    assert info["g_dot_Hd_diagnostic"]==8.
    assert alpha==min(1.,search.RADIUS,.5)


def test_finite_hvp_with_overflowing_curvature_is_refused() -> None:
    d=torch.ones(26,dtype=torch.float64);g=-d
    hd=torch.full_like(d,1e308)
    with pytest.raises(search.SearchRefusal,match="nonfinite"):
        search.initial_scale(g,d,hd)


def test_actual_alpha_and_norm_include_both_search_scales() -> None:
    direction_norm = 0.1367977784071735
    alpha_init = 0.05 / direction_norm
    helper_alpha = 0.5

    metrics = search.actual_step_metrics(alpha_init, helper_alpha, direction_norm)

    assert metrics["actual_alpha"] == alpha_init * helper_alpha
    assert metrics["actual_step_norm"] == 0.025
    assert metrics["actual_step_norm"] <= search.RADIUS


def test_original_j_armijo_can_accept_phi_increase(monkeypatch) -> None:
    monkeypatch.setattr(search.seed, "_valid_margins", lambda margins, complete: True)
    control = torch.zeros(26, dtype=torch.float64)
    control[0] = 1.0
    parameters = torch.zeros(1, dtype=torch.float64)
    gradient = torch.zeros_like(control)
    gradient[0] = -1.0
    step = torch.zeros_like(control)
    step[0] = 0.02

    def objective(point: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return -0.5 * point[0].square()

    def gradient_fn(point: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(point)
        result[0] = -point[0]
        return result

    branch = {"status": "passed_strict_branch", "euler_stages": 3600,
              "choice_stage_count": 3600, "face_sign_stage_count": 3600,
              "signature_sha256": "base"}
    margins = {"complete": True}

    accepted, trials, status = search.guard_policy.bounded_original_j_search(
        control, parameters, objective(control, parameters), gradient, step,
        objective, gradient_fn, lambda *_: (branch, margins), "base")

    assert status == "one_original_J_step_accepted"
    assert accepted is not None and accepted[0] > control[0]
    assert trials[0]["armijo_passed"] is True
    assert trials[0]["phi"] > float(torch.dot(gradient, gradient) / 2)


def test_final_recheck_requires_exact_fresh_gradient_and_branch(monkeypatch) -> None:
    monkeypatch.setattr(search.seed, "_valid_margins", lambda margins, complete: True)
    gradient = torch.zeros(26, dtype=torch.float64)
    gradient[0] = 1.0
    tentative = {"objective": 2.0, "phi": 0.5, "gradient": gradient.tolist(),
                 "branch": {"signature_sha256": "same"}}
    branch = {"status": "passed_strict_branch", "signature_sha256": "same"}

    passed = search._final_recheck(tentative, torch.tensor(2.0), gradient,
                                   torch.tensor(0.5), branch, {"complete": True})
    failed = search._final_recheck(tentative, torch.tensor(2.0), gradient + 1e-8,
                                   torch.tensor(0.5), branch, {"complete": True})

    assert passed["passed"] is True
    assert "decrease is never an acceptance criterion" in passed["phi_role"]
    assert failed["passed"] is False
    phi_only_difference = search._final_recheck(
        tentative, torch.tensor(2.0), gradient, torch.tensor(0.500001), branch,
        {"complete": True})
    assert phi_only_difference["passed"] is False
    assert phi_only_difference["checks"]["phi_consistency_diagnostic"]["passed"] is False


def test_failed_final_recheck_marks_trial_not_committed() -> None:
    record: dict[str, Any] = {"optimizer_steps_applied": 0, "candidate_committed": False,
                              "trials": [{"status": "accepted", "alpha": 0.5}]}

    search._mark_uncommitted(record, "final recheck mismatch")

    assert record["trials"][0]["status"] == "candidate_not_committed"
    assert record["trials"][0]["commit_refusal"] == "final recheck mismatch"
    assert record["optimizer_steps_applied"] == 0
    assert record["candidate_committed"] is False


def test_parent_child_command_is_accepted_by_real_cli_parser(tmp_path,monkeypatch):
    import json,sys
    digest='a'*64
    output=tmp_path/'result.json';resource=tmp_path/'resource.json';log=tmp_path/'run.log'
    monkeypatch.setattr(search,'_load_plan',lambda *_: {})
    monkeypatch.setattr(search,'_sha',lambda _:digest)
    reached=[]
    def child(plan,sha,path):
        reached.append((plan,sha,path))
        value={'phase':'finished','execution_status':'completed','numerical_status':'search_refusal',
               'source_unchanged':True,'fixed_input_unchanged':True,'runtime':{},'runtime_after':{},
               'plan_sha256':digest,'base_control_sha256':search.CONTROL_SHA}
        path.write_text(json.dumps(value));return value
    monkeypatch.setattr(search,'_run_child',child)
    def guard(command,**kwargs):
        monkeypatch.setattr(sys,'argv',[command[1],*command[2:]])
        search.main()
        return {'elapsed_seconds':.1,'wall_limit_seconds':300.,'rss_limit_bytes':1024**3,
                'sampled_peak_rss_bytes':0,'received_sigterm':False,'resource_termination':None,
                'monitor_error':None,'child_process_group_cleanup_error':None,
                'child_process_group_cleanup_sent':False,'exit_code':0}
    monkeypatch.setattr(search,'run_guarded_diagnostic',guard)
    result=search.run(search.PLAN,digest,output,resource,log)
    assert reached==[(search.PLAN,digest,output)]
    assert result['parent']['execution_status']=='completed'
