from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from advar import transport
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as probe


def test_selected_face_chart_and_coupled_system_use_the_full_unknowns():
    control = torch.zeros(26, dtype=torch.float64)
    control[20] = torch.atanh(torch.tensor(0.1, dtype=torch.float64))
    control[21] = torch.atanh(torch.tensor(-0.1, dtype=torch.float64))
    weights = torch.tensor([1.0, 1.0, -0.5, 0.25, 0.75], dtype=torch.float64)
    pivot = 20

    point, chart, _ = probe._chart_jacobian(control, weights, pivot, 0.0)
    normal = probe._face_normal(point, weights)
    assert abs(float(probe._face_value(point, weights))) < 1e-15
    assert chart.shape == (26, 25)
    assert torch.count_nonzero(chart[:20]) == 20
    assert float((normal @ chart).abs().max()) < 1e-14

    hminus = torch.eye(26, dtype=torch.float64) * 2
    hplus = torch.eye(26, dtype=torch.float64) * 3
    gminus = torch.linspace(-0.3, 0.4, 26, dtype=torch.float64)
    gplus = gminus.flip(0) * 0.6
    face_value = torch.tensor(0.03, dtype=torch.float64)
    qscale = weights.abs().max()
    matrix, residual, _ = probe._coupled_matrix(
        hminus, hplus, gminus, gplus, normal, face_value, 0.37, qscale,
    )
    solution, audit = probe._solve_coupled(matrix, residual)
    assert solution.shape == (27,)
    assert audit["passed"]
    assert torch.linalg.vector_norm(matrix @ solution + residual) < 1e-12
    assert audit["method"] == "torch.linalg.solve dense pivoted LU"


def test_minimum_norm_subgradient_is_clipped_on_the_current_segment():
    minus = torch.tensor([2.0, 0.0], dtype=torch.float64)
    plus = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    theta, mixed, _ = probe._min_norm_mix(minus, plus)
    assert theta == pytest.approx(2 / 3)
    torch.testing.assert_close(mixed, torch.zeros_like(mixed), atol=1e-15, rtol=0)

    theta, mixed, _ = probe._min_norm_mix(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0]))
    assert theta == 1.0
    torch.testing.assert_close(mixed, torch.tensor([1.0, 0.0]))


def test_660_second_resource_classifier_respects_wall_rss_and_monitor_outcomes():
    good = {"wall_limit_seconds": 660.0, "rss_limit_bytes": 1024**3,
        "exit_code": 0, "resource_termination": None, "monitor_error": None,
        "received_sigterm": False, "elapsed_seconds": 659.0,
        "sampled_peak_rss_bytes": 900_000_000}
    assert probe._execution_status(good) == "completed"
    assert probe._execution_status({**good, "elapsed_seconds": 661.0}) == "wall_timeout"
    assert probe._execution_status({**good, "resource_termination": "rss_limit"}) == "rss_limit"
    assert probe._execution_status({**good, "resource_termination": "cancelled"}) == "cancelled"
    assert probe._execution_status({**good, "resource_termination": "rss_monitor_unavailable"}) == "monitor_failure"


def test_candidate_geometry_keeps_zero_face_and_masks_only_a_sign_copy():
    control = torch.zeros(26, dtype=torch.float64)
    control[20] = torch.atanh(torch.tensor(0.1, dtype=torch.float64))
    control[21] = torch.atanh(torch.tensor(-0.1, dtype=torch.float64))
    weights = torch.tensor([1.0, 1.0, -0.5, 0.25, 0.75], dtype=torch.float64)
    pivot = 20
    base = probe._chart(control, weights, pivot, 0.0)
    delta = torch.zeros(27, dtype=torch.float64)
    delta[0] = 0.2
    delta[-1] = -0.1
    alpha = 0.5

    candidate, eta = probe._candidate_path(SimpleNamespace(), base, weights, pivot, 0.0, delta, alpha)
    assert eta == 0.0
    assert float(probe._face_value(candidate, weights)) == pytest.approx(0.0, abs=1e-15)
    assert torch.equal(candidate[1:20], base[1:20])
    assert torch.equal(candidate[25:], base[25:])
    assert probe.policy()["root_claim"] is False
    assert probe.policy()["minimum_claim"] is False
    assert probe.policy()["response_claim"] is False

    flux = torch.ones((6, 5), dtype=torch.float64)
    flux[4, 3] = 0.0
    before = flux.clone()
    copied = probe._face_sign_copy(flux, "y")
    assert copied[4][3] is None
    assert torch.equal(flux, before)


def test_analysis_trace_records_exact_360_stages_and_excludes_only_target_sign(monkeypatch):
    q = torch.arange(30, dtype=torch.float64).reshape(6, 5).square() + 1
    qx = torch.ones((6, 6), dtype=torch.float64)
    qy = torch.ones((7, 5), dtype=torch.float64)
    qy[4, 3] = 0.0

    class ToyProblem:
        def objective(self, _control, _parameters):
            for _ in range(probe.ANALYSIS_STAGES):
                callbacks[-1](q, qx, qy)
            return q.sum()

    callbacks = []

    @contextmanager
    def observer(callback):
        callbacks.append(callback)
        yield

    monkeypatch.setattr(transport, "observe_minmod_stages", observer)
    point = torch.zeros(26, dtype=torch.float64)
    trace = probe._analysis_trace(ToyProblem(), point, torch.zeros(1, dtype=torch.float64), 1)
    assert trace["stage_count"] == trace["observed_stage_count"] == 360
    assert trace["face_signs"][0]["y"][4][3] is None
    assert all(item == 0.0 for item in trace["target_fluxes"])
    assert trace["excluded_face_from_sign_gate"]["diagnostic_copy_only"] is True
    assert float(qy[4, 3]) == 0.0


def test_source_scope_fixture_allows_only_the_declared_transport_snapshot():
    import json

    inherited = json.loads(probe.PRODUCER_PLAN.read_text())
    manifest = json.loads(probe.SOURCE_MANIFEST.read_text())
    sources = dict(inherited["source_files"])
    transport_path = "src/advar/transport.py"
    sources[transport_path] = probe._sha(probe.ROOT / transport_path)
    for name in (probe.SELF, probe.TEST, probe.CORE_TEST):
        sources[name] = probe._sha(probe.ROOT / name)
    archives = dict(inherited["archive_files"])
    archives[probe.PRODUCER_PLAN.relative_to(probe.ROOT).as_posix()] = probe.PRODUCER_PLAN_SHA
    for path in (probe.BASE_STEP, probe.BASE_RUN, probe.BASE_RESOURCE, probe.SOURCE_MANIFEST):
        archives[path.relative_to(probe.ROOT).as_posix()] = probe._sha(path)
    snapshot = manifest["snapshots"][transport_path]
    archives[snapshot["archive_path"]] = snapshot["sha256"]
    plan = {"source_files": sources, "archive_files": archives,
            "producer_source_snapshots": manifest["snapshots"]}

    probe._source_maps_valid(plan)
    invalid = {**plan, "source_files": dict(sources)}
    invalid["source_files"]["examples/weather_scenarios/fv_minmod_inverse_probe.py"] = "0" * 64
    with pytest.raises(ValueError, match="inherited source pin changed"):
        probe._source_maps_valid(invalid)


def _toy_child(monkeypatch, *, source_drift=False):
    control = torch.zeros(26, dtype=torch.float64)
    control[20] = 0.2
    parameters = torch.zeros(1, dtype=torch.float64)
    original = control.clone()
    truth = torch.tensor([1.0], dtype=torch.float64)
    base_sha = probe._tensor_sha(control)
    monkeypatch.setattr(probe, "BASE_CONTROL_SHA", base_sha)
    class ToyProblem:
        def objective(self, c, _p):
            selected = transport._selected_face.get()
            flux = torch.tanh(c[20])
            kink = selected.side * flux if selected is not None else flux.abs()
            return 0.5 * torch.dot(c, c) - 0.9 * c[0] + kink

        def branch_check(self, _c, _p):
            return {"choices": [[1]], "face_signs": [[1]]}, "toy complete branch"

    problem = ToyProblem()
    parameters_sha = probe._tensor_sha(parameters)
    truth_sha = probe._tensor_sha(truth)
    identity = {"control_sha256": base_sha, "parameters_sha256": parameters_sha,
                "terminal_truth_sha256": truth_sha, "archived_input": "frozen"}
    runtime = {"device": "CPU FP64", "torch": "toy", "python": "toy"}
    state = {"objective": float(problem.objective(control, parameters)),
             "gradient": [-0.9] + [0.0] * 19 + [1.2] + [0.0] * 5,
             "branch": {"signature_sha256": probe._canonical_sha(
                 {"choices": [[1]], "face_signs": [[1]]})}}
    raw = {"parameters_sha256": parameters_sha}
    base = {"control": control, "input_identity": identity, "runtime": runtime,
            "raw": raw, "state": state, "expected_objective": state["objective"],
            "expected_gradient": torch.tensor(state["gradient"], dtype=torch.float64)}
    plan = {"source_files": {}, "archive_files": {}}
    monkeypatch.setattr(probe, "_load_plan", lambda *_: plan)
    monkeypatch.setattr(probe, "_load_base", lambda *_: base)
    monkeypatch.setattr(probe.seed, "_prepare_fixed_seed",
        lambda: (problem, original, control, parameters, truth, identity))
    monkeypatch.setattr(probe, "_input_identity", lambda _problem, _original, c, _p, _truth:
        {"control_sha256": probe._tensor_sha(c), "parameters_sha256": parameters_sha,
         "terminal_truth_sha256": truth_sha, "archived_input": "frozen"})
    monkeypatch.setattr(probe, "_runtime", lambda: runtime)
    monkeypatch.setattr(probe.face_geometry.model, "_fixed_input",
        lambda candidate, base_identity, sha: all(candidate.get(k) == base_identity.get(k)
            for k in ("parameters_sha256", "terminal_truth_sha256", "archived_input"))
            and candidate.get("control_sha256") == sha)
    calls = {"count": 0}
    def source_hashes(_plan, plan_path, plan_sha):
        calls["count"] += 1
        value = {plan_path.relative_to(probe.ROOT).as_posix(): plan_sha}
        if source_drift and calls["count"] == 2:
            value["mutated.py"] = "changed"
        return value
    monkeypatch.setattr(probe, "_source_hashes", source_hashes)
    monkeypatch.setattr(probe, "_face_weights", lambda *_args, **_kwargs:
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    monkeypatch.setattr(probe, "_production_face_value", lambda _p, c, **_kw: torch.tanh(c[20]))
    monkeypatch.setattr(probe, "_face_audit", lambda _p, c, _w, _face, _eta:
        (torch.tanh(c[20]), 128 * torch.finfo(c.dtype).eps, True))
    monkeypatch.setattr(probe, "_analysis_trace", lambda *_args, **_kwargs: {
        "stage_count": 360, "observed_stage_count": 360, "expected_stage_count": 360,
        "choices": [[1]], "face_signs": [[1]], "target_fluxes": [0.0],
        "minimum_margins": {}, "maximum_face_flux": 1.0,
        "nonfinite_or_tie": False, "signature_sha256": "toy"})
    # Make the endpoint's recorded gradient consistent with the actual branch.
    state["gradient"] = torch.func.grad(problem.objective, argnums=0)(control, parameters).tolist()
    base["expected_gradient"] = torch.tensor(state["gradient"], dtype=torch.float64)
    base["input_identity"] = {"control_sha256": base_sha, "parameters_sha256": parameters_sha,
        "terminal_truth_sha256": truth_sha, "archived_input": "frozen"}
    return plan, control, parameters


def test_toy_child_acceptance_commits_only_after_closed_final_repeat(tmp_path, monkeypatch):
    _toy_child(monkeypatch)
    output = tmp_path / "child.json"
    plan_path = probe.ROOT / "toy-plan.json"
    result = probe._run_child(plan_path, "plan-sha", output)
    assert result["execution_status"] == "completed"
    assert result["numerical_status"] == "one_nonsmooth_coupled_step_accepted"
    assert result["candidate_committed"] is True
    assert result["optimizer_steps_applied"] == 1
    assert result["candidate_type"] == "one_nonsmooth_coupled_step"
    assert result["full_smooth_root"] is False
    assert result["final_repeat"]["deadline_passed"] is True


def test_toy_child_source_drift_rejects_the_proposal_without_committing(tmp_path, monkeypatch):
    _toy_child(monkeypatch, source_drift=True)
    output = tmp_path / "child.json"
    result = probe._run_child(probe.ROOT / "toy-plan.json", "plan-sha", output)
    assert result["candidate_committed"] is False
    assert result["optimizer_steps_applied"] == 0
    assert result["current_control_sha256"] == probe.BASE_CONTROL_SHA
    assert result["numerical_status"] == "coupled_step_refused"


def test_toy_child_prerequisite_and_internal_deadline_failures_close_without_a_step(tmp_path, monkeypatch):
    _toy_child(monkeypatch)
    monkeypatch.setattr(probe, "_face_audit", lambda *_a, **_k:
        (torch.tensor(1.0), 1e-12, False))
    output = tmp_path / "prerequisite.json"
    result = probe._run_child(probe.ROOT / "toy-plan.json", "plan-sha", output)
    assert result["numerical_status"] == "face_qualification_refused"
    assert result["hvp_calls_started"] == 0
    assert result["source_unchanged"] and result["fixed_input_unchanged"]

    _toy_child(monkeypatch)
    monkeypatch.setattr(probe, "_counted_hvp", lambda *_a, **_k: (_ for _ in ()).throw(
        TimeoutError("toy internal deadline")))
    output = tmp_path / "timeout.json"
    result = probe._run_child(probe.ROOT / "toy-plan.json", "plan-sha", output)
    assert result["numerical_status"] == "internal_deadline_refused"
    assert result["execution_status"] == "completed"
    assert result["candidate_committed"] is False
    assert result["optimizer_steps_applied"] == 0
    assert result["fixed_input_unchanged"] and result["runtime_unchanged"]



def test_real_pr264_receipt_uses_its_own_resume_base_and_final_point():
    import json
    plan = {"archive_files": {
        p.relative_to(probe.ROOT).as_posix(): probe._sha(p)
        for p in (probe.BASE_STEP, probe.BASE_RUN, probe.BASE_RESOURCE)
    }}
    base = probe._load_base(plan)
    producer = json.loads(probe.PRODUCER_PLAN.read_text())
    assert base["raw"]["base_control_sha256"] == producer["base_control_sha256"]
    assert base["raw"]["base_control_sha256"].startswith("6c0fe485")
    assert probe._tensor_sha(base["control"]) == probe.BASE_CONTROL_SHA
