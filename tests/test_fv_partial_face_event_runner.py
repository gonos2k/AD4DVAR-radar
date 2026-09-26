"""Guard-runner regressions with archived controls and fake process guard."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples.weather_scenarios import fv_partial_face_event_runner as runner


def _child_report(identity: dict[str, Any], probe_sha: str,
                  input_identity: dict[str, Any], problem: Any) -> dict[str, Any]:
    expected = runner._expected_chords()
    archived_chords = runner._archived_chords()
    chord_records = []
    samples = []
    comparisons = []
    spec = problem.frozen.fv_transport
    assert spec is not None
    interval = problem.frozen.nowcast_config.interval_minutes

    for pin, archived in zip(expected, archived_chords):
        start, end = archived["start"], archived["end"]
        direction = end - start
        flux_at = lambda t: runner.probe._qy_20(
            start + t * direction, spec, interval
        )
        bracket = runner.probe.bisect_zero(flux_at)
        event_t = (bracket["left_t"] + bracket["right_t"]) * 0.5
        chord_records.append({
            **pin,
            "face": {"kind": "qy", "index": [2, 0]},
            "start_face_flux": flux_at(0.0),
            "end_face_flux": flux_at(1.0),
            "event_bracket": bracket,
            "event_parameter": event_t,
            "event_face_flux": flux_at(event_t),
            "event_derivative_evaluated": False,
        })
        for side, sign in (("left", -1), ("right", 1)):
            def values(rows: int, columns: int, value: Any) -> list[list[Any]]:
                return [[value for _ in range(columns)] for _ in range(rows)]

            signature = {
                "choices": [[
                    {"choose_left": values(2, 3, True), "slope_sign": values(2, 3, 1),
                     "left_sign": values(2, 3, 1), "right_sign": values(2, 3, 1)},
                    {"choose_left": values(2, 3, False), "slope_sign": values(2, 3, -1),
                     "left_sign": values(2, 3, -1), "right_sign": values(2, 3, -1)},
                ] for _ in range(54)],
                "face_signs": [
                    {"qx": values(4, 6, 1), "qy": values(5, 5, sign)}
                    for _ in range(54)
                ],
            }
            signature_sha = hashlib.sha256(json.dumps(
                signature, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()).hexdigest()
            side_samples = []
            for point in runner.probe.event_side_schedule(event_t):
                if point["side"] != side:
                    continue
                common = {
                    "chord": pin["label"], "side": side,
                    "offset": point["offset"], "t": point["t"],
                }
                if not point["available"]:
                    sample = {
                        **common, "status": "unavailable_outside_segment", "finite": None,
                    }
                else:
                    control = start + float(point["t"]) * direction
                    sample = {
                        **common,
                        "control_sha256": runner.probe._tensor_sha(control),
                        "status": "finite_off_event_derivatives", "finite": True,
                        "euler_stages": 54,
                        "minimum_scaled_slope_margin": 0.001,
                        "minimum_scaled_face_flux_margin": 0.0002,
                        "face_margins_by_stage": [0.0002] * 54,
                        "response_margin_qualified": True,
                        "branch_signature": signature,
                        "branch_signature_sha256": signature_sha,
                        "objective": 2.0, "gradient_inf": 0.01,
                        "gradient_l2": 0.02, "gradient_merit": 0.0002,
                        "gradient_dot_chord": 0.01,
                        "hessian_times_chord": [0.001] * 26,
                        "merit_directional_derivative": 0.0001,
                    }
                samples.append(sample)
                side_samples.append(sample)
            if all("branch_signature_sha256" in sample for sample in side_samples):
                digest_by_offset = {
                    str(sample["offset"]): sample["branch_signature_sha256"]
                    for sample in side_samples
                }
                same = len(set(digest_by_offset.values())) == 1
                comparisons.append({
                    "chord": pin["label"], "side": side,
                    "status": "same_signature" if same else "signature_changed",
                    "signature_sha256_by_offset": digest_by_offset,
                    "selected_face_attribution": (
                        "not_confounded_by_signature_change" if same
                        else "confounded_by_other_branch_changes"
                    ),
                })
            else:
                comparisons.append({
                    "chord": pin["label"], "side": side,
                    "status": "unsupported_sample",
                    "selected_face_attribution": "undetermined",
                })

    return {
        "pid": 123,
        "status": "face_event_diagnostic_only",
        "plan_sha256": runner.PLAN_SHA256,
        "reviewed_probe_sha256": probe_sha,
        "archive_manifest_sha256": runner.probe.ARCHIVE_MANIFEST_SHA256,
        "source_before": identity["sources"],
        "source_after": identity["sources"],
        "input_before": input_identity,
        "input_after": input_identity,
        "chord_count": 2,
        "chords": chord_records,
        "off_event_sample_count": 8,
        "off_event_samples": samples,
        "same_side_offset_comparisons": comparisons,
        "event_derivatives_evaluated": False,
        "new_gn_root_adjoint_reanalysis_runs": 0,
        "root_claim": False,
        "response_claim": False,
        "response_validation": "not_performed",
        "physical_validation": "not_performed",
    }


@pytest.fixture(scope="module")
def current_context():
    problem = runner._current_problem()
    input_identity = runner._current_input_identity()
    assert problem.identity == input_identity["current_problem_identity"]
    probe_sha = runner.probe._sha(Path(runner.probe.__file__))
    runner_sha = runner.probe._sha(Path(runner.__file__))
    identity = runner._identity(
        expected_plan_sha256=runner.PLAN_SHA256,
        expected_probe_sha256=probe_sha,
        expected_runner_sha256=runner_sha,
    )
    report = _child_report(identity, probe_sha, input_identity, problem)
    return {
        "problem": problem, "input_identity": input_identity,
        "probe_sha": probe_sha, "runner_sha": runner_sha,
        "identity": identity, "report": report,
    }


def _stub_context(monkeypatch, context):
    monkeypatch.setattr(runner, "_current_problem", lambda: context["problem"])
    monkeypatch.setattr(runner, "_current_input_identity", lambda: context["input_identity"])


def _resource(command, *, exit_code=0, termination=None):
    return {
        "command": command,
        "child_pid": 123,
        "exit_code": exit_code,
        "resource_termination": termination,
        "monitor_error": None,
        "elapsed_seconds": 1.0,
        "sampled_peak_rss_bytes": 4096,
        "rss_samples": 1,
        "wall_limit_seconds": runner.WALL_SECONDS,
        "rss_limit_bytes": runner.SAMPLED_RSS_BYTES,
    }


def test_runner_launches_exact_guarded_command_and_records_completion(
    tmp_path, monkeypatch, current_context,
):
    _stub_context(monkeypatch, current_context)
    calls = []

    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path):
        calls.append((command, wall_seconds, rss_bytes, report_path, log_path))
        assert command[command.index("--expected-plan-sha256") + 1] == runner.PLAN_SHA256
        assert command[command.index("--expected-probe-sha256") + 1] == current_context["probe_sha"]
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(current_context["report"]))
        return _resource(command)

    monkeypatch.setattr(runner, "run_guarded", fake_guard)
    result = runner.run(
        tmp_path / "attempt", expected_plan_sha256=runner.PLAN_SHA256,
        expected_probe_sha256=current_context["probe_sha"],
        expected_runner_sha256=current_context["runner_sha"],
    )

    assert len(calls) == 1
    assert calls[0][1:3] == (300, 1024**3)
    assert result["execution_status"] == "completed"
    assert result["numerical_status"] == "face_event_diagnostic_only"
    assert (tmp_path / "attempt/partial_face_event.run.json").exists()


@pytest.mark.parametrize(
    ("exit_code", "write_child", "expected_status"),
    [(1, False, "child_exception"), (2, True, "child_nonzero_exit")],
)
def test_runner_distinguishes_child_exception_from_nonzero_exit(
    tmp_path, monkeypatch, current_context, exit_code, write_child, expected_status,
):
    _stub_context(monkeypatch, current_context)

    def fake_guard(command, **_kwargs):
        if write_child:
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps(current_context["report"]))
        return _resource(command, exit_code=exit_code)

    monkeypatch.setattr(runner, "run_guarded", fake_guard)
    result = runner.run(
        tmp_path / "attempt", expected_plan_sha256=runner.PLAN_SHA256,
        expected_probe_sha256=current_context["probe_sha"],
        expected_runner_sha256=current_context["runner_sha"],
    )
    assert result["execution_status"] == expected_status
    assert json.loads((tmp_path / "attempt/partial_face_event.run.json").read_text()) == result


def test_runner_rejects_stale_plan_or_source_before_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_guarded", lambda *_a, **_k: pytest.fail("child launched"))
    with pytest.raises(ValueError, match="reviewed source or plan changed"):
        runner.run(tmp_path / "attempt", expected_plan_sha256="stale",
                   expected_probe_sha256="stale", expected_runner_sha256="stale")


def test_runner_refuses_nonempty_or_archive_output_directory(
    tmp_path, monkeypatch, current_context,
):
    monkeypatch.setattr(runner, "run_guarded", lambda *_a, **_k: pytest.fail("child launched"))
    _stub_context(monkeypatch, current_context)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.json").write_text("keep")
    expected = {
        "expected_plan_sha256": runner.PLAN_SHA256,
        "expected_probe_sha256": current_context["probe_sha"],
        "expected_runner_sha256": current_context["runner_sha"],
    }
    with pytest.raises(ValueError, match="empty"):
        runner.run(existing, **expected)
    with pytest.raises(ValueError, match="outside frozen raw archive"):
        runner.run(runner.probe.ARCHIVE / "attempt", **expected)
    assert (existing / "keep.json").read_text() == "keep"


def test_runner_marks_source_drift_after_child_as_identity_changed(
    tmp_path, monkeypatch, current_context,
):
    _stub_context(monkeypatch, current_context)
    calls = 0

    def identity_changes(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return current_context["identity"]
        return {**current_context["identity"], "runner_sha256": "changed"}

    monkeypatch.setattr(runner, "_identity", identity_changes)

    def fake_guard(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(current_context["report"]))
        return _resource(command)

    monkeypatch.setattr(runner, "run_guarded", fake_guard)
    result = runner.run(
        tmp_path / "attempt", expected_plan_sha256=runner.PLAN_SHA256,
        expected_probe_sha256=current_context["probe_sha"],
        expected_runner_sha256=current_context["runner_sha"],
    )
    assert result["execution_status"] == "identity_changed"
    assert result["identities_unchanged"] is False


@pytest.mark.parametrize("mutation", [
    "empty_chord", "empty_sample", "forged_margin", "forged_comparison",
    "forged_input", "forged_face", "forged_sample_hash",
])
def test_runner_rejects_forged_child(tmp_path, monkeypatch, current_context, mutation):
    _stub_context(monkeypatch, current_context)

    def fake_guard(command, **_kwargs):
        report = copy.deepcopy(current_context["report"])
        if mutation == "empty_chord":
            report["chords"][0] = {}
        elif mutation == "empty_sample":
            report["off_event_samples"][0] = {}
        elif mutation == "forged_margin":
            report["off_event_samples"][0]["response_margin_qualified"] = False
        elif mutation == "forged_comparison":
            report["same_side_offset_comparisons"][0][
                "selected_face_attribution"
            ] = "confounded_by_other_branch_changes"
        elif mutation == "forged_input":
            report["input_before"] = {"fixed": "forged"}
            report["input_after"] = {"fixed": "forged"}
        elif mutation == "forged_face":
            report["chords"][0]["start_face_flux"] *= -1.0
        else:
            report["off_event_samples"][0]["control_sha256"] = "0" * 64
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(report))
        return _resource(command)

    monkeypatch.setattr(runner, "run_guarded", fake_guard)
    result = runner.run(
        tmp_path / mutation, expected_plan_sha256=runner.PLAN_SHA256,
        expected_probe_sha256=current_context["probe_sha"],
        expected_runner_sha256=current_context["runner_sha"],
    )
    assert result["execution_status"] == "child_result_invalid"


@pytest.mark.parametrize("mutation", ["shape", "sign_value"])
def test_signature_validator_rejects_nonphysical_4x5_signature(mutation, current_context):
    sample = copy.deepcopy(current_context["report"]["off_event_samples"][0])
    assert runner._valid_signature(sample)
    if mutation == "shape":
        sample["branch_signature"]["choices"][0][0]["choose_left"] = [[True]]
    else:
        sample["branch_signature"]["face_signs"][0]["qy"][2][0] = 0
    sample["branch_signature_sha256"] = hashlib.sha256(json.dumps(
        sample["branch_signature"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    assert not runner._valid_signature(sample)
