"""A cancelled process cannot publish a partial FV local response."""

import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time

import pytest

from examples.weather_scenarios import fv_process_response_worker as worker
from examples.weather_scenarios import fv_response_job_lifecycle as jobs


def test_same_directory_second_coordinator_is_busy_before_worker(monkeypatch, tmp_path):
    entered, release = Event(), Event()
    calls = []
    def fake_job(directory, *, request_id, token):
        calls.append(request_id)
        entered.set()
        assert release.wait(5)
        return {"execution_status": "completed"}
    monkeypatch.setattr(jobs, "_run_job_locked", fake_job)
    outcomes = []
    first = Thread(target=lambda: outcomes.append(
        jobs.run_job(tmp_path / "same", request_id="first")))
    first.start()
    assert entered.wait(5)
    with pytest.raises(jobs.ResponseJobBusyError, match="already running"):
        jobs.run_job(tmp_path / "same", request_id="second")
    release.set()
    first.join(5)
    assert not first.is_alive()
    assert outcomes == [{"execution_status": "completed"}]
    assert calls == ["first"]
    assert (tmp_path / ".same.launch.lock").is_file()


def test_same_host_lock_blocks_another_process_then_releases(tmp_path):
    directory = tmp_path / "job"
    code = ("from pathlib import Path\n"
            "from examples.weather_scenarios import fv_response_job_lifecycle as jobs\n"
            "import sys\n"
            "with jobs._exclusive_job_launch(Path(sys.argv[1])):\n"
            "    print('acquired')\n")
    command = [sys.executable, "-c", code, str(directory)]
    with jobs._exclusive_job_launch(directory):
        blocked = subprocess.run(command, cwd=jobs.ROOT, capture_output=True,
                                 text=True, timeout=20, check=False)
        assert blocked.returncode != 0
        assert "ResponseJobBusyError" in blocked.stderr
        with jobs._exclusive_job_launch(tmp_path / "other"):
            pass
    acquired = subprocess.run(command, cwd=jobs.ROOT, capture_output=True,
                              text=True, timeout=20, check=False)
    assert acquired.returncode == 0, acquired.stderr
    assert acquired.stdout.strip() == "acquired"


def test_parent_path_aliases_contend_on_one_job_lock(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with jobs._exclusive_job_launch(real_parent / "job"):
        with pytest.raises(jobs.ResponseJobBusyError, match="already running"):
            with jobs._exclusive_job_launch(alias_parent / "job"):
                pass
    assert (real_parent / ".job.launch.lock").is_file()


def test_job_lock_releases_after_exception_and_rejects_symlinks(tmp_path):
    directory = tmp_path / "job"
    with pytest.raises(RuntimeError, match="injected"):
        with jobs._exclusive_job_launch(directory):
            raise RuntimeError("injected")
    with jobs._exclusive_job_launch(directory):
        pass
    lock_path = tmp_path / ".other.launch.lock"
    target = tmp_path / "user_data"
    target.write_text("preserve")
    lock_path.symlink_to(target)
    with pytest.raises(ValueError, match="lock cannot be a symlink"):
        with jobs._exclusive_job_launch(tmp_path / "other"):
            pass
    assert target.read_text() == "preserve"
    alias = tmp_path / "alias"
    directory.mkdir()
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="directory cannot be a symlink"):
        with jobs._exclusive_job_launch(alias):
            pass


def _archived_child():
    path = jobs.EVIDENCE / "process_response_attempt2/fv4x5.json"
    child = json.loads(path.read_text())
    identity, _ = jobs._identity()
    child.update(
        pid=123, executable=sys.executable,
        source_before=worker._hashes(), source_after=worker._hashes(),
        source_unchanged=True, inputs_unchanged=True,
        input_identity=identity, input_identity_after=identity,
    )
    duration = child["response_ended_monotonic"] - child["response_started_monotonic"]
    shift = time.monotonic() - duration - 2 - child["response_started_monotonic"]
    child["response_started_monotonic"] += shift
    child["response_ended_monotonic"] += shift
    monitor = child["response"]["pcg_monitor"]
    monitor["hvp_event_times"] = [value + shift for value in monitor["hvp_event_times"]]
    monitor["hvp_intervals"] = [[start + shift, end + shift]
                                for start, end in monitor["hvp_intervals"]]
    return child


def test_cancellation_token_serializes_publication(tmp_path):
    token = jobs.CancellationToken()
    assert token.cancel()
    assert token.is_cancelled()
    assert token.cancelled_at() is not None
    assert not token.publish_if_active(tmp_path / "never.json", {"value": 1})
    assert not (tmp_path / "never.json").exists()
    token2 = jobs.CancellationToken()
    assert token2.publish_if_active(tmp_path / "published.json", {"value": 2})
    assert token2.cancel() is False
    published = json.loads((tmp_path / "published.json").read_text())
    assert published["value"] == 2
    assert isinstance(published["publication_decision_monotonic"], float)


def test_cancellation_during_locked_publish_waits_and_cannot_retract(monkeypatch, tmp_path):
    token = jobs.CancellationToken()
    entered, release, cancel_returned = Event(), Event(), Event()
    original = jobs._atomic_json
    def paused_write(path, data):
        entered.set()
        assert release.wait(5)
        original(path, data)
    monkeypatch.setattr(jobs, "_atomic_json", paused_write)
    publication = []
    cancellation = []
    publisher = Thread(target=lambda: publication.append(
        token.publish_if_active(tmp_path / "published.json", {"value": 3})))
    def cancel_later():
        cancellation.append(token.cancel())
        cancel_returned.set()
    publisher.start()
    assert entered.wait(5)
    canceller = Thread(target=cancel_later)
    canceller.start()
    assert not cancel_returned.wait(.05)
    release.set()
    publisher.join(5)
    canceller.join(5)
    assert publication == [True] and cancellation == [False]
    assert json.loads((tmp_path / "published.json").read_text())["value"] == 3


def test_finished_worker_publishes_only_after_reap_and_validation(monkeypatch, tmp_path):
    child = _archived_child()
    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path,
                   cancel_requested):
        assert wall_seconds == 300 and rss_bytes == 768 * 1024**2
        assert Path(command[1]).is_absolute() and Path(command[1]).is_file()
        assert not cancel_requested()
        Path(command[-1]).write_text(json.dumps(child))
        Path(log_path).write_text("")
        resource = {"command": command, "child_pid": 123, "exit_code": 0,
                    "resource_termination": None, "monitor_error": None,
                    "rss_samples": 3, "sampled_peak_rss_bytes": 300_000_000,
                    "wall_limit_seconds": 300, "rss_limit_bytes": 768 * 1024**2,
                    "elapsed_seconds": 5.0}
        Path(report_path).write_text(json.dumps(resource))
        assert not (tmp_path / "success/published.json").exists()
        return resource
    monkeypatch.setattr(jobs, "run_guarded", fake_guard)
    result = jobs.run_job(tmp_path / "success", request_id="success")
    assert result["execution_status"] == "completed"
    assert result["numerical_status"] == "eligible"
    assert result["response_published"] is True
    published = json.loads((tmp_path / "success/published.json").read_text())
    assert published["worker_pid"] == 123
    assert published["worker_response_ended_monotonic"] <= published["child_completed_monotonic"]
    assert published["child_completed_monotonic"] <= published["publication_decision_monotonic"]
    assert result["published_monotonic"] >= published["publication_decision_monotonic"]
    assert not (tmp_path / "success/published.json.tmp").exists()


def test_cancelled_worker_with_partial_report_never_publishes(monkeypatch, tmp_path):
    token = jobs.CancellationToken()
    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path,
                   cancel_requested):
        Path(command[-1]).write_text(json.dumps({"status": "running", "pid": 456}))
        Path(log_path).write_text("")
        assert token.cancel()
        assert cancel_requested()
        resource = {"command": command, "child_pid": 456, "exit_code": -15,
                    "resource_termination": "cancelled", "monitor_error": None,
                    "rss_samples": 2, "sampled_peak_rss_bytes": 200_000_000,
                    "wall_limit_seconds": wall_seconds, "rss_limit_bytes": rss_bytes,
                    "elapsed_seconds": 0.2}
        Path(report_path).write_text(json.dumps(resource))
        return resource
    monkeypatch.setattr(jobs, "run_guarded", fake_guard)
    result = jobs.run_job(tmp_path / "cancel", request_id="cancel", token=token)
    assert result["execution_status"] == "cancelled"
    assert result["response_published"] is False
    assert result["numerical_status"] == "not_verified"
    assert result["response_validation"] == "not_performed"
    assert not (tmp_path / "cancel/published.json").exists()


def test_late_cancellation_before_publication_wins(monkeypatch, tmp_path):
    token = jobs.CancellationToken()
    child = _archived_child()
    def fake_guard(command, *, wall_seconds, rss_bytes, report_path, log_path,
                   cancel_requested):
        Path(command[-1]).write_text(json.dumps(child))
        Path(log_path).write_text("")
        assert token.cancel()
        assert cancel_requested()
        resource = {"command": command, "child_pid": 123, "exit_code": 0,
                    "resource_termination": None, "monitor_error": None,
                    "rss_samples": 3, "sampled_peak_rss_bytes": 300_000_000,
                    "wall_limit_seconds": wall_seconds, "rss_limit_bytes": rss_bytes,
                    "elapsed_seconds": 5.0}
        Path(report_path).write_text(json.dumps(resource))
        return resource
    monkeypatch.setattr(jobs, "run_guarded", fake_guard)
    result = jobs.run_job(tmp_path / "late", request_id="late", token=token)
    assert result["execution_status"] == "cancelled"
    assert result["response_published"] is False
    assert result["numerical_status"] == "eligible"
    assert not (tmp_path / "late/published.json").exists()


def test_pair_cancellation_is_local_and_only_success_is_published(monkeypatch, tmp_path):
    def fake_job(directory, *, request_id, token):
        directory.mkdir(parents=True)
        if request_id == "cancel":
            (directory / "worker.raw.json").write_text('{"status":')
            time.sleep(.05)
            assert not token.is_cancelled()
            (directory / "worker.raw.json").write_text(
                '{"status":"running","phase":"response","pid":202}\n')
            until = time.monotonic() + 5
            while not token.is_cancelled() and time.monotonic() < until:
                time.sleep(.01)
            assert token.is_cancelled()
            return {"execution_status": "cancelled", "response_published": False,
                    "resource": {"child_pid": 202, "resource_termination": "cancelled"},
                    "child_completed_monotonic": time.monotonic()}
        until = time.monotonic() + 5
        while not (directory.parent / "cancel/worker.raw.json").exists() and time.monotonic() < until:
            time.sleep(.01)
        (directory / "published.json").write_text('{"request_id":"complete"}\n')
        return {"execution_status": "completed", "response_published": True,
                "resource": {"child_pid": 101, "resource_termination": None},
                "child_completed_monotonic": time.monotonic()}
    monkeypatch.setattr(jobs, "run_job", fake_job)
    result = jobs.run_pair(tmp_path / "pair")
    assert result["execution_status"] == "completed"
    assert result["parseable_running_report_seen_before_cancellation"] is True
    assert result["cancellation_requested_before_worker_finished"] is True
    assert result["distinct_child_pids"] is True
    assert (tmp_path / "pair/complete/published.json").is_file()
    assert not (tmp_path / "pair/cancel/published.json").exists()


def test_worker_direct_cli_imports_from_unrelated_cwd(tmp_path):
    script = jobs.ROOT / "examples/weather_scenarios/fv_process_response_worker.py"
    completed = subprocess.run([sys.executable, str(script), "--help"],
                               cwd=tmp_path, capture_output=True, text=True,
                               timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert "--case" in completed.stdout and "--output" in completed.stdout
