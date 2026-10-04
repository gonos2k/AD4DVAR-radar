"""Toy subprocess tests for cancellation-safe resource-guard plumbing."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from examples.weather_scenarios.fv_diagnostic_guard import (
    atomic_write_text, run_guarded_diagnostic,
)


def _toy_command(marker: Path, *, sleep_seconds: float = 10.0) -> list[str]:
    code = (f"import os,time; open({str(marker)!r},'w').write(str(os.getpid())); "
            f"time.sleep({sleep_seconds!r})")
    return [sys.executable, "-c", code]


@pytest.mark.parametrize("log_kind", ("existing", "dangling_symlink"))
def test_existing_or_dangling_log_is_rejected_without_clobber_or_child(tmp_path: Path,
                                                                      log_kind: str):
    report = tmp_path / "audit.resource.json"
    log = tmp_path / "audit.log"
    target = tmp_path / "protected.txt"
    target.write_text("original content")
    if log_kind == "existing":
        log.write_text("prior log content")
    else:
        target = tmp_path / "missing-log-target.txt"
        log.symlink_to(target)
    marker = tmp_path / "child.started"

    with pytest.raises(FileExistsError):
        run_guarded_diagnostic(_toy_command(marker), wall_seconds=3, rss_bytes=10**9,
                               report_path=report, log_path=log)

    assert not marker.exists()
    assert not report.exists()
    if log_kind == "existing":
        assert target.read_text() == "original content"
        assert log.read_text() == "prior log content"
    else:
        assert log.is_symlink() and log.resolve() == target
        assert not target.exists()


@pytest.mark.parametrize("report_kind", ("existing", "dangling_symlink"))
def test_existing_or_dangling_report_path_is_rejected_before_child(tmp_path: Path,
                                                                  report_kind: str):
    report = tmp_path / "audit.resource.json"
    log = tmp_path / "audit.log"
    target = tmp_path / "protected.txt"
    target.write_text("original content")
    if report_kind == "existing":
        report.write_text("prior report content")
    else:
        target = tmp_path / "missing-report-target.txt"
        report.symlink_to(target)
    marker = tmp_path / "child.started"

    with pytest.raises(FileExistsError):
        run_guarded_diagnostic(_toy_command(marker), wall_seconds=3, rss_bytes=10**9,
                               report_path=report, log_path=log)

    assert not marker.exists()
    assert not log.exists()
    if report_kind == "existing":
        assert target.read_text() == "original content"
        assert report.read_text() == "prior report content"
    else:
        assert report.is_symlink() and report.resolve() == target
        assert not target.exists()


def test_atomic_text_writer_ignores_preexisting_temp_symlink(tmp_path: Path):
    output = tmp_path / "new-report.txt"
    user_input = tmp_path / "user-input.txt"
    temporary = output.with_suffix(output.suffix + ".tmp")
    user_input.write_text("do not overwrite")
    temporary.symlink_to(user_input)

    assert atomic_write_text(output, "new content") == len("new content")

    assert output.read_text() == "new content"
    assert user_input.read_text() == "do not overwrite"
    assert temporary.is_symlink() and temporary.resolve() == user_input


def test_sigterm_cancels_and_reaps_child_and_restores_prior_handler(tmp_path: Path):
    marker = tmp_path / "child.pid"
    report = tmp_path / "resource.json"
    log = tmp_path / "child.log"
    result_path = tmp_path / "wrapper-result.json"
    child_code = (f"import os,time; open({str(marker)!r},'w').write(str(os.getpid())); "
                  "time.sleep(30)")
    inner = f"""
import json, signal, sys
from pathlib import Path
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic
def previous_handler(_signum, _frame):
    pass
signal.signal(signal.SIGTERM, previous_handler)
result = run_guarded_diagnostic([sys.executable, '-c', {child_code!r}],
    wall_seconds=60, rss_bytes=10**9, report_path=Path({str(report)!r}),
    log_path=Path({str(log)!r}))
print(json.dumps({{'result': result,
    'previous_handler_restored': signal.getsignal(signal.SIGTERM) is previous_handler}}))
"""
    wrapper = subprocess.Popen([sys.executable, "-c", inner], cwd=Path(__file__).resolve().parents[1],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            if wrapper.poll() is not None:
                stdout, stderr = wrapper.communicate()
                raise AssertionError(f"wrapper exited before child start: {stdout}\n{stderr}")
            time.sleep(0.02)
        assert marker.exists(), "toy child failed to start"
        child_pid = int(marker.read_text())
        wrapper.send_signal(signal.SIGTERM)
        stdout, stderr = wrapper.communicate(timeout=8)
        assert wrapper.returncode == 0, f"wrapper failed: {stdout}\n{stderr}"
        result = json.loads(stdout.strip())
        assert result["previous_handler_restored"] is True
        assert result["result"]["received_sigterm"] is True
        assert result["result"]["resource_termination"] == "cancelled"
        assert result["result"]["child_process_group_cleanup_sent"] is False
        persisted = json.loads(report.read_text())
        assert persisted == result["result"]
        assert persisted["resource_termination"] == "cancelled"
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait()
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cancel_callback_is_composed_with_signal_check(tmp_path: Path):
    marker = tmp_path / "child.pid"
    report = tmp_path / "resource.json"
    result = run_guarded_diagnostic(_toy_command(marker), wall_seconds=3, rss_bytes=10**9,
                                    report_path=report, log_path=tmp_path / "child.log",
                                    cancel_requested=lambda: marker.exists())
    assert result["received_sigterm"] is False
    assert result["resource_termination"] == "cancelled"
    assert marker.exists()
    assert json.loads(report.read_text()) == result


def test_normal_resource_report_matches_returned_legacy_fields(tmp_path: Path):
    report = tmp_path / "resource.json"
    result = run_guarded_diagnostic([sys.executable, "-c", "print('toy child')"],
                                    wall_seconds=3, rss_bytes=10**9,
                                    report_path=report, log_path=tmp_path / "child.log")
    assert result["received_sigterm"] is False
    assert result["child_process_group_cleanup_sent"] is False
    assert result["exit_code"] == 0 and result["resource_termination"] is None
    assert json.loads(report.read_text()) == result


def _process_running(pid: int) -> bool:
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                           capture_output=True, text=True, check=False).stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_sigterm_kills_residual_grandchild_after_leader_exits(tmp_path: Path):
    grandchild_marker = tmp_path / "grandchild.pid"
    report = tmp_path / "resource.json"
    log = tmp_path / "child.log"
    grandchild_code = ("import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                       f"open({str(grandchild_marker)!r},'w').write(str(os.getpid())); time.sleep(30)")
    leader_code = ("import subprocess,sys,time; "
                   f"subprocess.Popen([sys.executable,'-c',{grandchild_code!r}]); time.sleep(30)")
    inner = f"""
import json, signal, sys
from pathlib import Path
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic
def prior_handler(_signum, _frame):
    pass
signal.signal(signal.SIGTERM, prior_handler)
result = run_guarded_diagnostic([sys.executable, '-c', {leader_code!r}],
    wall_seconds=60, rss_bytes=10**9, report_path=Path({str(report)!r}),
    log_path=Path({str(log)!r}))
print(json.dumps({{'result': result,
    'previous_handler_restored': signal.getsignal(signal.SIGTERM) is prior_handler}}))
"""
    wrapper = subprocess.Popen([sys.executable, "-c", inner], cwd=Path(__file__).resolve().parents[1],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    group_pid: int | None = None
    grandchild_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while not grandchild_marker.exists() and time.monotonic() < deadline:
            if wrapper.poll() is not None:
                stdout, stderr = wrapper.communicate()
                raise AssertionError(f"wrapper exited before grandchild start: {stdout}\n{stderr}")
            time.sleep(0.02)
        assert grandchild_marker.exists(), "toy grandchild failed to start"
        grandchild_pid = int(grandchild_marker.read_text())
        wrapper.send_signal(signal.SIGTERM)
        stdout, stderr = wrapper.communicate(timeout=8)
        assert wrapper.returncode == 0, f"wrapper failed: {stdout}\n{stderr}"
        result = json.loads(stdout.strip())
        resource = result["result"]
        group_pid = resource["child_pid"]
        assert result["previous_handler_restored"] is True
        assert resource["resource_termination"] == "cancelled"
        assert resource["child_process_group_cleanup_sent"] is True
        assert json.loads(report.read_text()) == resource
        deadline = time.monotonic() + 3
        while _process_running(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.03)
        assert not _process_running(grandchild_pid)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait()
        if group_pid is not None:
            try:
                os.killpg(group_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_signal_support_requires_main_thread(tmp_path: Path):
    result: list[BaseException] = []

    def invoke() -> None:
        try:
            run_guarded_diagnostic(_toy_command(tmp_path / "child.pid"), wall_seconds=1,
                                   rss_bytes=10**9, report_path=tmp_path / "resource.json",
                                   log_path=tmp_path / "child.log")
        except BaseException as error:
            result.append(error)

    worker = threading.Thread(target=invoke)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(result) == 1 and isinstance(result[0], ValueError)
    assert "main thread" in str(result[0])
    assert not (tmp_path / "resource.json").exists()


@pytest.mark.parametrize(("wall_seconds", "rss_bytes"), [
    (float("nan"), 10**9), (float("inf"), 10**9), (0.0, 10**9),
    (True, 10**9), (3.0, 0), (3.0, -1), (3.0, False), (3.0, 1.5),
])
def test_invalid_limits_refuse_before_outputs_or_child(tmp_path: Path,
                                                     wall_seconds: Any,
                                                     rss_bytes: Any):
    marker = tmp_path / "child.started"
    report = tmp_path / "resource.json"
    log = tmp_path / "child.log"
    with pytest.raises(ValueError, match="wall_seconds|rss_bytes"):
        run_guarded_diagnostic(_toy_command(marker), wall_seconds=wall_seconds,
                               rss_bytes=rss_bytes, report_path=report, log_path=log)
    assert not marker.exists() and not report.exists() and not log.exists()
