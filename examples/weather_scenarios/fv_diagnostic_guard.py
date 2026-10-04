"""Cancellation-safe, exclusive-output wrapper for the legacy resource monitor."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import tempfile
import threading
import json
import math
from numbers import Real
from collections.abc import Callable, Sequence
from typing import Any, TextIO

from examples.weather_scenarios.fv86_resource_runner import run_guarded


def atomic_write_text(path: Path, text: str) -> int:
    """Atomically replace a text output through a fresh, exclusive temporary file."""
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent directory is missing: {path.parent}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            written = stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        return written
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


class _ExclusiveLogPath:
    """Duck path whose legacy ``open('w')`` request is exclusive."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def open(self, mode: str = "r", *args: Any, **kwargs: Any) -> TextIO:
        if mode != "w":
            raise ValueError("guard log only supports the legacy write-open contract")
        return self._path.open("x", *args, **kwargs)


class _ReservedAtomicReportPath:
    """Duck path replacing the legacy direct report write with atomic replacement."""

    def __init__(self, path: Path, identity: tuple[int, int]) -> None:
        self._path = Path(path)
        self._identity = identity
        self.written = False

    def write_text(self, text: str, *args: Any, **kwargs: Any) -> int:
        if args or kwargs:
            raise ValueError("guard report only supports the legacy text argument")
        written = atomic_write_text(self._path, text)
        self.written = True
        return written

    def remove_unused_reservation(self) -> None:
        if self.written:
            return
        try:
            current = self._path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) == self._identity:
            self._path.unlink()


def _reserve_report(path: Path) -> _ReservedAtomicReportPath:
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"resource report parent directory is missing: {path.parent}")
    with path.open("x") as stream:
        identity = os.fstat(stream.fileno())
    return _ReservedAtomicReportPath(path, (identity.st_dev, identity.st_ino))


def _kill_remaining_process_group(child_pid: int) -> bool:
    """Kill residual descendants after the legacy helper has reaped its leader."""
    if child_pid <= 0 or child_pid == os.getpgrp():
        raise ValueError("child process-group id is invalid or refers to the guard process group")
    try:
        os.killpg(child_pid, 0)
    except ProcessLookupError:
        return False
    try:
        os.killpg(child_pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    return True


def run_guarded_diagnostic(command: Sequence[str], *, wall_seconds: float, rss_bytes: int,
                           report_path: Path, log_path: Path,
                           cancel_requested: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Run the legacy monitor with SIGTERM cancellation and nonclobbering outputs.

    SIGTERM is converted to a flag so ``run_guarded`` observes cancellation in its
    poll loop, signals the child process group, and reaps its direct child. Residual
    descendants are killed after the legacy helper returns. The caller's prior
    handler is restored on every Python exit path.
    """
    if (isinstance(wall_seconds, bool) or not isinstance(wall_seconds, Real)
            or not math.isfinite(float(wall_seconds)) or float(wall_seconds) <= 0):
        raise ValueError("wall_seconds must be a finite positive real")
    if isinstance(rss_bytes, bool) or not isinstance(rss_bytes, int) or rss_bytes <= 0:
        raise ValueError("rss_bytes must be a positive integer")
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("guarded diagnostics must run on the main thread for SIGTERM handling")
    old_handler = signal.getsignal(signal.SIGTERM)
    state = {"received_sigterm": False}

    def on_sigterm(_signum: int, _frame: Any) -> None:
        state["received_sigterm"] = True

    report = None
    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        report = _reserve_report(Path(report_path))

        def cancelled() -> bool:
            return (state["received_sigterm"]
                    or (cancel_requested is not None and cancel_requested()))

        result = run_guarded(
            list(command), wall_seconds=wall_seconds, rss_bytes=rss_bytes,
            report_path=report, log_path=_ExclusiveLogPath(Path(log_path)),
            cancel_requested=cancelled,
        )
        final_result = {**result, "received_sigterm": state["received_sigterm"]}
        child_pid = final_result.get("child_pid")
        if isinstance(child_pid, bool) or not isinstance(child_pid, int):
            final_result.update(child_process_group_cleanup_sent=False,
                                child_process_group_cleanup_error="legacy helper returned no child PID")
            report.write_text(json.dumps(final_result, indent=2) + "\n")
            raise ValueError("legacy resource report lacks a child PID")
        try:
            cleanup_sent = _kill_remaining_process_group(child_pid)
        except OSError as error:
            final_result.update(child_process_group_cleanup_sent=False,
                                child_process_group_cleanup_error=str(error))
            report.write_text(json.dumps(final_result, indent=2) + "\n")
            raise RuntimeError("remaining child process-group cleanup failed") from error
        final_result.update(child_process_group_cleanup_sent=cleanup_sent,
                            child_process_group_cleanup_error=None)
        report.write_text(json.dumps(final_result, indent=2) + "\n")
        return final_result
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        if report is not None:
            report.remove_unused_reservation()
