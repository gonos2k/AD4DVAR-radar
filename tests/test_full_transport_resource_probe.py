"""Focused tests for the isolated transport worker resource gate."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "examples/weather_scenarios" / \
    "full_transport_resource_probe.py"
SPEC = importlib.util.spec_from_file_location("full_transport_resource_probe", SOURCE)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _worker_result(scheme="donorcell", operation="forward", replay=False):
    return {
        "status": "passed",
        "scheme": scheme,
        "operation": operation,
        "seconds": 1.0,
        "peak_rss_bytes": 1024,
        "output_shape": [18, 128, 128],
        "product_bytes": 1024,
        "output_sum": 2.0,
        "python": "3.12.0",
        "torch": "2.8.0",
        "machine": "test",
        "dtype": "float64",
        "threads": 1,
        "steps": 18 * 39,
        "flow_controls": 24,
        "replay": replay,
        "directional_score": None,
    }


class _ExitsOnSecondPoll:
    pid = 1234
    returncode = 0

    def __init__(self):
        self.polls = 0
        self.terminated = False

    def poll(self):
        self.polls += 1
        return None if self.polls == 1 else 0

    def terminate(self):
        self.terminated = True


def test_post_exit_wall_time_is_still_a_failure():
    process = _ExitsOnSecondPoll()
    with patch.object(probe.subprocess, "run", return_value=SimpleNamespace(stdout="0")), \
            patch.object(probe.time, "perf_counter", side_effect=[119.9, 121.0]), \
            patch.object(probe.time, "sleep"):
        sampled_peak, exceeded, elapsed, returncode = probe._monitor_worker(process, 0.0)

    assert sampled_peak == 0
    assert exceeded == "wall_time"
    assert elapsed == 121.0
    assert returncode == 0
    assert not process.terminated


def test_missing_worker_output_cannot_reuse_stale_report(tmp_path):
    output = tmp_path / "report.json"
    stale = output.with_name("report-donorcell-forward.json")
    stale.write_text(json.dumps(_worker_result()) + "\n")

    class NoOutputProcess:
        pid = 1234
        returncode = 0

        def poll(self):
            return 0

    with patch.object(probe.subprocess, "Popen", return_value=NoOutputProcess()):
        assert not probe.run(output, operations=("forward",))

    report = json.loads(output.read_text())
    assert not stale.exists()
    assert len(report["results"]) == 2
    assert all(row["status"] == "failed" for row in report["results"])
    assert all("worker result unavailable" in row["reason"]
               for row in report["results"])


def test_worker_result_rejects_missing_nonfinite_and_mismatched_evidence(tmp_path):
    cases = []

    missing = _worker_result()
    del missing["peak_rss_bytes"]
    cases.append((missing, "missing fields"))

    nonfinite = _worker_result()
    nonfinite["output_sum"] = float("nan")
    cases.append((nonfinite, "not finite"))

    mismatched = _worker_result(operation="jvp")
    cases.append((mismatched, "does not match"))

    wrong_dtype = _worker_result()
    wrong_dtype["dtype"] = "float32"
    cases.append((wrong_dtype, "dtype"))

    overlong = _worker_result()
    overlong["seconds"] = probe.WALL_LIMIT + 1
    cases.append((overlong, "seconds exceeds"))

    for field in ("peak_rss_bytes", "product_bytes"):
        zero_bytes = _worker_result()
        zero_bytes[field] = 0
        cases.append((zero_bytes, "must be positive"))

    for index, (payload, reason_text) in enumerate(cases):
        path = tmp_path / f"worker-{index}.json"
        path.write_text(json.dumps(payload, allow_nan=True) + "\n")
        result, reason = probe._load_worker_result(path, "donorcell", "forward", False)
        assert result is None
        assert reason_text in reason


def test_worker_result_accepts_the_worker_contract(tmp_path):
    path = tmp_path / "worker.json"
    expected = _worker_result()
    path.write_text(json.dumps(expected) + "\n")

    result, reason = probe._load_worker_result(path, "donorcell", "forward", False)

    assert reason is None
    assert result == expected
