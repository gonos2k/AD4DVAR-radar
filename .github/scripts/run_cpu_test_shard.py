#!/usr/bin/env python3
"""Collect, partition, and run one complete module shard of the CPU suite."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
SHARD_COUNT = 6
FILTER_OPTIONS = (
    "-k", "--keyword", "-m", "--markexpr", "--ignore", "--ignore-glob",
    "--deselect", "--lf", "--last-failed", "--lfnf",
    "--last-failed-no-failures", "--failed-first", "--ff", "--new-first",
    "--nf", "--stepwise", "--stepwise-skip", "--sw",
)


class ShardError(ValueError):
    """The inventory, shard assignment, or execution record is incomplete."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _sanitized_pytest_env() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    return environment


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
    temporary.replace(path)


def _filter_options(config: Any) -> list[str]:
    found: set[str] = set()
    configured = config.getini("addopts")
    if isinstance(configured, str):
        configured = configured.split()
    args = [*configured, *config.invocation_params.args]
    for argument in args:
        for option in FILTER_OPTIONS:
            if (argument == option or argument.startswith(option + "=")
                    or (option in {"-k", "-m"} and argument.startswith(option)
                        and len(argument) > len(option))):
                found.add(option)
    option_state = config.option
    for name, option in (("keyword", "-k"), ("markexpr", "-m"),
                         ("ignore", "--ignore"), ("ignore_glob", "--ignore-glob"),
                         ("deselect", "--deselect"), ("lf", "--lf"),
                         ("lfnf", "--lfnf"), ("failedfirst", "--failed-first"),
                         ("newfirst", "--new-first"), ("stepwise", "--stepwise"),
                         ("stepwise_skip", "--stepwise-skip")):
        value = getattr(option_state, name, None)
        if value:
            found.add(option)
    return sorted(found)


def inventory_digest(nodeids: list[str], source: dict[str, object],
                     runtime: dict[str, object], shard_count: int) -> str:
    return _sha256(_canonical_json({"nodeids": nodeids, "source": source,
                                    "runtime": runtime, "shard_count": shard_count}))


def summarize_phases(phase_reports: object) -> dict[str, Any]:
    """Validate a pytest lifecycle trace and derive its terminal summary."""
    if not isinstance(phase_reports, list):
        raise ShardError("phase reports must be a list")
    reports: list[dict[str, str]] = []
    for report in phase_reports:
        if (not isinstance(report, dict) or report.get("phase") not in
                {"setup", "call", "teardown"} or report.get("outcome") not in
                {"passed", "failed", "skipped"}):
            raise ShardError("phase report has an unknown phase or outcome")
        if set(report) - {"phase", "outcome", "wasxfail", "skip_reason"}:
            raise ShardError("phase report has unexpected fields")
        if "wasxfail" in report and not isinstance(report["wasxfail"], str):
            raise ShardError("xfail reason must be text")
        if report["outcome"] == "skipped":
            if not isinstance(report.get("skip_reason"), str) or not report["skip_reason"].strip():
                raise ShardError("skipped phase lacks its reason")
        elif "skip_reason" in report:
            raise ShardError("only skipped phases may carry a skip reason")
        reports.append(report)

    phases = [report["phase"] for report in reports]
    valid_sequences = ([], ["setup"], ["setup", "call"], ["setup", "teardown"],
                       ["setup", "call", "teardown"])
    if phases not in valid_sequences:
        raise ShardError("phase reports are duplicated or out of order")
    if reports:
        setup_passed = reports[0]["outcome"] == "passed"
        if "call" in phases and not setup_passed:
            raise ShardError("call report cannot follow skipped or failed setup")
        if phases == ["setup", "teardown"] and setup_passed:
            raise ShardError("passed setup without a call cannot be terminal")
    terminal = bool(phases and phases[-1] == "teardown")
    if any(report["outcome"] == "failed" for report in reports):
        outcome = "failed"
    elif any(report["outcome"] == "skipped" and "wasxfail" in report
             for report in reports):
        outcome = "xfailed"
    elif any(report["outcome"] == "skipped" for report in reports):
        outcome = "skipped"
    elif any(report["phase"] == "call" and report["outcome"] == "passed"
             and "wasxfail" in report for report in reports):
        outcome = "xpassed"
    elif terminal and "call" in phases:
        outcome = "passed"
    else:
        outcome = "running" if reports else "missing"
    return {"terminal": terminal, "terminal_phase": phases[-1] if phases else None,
            "outcome": outcome}


def _module(nodeid: str) -> str:
    module, separator, _ = nodeid.partition("::")
    if not separator or not module:
        raise ShardError(f"invalid pytest node ID: {nodeid!r}")
    return module


def partition_modules(nodeids: list[str], shard_count: int) -> list[dict[str, Any]]:
    """Greedily balance whole test modules; preserve original order within shards."""
    if type(shard_count) is not int or shard_count < 1:
        raise ShardError("shard_count must be a positive integer")
    if len(nodeids) != len(set(nodeids)):
        raise ShardError("full inventory contains duplicate node IDs")
    module_nodes: dict[str, list[str]] = {}
    module_order: list[str] = []
    for nodeid in nodeids:
        module = _module(nodeid)
        if module not in module_nodes:
            module_nodes[module] = []
            module_order.append(module)
        module_nodes[module].append(nodeid)
    if len(module_nodes) < shard_count:
        raise ShardError("every shard must receive at least one complete test module")

    loads = [0] * shard_count
    assigned_modules: list[set[str]] = [set() for _ in range(shard_count)]
    for module in sorted(module_nodes, key=lambda name: (-len(module_nodes[name]), name)):
        shard = min(range(shard_count), key=lambda index: (loads[index], index))
        assigned_modules[shard].add(module)
        loads[shard] += len(module_nodes[module])

    return [
        {
            "shard_index": index,
            "shard_count": shard_count,
            "module_paths": [module for module in module_order
                             if module in assigned_modules[index]],
            "nodeids": [nodeid for nodeid in nodeids
                        if _module(nodeid) in assigned_modules[index]],
            "expected_count": loads[index],
        }
        for index in range(shard_count)
    ]


def reconcile_shards(
    inventory: dict[str, Any],
    assignments: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit a complete set of shard artifacts without trusting their counts."""
    errors: list[str] = []
    if inventory.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported inventory schema")
    if inventory.get("status") != "complete":
        errors.append("full collection did not complete")
    nodeids = inventory.get("nodeids")
    source = inventory.get("source")
    runtime = inventory.get("runtime")
    shard_count = inventory.get("shard_count")
    if (not isinstance(nodeids, list) or not all(isinstance(item, str) for item in nodeids)
            or not isinstance(source, dict) or not isinstance(runtime, dict)
            or type(shard_count) is not int
            or shard_count < 1):
        return {"status": "incomplete", "errors": errors + ["malformed full inventory"]}
    if len(nodeids) != len(set(nodeids)):
        errors.append("full inventory contains duplicate node IDs")
    if (inventory.get("node_count") != len(nodeids)
            or inventory.get("pytest_return_code") != 0
            or inventory.get("session_exit_code") != 0
            or inventory.get("collection_process_return_code") != 0
            or inventory.get("collection_errors") != []
            or inventory.get("collection_skips") != []
            or inventory.get("collection_filter_options") != []
            or inventory.get("deselected_nodeids") != []
            or inventory.get("duplicate_nodeids") is not False):
        errors.append("collection report is not a clean full inventory")
    digest = inventory_digest(nodeids, source, runtime, shard_count)
    if inventory.get("inventory_sha256") != digest:
        errors.append("full inventory digest mismatch")
    try:
        expected = partition_modules(nodeids, shard_count)
    except ShardError as error:
        return {"status": "incomplete", "errors": errors + [str(error)]}

    by_assignment: dict[int, dict[str, Any]] = {}
    for assignment in assignments:
        index = assignment.get("shard_index")
        if type(index) is not int or index in by_assignment:
            errors.append("duplicate or invalid shard assignment index")
            continue
        by_assignment[index] = assignment
    by_result: dict[int, dict[str, Any]] = {}
    for result in results:
        index = result.get("shard_index")
        if type(index) is not int or index in by_result:
            errors.append("duplicate or invalid shard result index")
            continue
        by_result[index] = result
    if any(index < 0 or index >= shard_count for index in by_assignment):
        errors.append("unexpected shard assignment index")
    if any(index < 0 or index >= shard_count for index in by_result):
        errors.append("unexpected shard result index")

    all_assigned: list[str] = []
    outcome_counts: dict[str, int] = {}
    for expected_assignment in expected:
        index = expected_assignment["shard_index"]
        assignment = by_assignment.get(index)
        if assignment is None:
            errors.append(f"missing assignment for shard {index}")
        elif assignment.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"unsupported assignment schema for shard {index}")
        elif (assignment.get("inventory_sha256") != digest
              or assignment.get("source") != source or assignment.get("runtime") != runtime
              or assignment.get("shard_count") != shard_count
              or assignment.get("module_paths") != expected_assignment["module_paths"]
              or assignment.get("nodeids") != expected_assignment["nodeids"]
              or assignment.get("expected_count") != len(expected_assignment["nodeids"])
              or assignment.get("assignment_sha256") != _sha256(
                  _canonical_json(expected_assignment))):
            errors.append(f"assignment mismatch for shard {index}")
        if assignment is not None:
            assigned_ids = assignment.get("nodeids")
            if isinstance(assigned_ids, list) and all(isinstance(item, str) for item in assigned_ids):
                all_assigned.extend(assigned_ids)
            else:
                errors.append(f"invalid assigned node IDs in shard {index}")

        result = by_result.get(index)
        if result is None:
            errors.append(f"missing terminal result for shard {index}")
            continue
        if result.get("inventory_sha256") != digest:
            errors.append(f"inventory digest mismatch in shard {index} result")
        if result.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"unsupported result schema for shard {index}")
        if result.get("source") != source or result.get("runtime") != runtime:
            errors.append(f"source/runtime mismatch in shard {index} result")
        if result.get("source_unchanged") is not True or result.get("runtime_unchanged") is not True:
            errors.append(f"source/runtime changed during shard {index}")
        if assignment is not None and result.get("assignment_sha256") != assignment.get(
                "assignment_sha256"):
            errors.append(f"assignment digest mismatch in shard {index} result")
        if (result.get("status") != "complete" or result.get("terminal") is not True
                or result.get("pytest_return_code") != 0):
            errors.append(f"pytest did not complete successfully in shard {index}")
        if result.get("session_exit_code") != 0:
            errors.append(f"pytest session did not finish successfully in shard {index}")
        if (result.get("selection_matches") is not True or result.get("collection_errors") != []
                or result.get("collection_filter_options") != []
                or result.get("deselected_nodeids") != []
                or result.get("expected_count") != len(expected_assignment["nodeids"])):
            errors.append(f"selection or collection evidence is incomplete in shard {index}")
        if result.get("actual_nodeids") != expected_assignment["nodeids"]:
            errors.append(f"actual collected node IDs differ in shard {index}")
        cases = result.get("cases")
        if not isinstance(cases, list):
            errors.append(f"terminal case outcomes missing in shard {index}")
            continue
        case_ids = [case.get("nodeid") for case in cases if isinstance(case, dict)]
        if case_ids != expected_assignment["nodeids"]:
            errors.append(f"terminal outcomes do not exactly cover shard {index}")
            continue
        for case in cases:
            try:
                derived = summarize_phases(case.get("phase_reports"))
            except ShardError as error:
                errors.append(f"invalid phase report in shard {index}: {error}")
                continue
            if any(case.get(key) != derived[key]
                   for key in ("terminal", "terminal_phase", "outcome")):
                errors.append(f"case summary differs from phase reports in shard {index}")
            if derived["terminal"] is not True:
                errors.append(f"nonterminal case outcome in shard {index}")
            outcome = derived["outcome"]
            if outcome not in {"passed", "skipped", "xfailed", "xpassed"}:
                errors.append(f"non-success outcome in shard {index}: {outcome!r}")
            else:
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    if len(all_assigned) != len(nodeids) or set(all_assigned) != set(nodeids):
        errors.append("shard assignments are not a disjoint full-inventory union")
    return {
        "status": "complete" if not errors else "incomplete",
        "errors": errors,
        "inventory_sha256": digest,
        "shard_count": shard_count,
        "node_count": len(nodeids),
        "terminal_outcome_counts": outcome_counts,
    }


def _source_record() -> dict[str, Any]:
    tracked = (
        ".github/scripts/run_cpu_test_shard.py",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "requirements/ci-py312-linux.lock",
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {"commit": commit,
            "sha256": {name: _file_sha256(ROOT / name) for name in tracked}}


def _runtime_record() -> dict[str, Any]:
    return {"python": platform.python_version(), "torch": version("torch"),
            "pytest": pytest.__version__}


class _CollectionPlugin:
    def __init__(self, output_path: Path, source: dict[str, Any],
                 runtime: dict[str, Any], shard_count: int) -> None:
        self.output_path = output_path
        self.source = source
        self.runtime = runtime
        self.shard_count = shard_count
        self.nodeids: list[str] = []
        self.errors: list[dict[str, str]] = []
        self.skips: list[dict[str, str]] = []
        self.deselected: list[str] = []
        self.filter_options: list[str] = []
        self.session_exit_code: int | None = None

    def pytest_configure(self, config: Any) -> None:
        self.filter_options = _filter_options(config)
        if self.filter_options:
            self.errors.append({
                "nodeid": "<pytest-config>", "outcome": "filtered-collection",
                "reason": ", ".join(self.filter_options),
            })
            _write_json(self.output_path, self._record("failed", []))
            raise pytest.UsageError(
                "full CPU collection refuses filter/ignore options: "
                + ", ".join(self.filter_options)
            )

    def pytest_collectreport(self, report: Any) -> None:
        if report.outcome in {"failed", "skipped"}:
            row = {"nodeid": str(report.nodeid), "outcome": str(report.outcome),
                   "reason": str(report.longrepr)}
            (self.errors if report.failed else self.skips).append(row)

    def pytest_collection_finish(self, session: Any) -> None:
        self.nodeids = [item.nodeid for item in session.items]

    def pytest_deselected(self, items: list[Any]) -> None:
        self.deselected.extend(item.nodeid for item in items)

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        self.session_exit_code = int(exitstatus)

    def _record(self, status: str, nodeids: list[str]) -> dict[str, Any]:
        unique = len(nodeids) == len(set(nodeids))
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "source": self.source,
            "runtime": self.runtime,
            "shard_count": self.shard_count,
            "inventory_sha256": inventory_digest(
                nodeids, self.source, self.runtime, self.shard_count,
            ),
            "node_count": len(nodeids),
            "pytest_return_code": None,
            "session_exit_code": self.session_exit_code,
            "collection_errors": self.errors,
            "collection_skips": self.skips,
            "collection_filter_options": self.filter_options,
            "deselected_nodeids": self.deselected,
            "duplicate_nodeids": not unique,
            "nodeids": nodeids,
        }


def _collect_worker(output_path: Path, shard_count: int) -> int:
    source = _source_record()
    runtime = _runtime_record()
    plugin = _CollectionPlugin(output_path, source, runtime, shard_count)
    _write_json(output_path, {
        "schema_version": SCHEMA_VERSION,
        "status": "collecting",
        "source": source,
        "runtime": runtime,
        "shard_count": shard_count,
        "nodeids": [],
        "collection_errors": [],
        "collection_skips": [],
        "collection_filter_options": [],
        "deselected_nodeids": [],
    })
    exit_code = int(pytest.main(["--collect-only", "-q"], plugins=[plugin]))
    unique = len(plugin.nodeids) == len(set(plugin.nodeids))
    status = "complete" if (exit_code == 0 and plugin.session_exit_code == 0
                            and not plugin.errors and not plugin.skips
                            and not plugin.filter_options and not plugin.deselected
                            and unique and plugin.nodeids) else "failed"
    record = plugin._record(status, plugin.nodeids)
    record["pytest_return_code"] = exit_code
    _write_json(output_path, record)
    return exit_code if status == "complete" else 2


class _RunPlugin:
    def __init__(self, assignment: dict[str, Any], output_path: Path) -> None:
        self.assignment = assignment
        self.output_path = output_path
        self.expected = assignment["nodeids"]
        self.actual: list[str] = []
        self.errors: list[dict[str, str]] = []
        self.reports: dict[str, list[dict[str, str]]] = {}
        self.session_exit_code: int | None = None
        self.selection_matches = False
        self.active_nodeid: str | None = None
        self.filter_options: list[str] = []
        self.deselected: list[str] = []

    def pytest_configure(self, config: Any) -> None:
        self.filter_options = _filter_options(config)

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.errors.append({"nodeid": str(report.nodeid), "outcome": str(report.outcome)})

    def pytest_collection_finish(self, session: Any) -> None:
        self.actual = [item.nodeid for item in session.items]
        self.selection_matches = (self.actual == self.expected and not self.filter_options
                                  and not self.deselected)

    def pytest_deselected(self, items: list[Any]) -> None:
        self.deselected.extend(item.nodeid for item in items)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session: Any) -> bool | None:
        if not self.selection_matches:
            session.exitstatus = 1
            return True  # Fail closed before executing an unexpected selection.
        return None

    def pytest_runtest_logreport(self, report: Any) -> None:
        row = {"phase": str(report.when), "outcome": str(report.outcome)}
        wasxfail = getattr(report, "wasxfail", None)
        if wasxfail is not None:
            row["wasxfail"] = str(wasxfail)
        if report.outcome == "skipped":
            longrepr = report.longrepr
            reason = longrepr[2] if isinstance(longrepr, tuple) and len(longrepr) >= 3 else longrepr
            row["skip_reason"] = str(reason) if reason is not None else ""
        self.reports.setdefault(report.nodeid, []).append(row)
        if report.when == "teardown" or report.outcome in {"failed", "skipped"}:
            if report.when == "teardown":
                self.active_nodeid = None
            self._persist(terminal=False)

    def pytest_runtest_logstart(self, nodeid: str, location: object) -> None:
        del location
        self.active_nodeid = nodeid
        self._persist(terminal=False)

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        self.session_exit_code = int(exitstatus)
        self._persist(terminal=True)

    def cases(self) -> list[dict[str, Any]]:
        result = []
        for nodeid in self.expected:
            phases = self.reports.get(nodeid, [])
            try:
                summary = summarize_phases(phases)
            except ShardError as error:
                summary = {"terminal": False, "terminal_phase": None,
                           "outcome": "invalid", "phase_error": str(error)}
            result.append({"nodeid": nodeid, **summary, "phase_reports": phases})
        return result

    def _record(self, *, terminal: bool, pytest_return_code: int | None = None) -> dict[str, Any]:
        cases = [case for case in self.cases() if case["phase_reports"]]
        all_terminal = len(cases) == len(self.expected) and all(case["terminal"] for case in cases)
        complete = (
            terminal and all_terminal and self.selection_matches and not self.errors
            and self.session_exit_code == 0 and pytest_return_code == 0
            and not self.filter_options and not self.deselected
            and all(case["outcome"] in {"passed", "skipped", "xfailed", "xpassed"}
                    for case in cases)
        )
        if complete:
            status = "complete"
        elif not terminal:
            status = "running"
        elif pytest_return_code is None:
            status = "finished_pending_return_code"
        else:
            status = "failed"
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "terminal": terminal,
            "shard_index": self.assignment["shard_index"],
            "inventory_sha256": self.assignment["inventory_sha256"],
            "assignment_sha256": self.assignment["assignment_sha256"],
            "source": self.assignment["source"],
            "runtime": self.assignment["runtime"],
            "expected_count": len(self.expected),
            "pytest_return_code": pytest_return_code,
            "session_exit_code": self.session_exit_code,
            "selection_matches": self.selection_matches,
            "collection_errors": self.errors,
            "collection_filter_options": self.filter_options,
            "deselected_nodeids": self.deselected,
            "actual_nodeids": self.actual,
            "active_nodeid": getattr(self, "active_nodeid", None),
            "cases": cases,
        }

    def _persist(self, *, terminal: bool, pytest_return_code: int | None = None) -> None:
        _write_json(self.output_path, self._record(
            terminal=terminal, pytest_return_code=pytest_return_code,
        ))


def _run_worker(assignment_path: Path, output_path: Path) -> int:
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    plugin = _RunPlugin(assignment, output_path)
    plugin._persist(terminal=False)
    if (_source_record() != assignment.get("source")
            or _runtime_record() != assignment.get("runtime")):
        record = plugin._record(terminal=False)
        record["status"] = "worker_error"
        record["worker_error"] = "source or runtime differs from the collected inventory"
        _write_json(output_path, record)
        return 2
    exit_code = int(pytest.main(["-vv", "--color=no", *assignment["module_paths"]],
                                plugins=[plugin]))
    record = plugin._record(terminal=plugin.session_exit_code is not None,
                            pytest_return_code=exit_code)
    _write_json(output_path, record)
    return exit_code if record["status"] == "complete" else 1


def _launch(mode: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()),
                           mode, *args], cwd=ROOT, text=True, check=False,
                          env=_sanitized_pytest_env())


def run_shard(shard_index: int, shard_count: int, record_dir: Path) -> int:
    if (type(shard_index) is not int or type(shard_count) is not int
            or not 0 <= shard_index < shard_count):
        raise ShardError("shard index must be in [0, shard_count)")
    record_dir = record_dir.resolve()
    record_dir.mkdir(parents=True, exist_ok=True)
    if any(record_dir.iterdir()):
        raise ShardError(f"record directory must be empty: {record_dir}")

    inventory_path = record_dir / "inventory.json"
    collection_process = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--collect-worker",
         str(inventory_path), "--shard-count", str(shard_count)],
        cwd=ROOT, text=True, check=False, capture_output=True,
        env=_sanitized_pytest_env(),
    )
    (record_dir / "collection.stdout.log").write_text(collection_process.stdout,
                                                       encoding="utf-8")
    (record_dir / "collection.stderr.log").write_text(collection_process.stderr,
                                                       encoding="utf-8")
    if not inventory_path.is_file():
        raise ShardError("isolated collection worker produced no inventory record")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["collection_process_return_code"] = collection_process.returncode
    nodeids = inventory.get("nodeids")
    source = inventory.get("source")
    runtime = inventory.get("runtime")
    metadata_valid = (isinstance(nodeids, list) and all(isinstance(item, str) for item in nodeids)
                      and isinstance(source, dict) and isinstance(runtime, dict))
    collection_complete = (metadata_valid and collection_process.returncode == 0
                           and inventory.get("status") == "complete")
    if not metadata_valid:
        inventory["status"] = "failed"
        inventory["collection_error"] = "inventory lacks node IDs or source/runtime metadata"
        _write_json(inventory_path, inventory)
        return 2
    if not collection_complete and inventory.get("status") == "collecting":
        inventory["status"] = "failed"
        inventory["collection_error"] = "collection subprocess ended before terminal report"
    inventory_sha = inventory.get("inventory_sha256")
    if not isinstance(inventory_sha, str) or inventory_sha != inventory_digest(
            nodeids, source, runtime, shard_count):
        inventory["status"] = "failed"
        inventory["collection_error"] = "inventory digest mismatch"
        _write_json(inventory_path, inventory)
        return 2
    try:
        partitions = partition_modules(nodeids, shard_count) if collection_complete else []
    except ShardError as error:
        inventory["status"] = "failed"
        inventory["partition_error"] = str(error)
        collection_complete = False
        partitions = []
    _write_json(inventory_path, inventory)
    if not collection_complete:
        return 2

    assignment = {
        **partitions[shard_index],
        "schema_version": SCHEMA_VERSION,
        "inventory_sha256": inventory_sha,
        "source": source,
        "runtime": runtime,
        "assignment_sha256": _sha256(_canonical_json(partitions[shard_index])),
    }
    assigned_nodeids = assignment["nodeids"]
    if not isinstance(assigned_nodeids, list) or not all(
            isinstance(item, str) for item in assigned_nodeids):
        raise ShardError("generated shard contains invalid node IDs")
    assignment_path = record_dir / f"shard-{shard_index:02d}-assignment.json"
    result_path = record_dir / f"shard-{shard_index:02d}-result.json"
    _write_json(assignment_path, assignment)
    process = _launch("--run-worker", str(assignment_path), str(result_path))
    if not result_path.is_file():
        return 2
    result = json.loads(result_path.read_text(encoding="utf-8"))
    end_source = _source_record()
    end_runtime = _runtime_record()
    result["source_unchanged"] = end_source == source
    result["runtime_unchanged"] = end_runtime == runtime
    if not result["source_unchanged"] or not result["runtime_unchanged"]:
        result["status"] = "failed"
    _write_json(result_path, result)
    if result.get("status") != "complete" or process.returncode != 0:
        return 1
    print(f"shard {shard_index + 1}/{shard_count}: {len(assigned_nodeids)} cases complete")
    return 0


def main(argv: list[str] | None = None) -> int:
    os.environ.pop("PYTEST_ADDOPTS", None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--collect-worker", type=Path, metavar="OUTPUT", help=argparse.SUPPRESS)
    parser.add_argument("--run-worker", nargs=2, metavar=("ASSIGNMENT", "OUTPUT"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.collect_worker is not None:
        return _collect_worker(args.collect_worker, args.shard_count)
    if args.run_worker is not None:
        return _run_worker(Path(args.run_worker[0]), Path(args.run_worker[1]))
    if args.shard_index is None or args.record_dir is None:
        parser.error("--shard-index and --record-dir are required")
    try:
        return run_shard(args.shard_index, args.shard_count, args.record_dir)
    except (OSError, ShardError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"CPU test shard failed closed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
