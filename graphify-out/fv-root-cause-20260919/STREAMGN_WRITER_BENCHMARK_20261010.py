"""Compare isolated checkpoint writers on synthetic JSON, never FV or AD."""
from pathlib import Path
from typing import Any
import ast
import hashlib
import json
import os
import tempfile
import time
import tracemalloc

E = Path(__file__).resolve().parent
ROOT = E.parents[1]


def writer(path: Path):
    node = next(node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_write")
    namespace = {"Path": Path, "Any": Any, "json": json, "os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_write"]


def main() -> None:
    writers = {
        "legacy": writer(E / "streamgn_source_20261010/fv_point_3h_tangent_continuation.py"),
        "stream": writer(ROOT / "examples/weather_scenarios/fv_point_3h_tangent_continuation.py"),
    }
    value = {"한국어": "바이트 보존", "negative_zero": -0.0,
        "records": [{"index": i, "gradient": [.12, -.25, .5], "branch": [True, False, True],
            "label": "current point"} for i in range(24000)]}
    statistics = {}
    with tempfile.TemporaryDirectory(prefix="advar-stream-write-only-") as directory:
        for name, write in writers.items():
            output = Path(directory) / f"{name}.json"
            tracemalloc.start()
            start = time.monotonic()
            write(output, value)
            elapsed = time.monotonic() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            data = output.read_bytes()
            statistics[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "write_seconds": elapsed, "python_traced_peak_bytes": peak}
    same = statistics["legacy"]["sha256"] == statistics["stream"]["sha256"]
    assert same
    result = {"scope": "Synthetic JSON; functions/input prepared before tracing; write-interval Python allocations only, not FV/Tensor/process RSS or general performance",
        "same_bytes": same, "measurements": statistics}
    (E / "STREAMGN_WRITER_BENCHMARK_20261010.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
