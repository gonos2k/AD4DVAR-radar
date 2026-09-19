"""Bounded CLI probe for the existing exploratory P1 learning path."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_learning import run_mean_only_learning

ROOT = Path(__file__).resolve().parents[2]
SOURCES = tuple(ROOT / path for path in (
    "examples/weather_scenarios/prior_learning.py", "src/advar/nowcast.py",
    "src/advar/physics.py", "src/advar/variational.py", "src/advar/sensitivity.py",
)) + (Path(__file__).resolve(),)

def _rss() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)

def _destination(path: Path) -> Path:
    index = 0
    while True:
        candidate = path if index == 0 else path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-destination", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    report = {"status": "failed", "scope": "existing global-motion P1 mean-only one-parameter one-step learning; new FV not connected", "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCES}, "results": {}}
    try:
        with tempfile.TemporaryDirectory(prefix="advar-existing-p1-") as temp:
            checkpoint = Path(temp) / "mean_only_prior.pt"
            result = run_mean_only_learning(checkpoint_path=checkpoint)
            report["results"] = asdict(result)
            destination = _destination(args.checkpoint_destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(checkpoint.read_bytes())
            report["checkpoint_path"] = str(destination)
            report["status"] = "passed"
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    report["wall_seconds"] = time.perf_counter() - started
    report["peak_rss_bytes"] = _rss()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0 if report["status"] == "passed" else 1

if __name__ == "__main__":
    raise SystemExit(main())
