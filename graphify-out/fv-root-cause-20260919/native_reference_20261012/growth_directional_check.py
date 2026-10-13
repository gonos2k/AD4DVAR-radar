"""Actual fixed-control growth ray: finite costs, branch changes and derivatives."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
OUT = HERE / "growth_ray_20261013"
INPUT = HERE / "two_face_intersection_20261013/continuation2_20261013/native24_chord_extension_result.json"
spec = importlib.util.spec_from_file_location("growth_qualifier", HERE / "qualify_candidate.py")
q = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = q
spec.loader.exec_module(q)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tensor_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()

def run():
    OUT.mkdir(exist_ok=True)
    if (OUT / "RESULT.json").exists():
        raise ValueError("growth-ray receipt already exists")
    raw = json.loads(INPUT.read_text())
    control = torch.tensor(raw["current_control"], dtype=torch.float64)
    if tensor_sha(control) != raw["current_control_sha256"]:
        raise ValueError("terminal control hash mismatch")
    case = q.cycle.load_fixed_case()
    problem, parameters = case["problem"], case["parameters"]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in (
        ROOT / "src/advar/transport.py", ROOT / "src/advar/variational.py",
        ROOT / "src/advar/matrix_free.py", Path(__file__), HERE / "qualify_candidate.py")}
    plan = dict(scope="fixed flow/field controls; native J and growth partial only",
        input_sha256=sha(INPUT), control_sha256=tensor_sha(control),
        parameters_sha256=tensor_sha(parameters), case_sha256=sha(q.cycle.CASE),
        source_hashes=sources, perturbations=[-2e-7, -2e-8, 0.0, 2e-8, 2e-7],
        native_J_cap=5, growth_gradient_cap=4, seconds_cap=30,
        cumulative_native_J_before=493, cumulative_gradient_before=124,
        original_remaining_seconds_estimate=186.379,
        no_optimizer=True, no_score=True, no_minimum_claim=True)
    (OUT / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    start = time.monotonic()
    rows = []
    for delta in plan["perturbations"]:
        if time.monotonic() - start >= plan["seconds_cap"]:
            raise TimeoutError("growth ray time cap reached")
        point = control.clone()
        point[25] += delta
        branch = q._analysis_branch(problem, point, parameters,
                                    deadline=start + plan["seconds_cap"])
        value = branch["native_J_on_audited_path"]
        row = dict(delta=delta, native_J=value, branch=branch,
                   control_sha256=tensor_sha(point), growth_partial=None)
        if delta:
            gradient = torch.func.grad(problem.objective, argnums=0)(point, parameters)
            row["growth_partial"] = float(gradient[25])
        rows.append(row)
        (OUT / "PREFIX.json").write_text(json.dumps(rows, indent=2) + "\n")
    baseline = rows[2]["native_J"]
    for row in rows:
        row["native_J_change"] = row["native_J"] - baseline
        row["secant_slope"] = None if not row["delta"] else row["native_J_change"] / row["delta"]
    result = dict(scope=plan["scope"], status="completed", rows=rows,
        native_J_calls=5, gradient_calls=4, cumulative_native_J_after=498,
        cumulative_gradient_after=128, Hessian_rows=0, score_calls=0,
        source_unchanged=all(sha(ROOT / name) == digest for name, digest in sources.items()),
        input_unchanged=sha(INPUT) == plan["input_sha256"],
        parameters_unchanged=tensor_sha(parameters) == plan["parameters_sha256"],
        baseline_stored_J=raw["native_J"], baseline_J=baseline,
        elapsed_seconds=time.monotonic() - start, minimum_claim=False,
        interpretation="Growth partial is along this fixed-flow ray, not a full Clarke stationarity certificate.")
    (OUT / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
        "rows": [{k: row[k] for k in ("delta", "native_J_change", "growth_partial", "secant_slope")} for row in rows]}))

if __name__ == "__main__":
    run()
