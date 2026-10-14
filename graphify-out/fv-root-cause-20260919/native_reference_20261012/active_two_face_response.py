"""Gated actual parameter response on the qualified y[4,3]∩y[2,1] chart."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
DIAGNOSTIC_SOURCE = HERE / "two_face_intersection_20261013" / "diagnose.py"
COMMON_SOURCE = HERE / "perturbation_response.py"
BASE_NATIVE_PLAN_SHA = "b6f3a08e02ec6da684ea8ea7c9611e6eedf3b8de9de8a9ebc6830347dd13d9ac"
SUPPORT_BEFORE_SHA = "c9dde24c12237d5ffee6b8ca0766b4d5619abf6110f3e858ee447bf49c1f7da9"
SUPPORT_AFTER_SHA = "f6c680da7ca130fcbc505a03b73741591dc023f57a7745e47cfd49dd72622fab"
TRANSPORT_PATH = "src/advar/transport.py"
FIXED_PROBLEM_SHA = "16fa95b534c3b61cf5d1506770cbfc77edb527c40d1f4c536f5130cba24e1fc3"
FACES = (("y", 4, 3), ("y", 2, 1))
FACE_LABELS = ("y:4:3", "y:2:1")
SECTORS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
PARAMETERS = ((0, "observation_dbz_0"), (12, "background_beta"))
CONTROL_SIZE = 26
TANGENT_SIZE = 24
STATIONARITY_TOL = 1e-10
ROUNDING_MULTIPLIER = 128.0
TOTAL_WALL_CAP_SECONDS = 3600.0

_diagnostic_spec = importlib.util.spec_from_file_location("active_two_face_geometry", DIAGNOSTIC_SOURCE)
if _diagnostic_spec is None or _diagnostic_spec.loader is None:
    raise RuntimeError("pinned two-face geometry helpers cannot be loaded")
diagnose = importlib.util.module_from_spec(_diagnostic_spec)
sys.modules[_diagnostic_spec.name] = diagnose
_diagnostic_spec.loader.exec_module(diagnose)
_RED_QUALIFIER: Any | None = None


def _load_red_qualifier() -> Any:
    global _RED_QUALIFIER
    if _RED_QUALIFIER is not None:
        return _RED_QUALIFIER
    path = HERE / "qualify_candidate.py"
    spec = importlib.util.spec_from_file_location("active_two_face_native_qualifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("two-face RED branch auditor cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _RED_QUALIFIER = module
    return module


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    if value.dtype != torch.float64 or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
        raise ValueError("two-face response tensors must be finite CPU FP64")
    return _sha_bytes(value.detach().contiguous().numpy().tobytes())


def _weighted_tensor_sum(weights: list[float] | Tensor, values: list[Tensor]) -> Tensor:
    if not values:
        raise ValueError("weighted tensor sum requires at least one tensor")
    result = torch.zeros_like(values[0])
    for weight, value in zip(weights, values):
        result = result + weight * value
    return result


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_twoface_qualification_plan(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.is_symlink() or _sha_bytes(path.read_bytes()) != expected_sha:
        raise ValueError("two-face A qualification plan seal mismatch")
    sidecar = path.with_name(f"{path.stem}_SHA256.txt")
    if not sidecar.is_file() or sidecar.read_text().strip() != expected_sha:
        raise ValueError("two-face A qualification plan sidecar mismatch")
    plan = _read(path)
    if plan.get("status") != "frozen_native24_continuation":
        raise ValueError("two-face A qualification plan is not frozen")
    transition = plan.get("transport_extension_transition")
    if (not isinstance(transition, dict) or transition.get("path") != TRANSPORT_PATH
            or transition.get("before") != SUPPORT_BEFORE_SHA
            or transition.get("after") != SUPPORT_AFTER_SHA):
        raise ValueError("two-face A qualification plan lacks the c9dde24→f6c680da source transition")
    pins = plan.get("source_files")
    archives = plan.get("artifact_pins")
    if not isinstance(pins, dict) or not isinstance(archives, dict):
        raise ValueError("two-face A qualification plan source/archive pins are malformed")
    if pins.get(TRANSPORT_PATH) != SUPPORT_AFTER_SHA:
        raise ValueError("two-face A source pins do not include current transport")
    for name, digest in {**pins, **archives}.items():
        file_path = ROOT / name
        if (file_path.is_symlink() or not file_path.is_file()
                or not file_path.resolve().is_relative_to(ROOT.resolve())
                or _sha_bytes(file_path.read_bytes()) != digest):
            raise ValueError(f"two-face A qualification source/archive pin mismatch: {name}")
    snapshot = plan.get("launch_snapshot_sha256")
    snapshot_relative = plan.get("launch_snapshot_path")
    if not isinstance(snapshot, str) or not isinstance(snapshot_relative, str):
        raise ValueError("two-face A qualification plan omits its retained launch snapshot")
    snapshot_path = ROOT / snapshot_relative
    if snapshot_path.is_symlink() or not snapshot_path.is_file() or _sha_bytes(snapshot_path.read_bytes()) != snapshot:
        raise ValueError("two-face A launch snapshot digest mismatch")
    return plan


def _write(output: Path, record: dict[str, Any]) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


def _gate(output: Path, status: str, reason: str, *, control_sha: str | None = None,
        plan_sha: str | None = None) -> dict[str, Any]:
    return _write(output, {"status": status, "reason": reason,
        "base_control_sha256": control_sha, "phase_plan_sha256": plan_sha,
        "minimum_claim": False, "response_claim": False, "selection_use": False,
        "native_J_evaluations_started": 0, "score_evaluations_started": 0})


def _qualified_base(record: Any, parameters: Tensor) -> tuple[Tensor, list[float], str, str]:
    if not isinstance(record, dict):
        raise ValueError("two-face RED receipt must be a JSON object")
    if (record.get("status") != "qualified_original_native_J_two_face_minimum"
            or record.get("minimum_claim") is not True):
        raise ValueError("RED has not qualified the actual two-face original-J minimum")
    point = record.get("final_point")
    if not isinstance(point, dict):
        raise ValueError("two-face qualification is missing final_point")
    control = torch.as_tensor(point.get("control"), dtype=torch.float64, device="cpu")
    if (control.shape != (CONTROL_SIZE,) or _tensor_sha(control) != point.get("control_sha256")
            or not bool(torch.isfinite(control).all())):
        raise ValueError("qualified two-face control/hash is malformed")
    if record.get("input_parameters_sha256") != _tensor_sha(parameters):
        raise ValueError("two-face qualification parameter digest differs from the fixed case")
    transition = record.get("source_transition", record.get("transport_extension_transition"))
    if (not isinstance(transition, dict) or transition.get("path") != TRANSPORT_PATH
            or transition.get("before") != SUPPORT_BEFORE_SHA
            or transition.get("after") != SUPPORT_AFTER_SHA):
        raise ValueError("two-face qualification is missing the current tuple-extension source transition")
    branch = point.get("branch")
    if (not isinstance(branch, dict) or branch.get("classification") != "multiplefaces"
            or branch.get("stage_count") != 360
            or branch.get("unique_nearzero_face_entities") != ["y:2:1", "y:4:3"]
            or branch.get("unique_near_tie_slope_entities") != []):
        raise ValueError("base point does not have exactly the two certified 360-stage faces")
    if (point.get("branch_supported_exactly_two_faces") is not True
            or point.get("first_order_pass") is not True):
        raise ValueError("base point lacks the final two-face first-order qualification")
    hull = point.get("full_clarke_hull")
    if not isinstance(hull, dict) or not isinstance(hull.get("weights"), list):
        raise ValueError("base point is missing reference four-sector mixture weights")
    weights = [float(value) for value in hull["weights"]]
    if (len(weights) != 4 or any(not math.isfinite(value) or value < 0 for value in weights)
            or abs(sum(weights) - 1.0) > 256 * torch.finfo(torch.float64).eps):
        raise ValueError("base four-sector reference weights must be nonnegative and sum to one")
    if float(torch.as_tensor(hull["vector"], dtype=torch.float64).abs().max()) >= STATIONARITY_TOL:
        raise ValueError("base full Clarke-hull gradient is not stationary")
    if float(torch.as_tensor(point["tangent_hull"]["vector"], dtype=torch.float64).abs().max()) >= STATIONARITY_TOL:
        raise ValueError("base 24-coordinate tangent hull is not stationary")
    normal_coefficients = point.get("normal_coefficients")
    if (not isinstance(normal_coefficients, list) or len(normal_coefficients) != 4
            or any(not isinstance(row, dict) or row.get("sides") != list(sides)
                or row.get("strict_sign_match") != [True, True]
                for row, sides in zip(normal_coefficients, SECTORS))):
        raise ValueError("base two-face sector normal coefficients lack strict signed margins")
    branch_sha = branch.get("actual_selector_signature_sha256")
    plan_sha = record.get("plan_sha256")
    if not isinstance(branch_sha, str) or not isinstance(plan_sha, str):
        raise ValueError("two-face qualification is missing branch or qualification-plan digest")
    return control, weights, branch_sha, plan_sha


def _two_face_base_hessians(record: dict[str, Any], weights: list[float]) -> tuple[list[Tensor], Tensor]:
    point = record["final_point"]
    records = record.get("final_hessians")
    if not isinstance(records, dict):
        raise ValueError("P2-qualified two-face receipt has no exact sector Hessian records")
    matrices: list[Tensor] = []
    mixed = torch.zeros((24, 24), dtype=torch.float64)
    for sides, weight in zip(SECTORS, weights):
        key = f"{sides[0]:+d},{sides[1]:+d}"
        row = records.get(key)
        if not isinstance(row, dict) or row.get("positive_definite") is not True:
            raise ValueError(f"qualified two-face receipt has no positive Hessian for sector {key}")
        if row.get("control_sha256") != point.get("control_sha256"):
            raise ValueError(f"qualified two-face sector {key} Hessian is not from the final control")
        hessian = torch.as_tensor(row.get("hessian"), dtype=torch.float64, device="cpu")
        if (hessian.shape != (TANGENT_SIZE, TANGENT_SIZE)
                or _tensor_sha(hessian) != row.get("hessian_sha256")):
            raise ValueError(f"two-face sector {key} Hessian bytes/hash mismatch")
        if not bool(torch.isfinite(hessian).all()) or not bool(torch.allclose(
                hessian, hessian.T, rtol=1e-12, atol=1e-14)):
            raise ValueError(f"two-face sector {key} Hessian is not finite symmetric FP64")
        if int(torch.linalg.cholesky_ex(hessian).info) != 0:
            raise ValueError(f"two-face sector {key} Hessian is not positive definite")
        matrices.append(hessian)
        mixed = mixed + weight * hessian
    mixed = 0.5 * (mixed + mixed.T)
    if not torch.isfinite(mixed).all():
        raise ValueError("reference-weighted two-face Hessian is nonfinite")
    return matrices, mixed


def _qualified_twoface_base(record: Any, parameters: Tensor) -> tuple[Tensor, list[float],
        str, str, dict[str, Any]]:
    control, weights, signature, plan_sha = _qualified_base(record, parameters)
    point = record["final_point"]
    matrix_records, _unused_hessian = _two_face_base_hessians(record, weights)
    del _unused_hessian
    return control, weights, signature, plan_sha, point


def _phase_plan(path: Path, expected_sha: str, qualification: dict[str, Any],
        control: Tensor, parameters: Tensor, chart_metadata: dict[str, Any],
        qualification_plan_sha: str) -> dict[str, Any]:
    if path.is_symlink() or _sha_bytes(path.read_bytes()) != expected_sha:
        raise ValueError("two-face active-response plan seal mismatch")
    plan = _read(path)
    qsha = _sha_bytes(json.dumps(qualification, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode())
    required = {
        "phase_kind": "active_face_reoptimized_parameter_response",
        "geometry_mode": "two_face_intersection_24d",
        "status": "frozen_active_face_response_plan",
        "root_launch_clearance": True,
        "faces": [list(face) for face in FACES],
        "base_control_sha256": _tensor_sha(control),
        "base_parameters_sha256": _tensor_sha(parameters),
        "qualification_sha256": qsha,
        "qualification_plan_sha256": qualification_plan_sha,
        "chart": chart_metadata,
        "runner_sha256": _sha_bytes(SELF.read_bytes()),
        "dispatcher_sha256": _sha_bytes((HERE / "active_face_response.py").read_bytes()),
        "geometry_helper_sha256": _sha_bytes(DIAGNOSTIC_SOURCE.read_bytes()),
        "common_runner_sha256": _sha_bytes((ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/perturbation_response.py").read_bytes()),
    }
    for key, expected in required.items():
        if plan.get(key) != expected:
            raise ValueError(f"two-face active-response plan mismatch: {key}")
    transition = plan.get("source_transition")
    if (not isinstance(transition, dict) or transition.get("path") != TRANSPORT_PATH
            or transition.get("before") != SUPPORT_BEFORE_SHA
            or transition.get("after") != SUPPORT_AFTER_SHA):
        raise ValueError("two-face phase must pin the c9dde24 to f6c680da transport transition")
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if not isinstance(sources, dict) or not isinstance(archives, dict):
        raise ValueError("two-face phase plan requires explicit source_files/archive_files pins")
    if sources.get(TRANSPORT_PATH) != SUPPORT_AFTER_SHA:
        raise ValueError("two-face phase source map lacks the current tuple-extension transport")
    for name, expected in {**sources, **archives}.items():
        item = ROOT / name
        if (item.is_symlink() or not item.is_file()
                or not item.resolve().is_relative_to(ROOT.resolve())
                or _sha_bytes(item.read_bytes()) != expected):
            raise ValueError(f"two-face phase source/archive pin mismatch: {name}")
    budget = plan.get("budget")
    if not isinstance(budget, dict):
        raise ValueError("two-face phase plan is missing a cumulative budget")
    charged = float(budget.get("charged_seconds_before", -1.0))
    phase_seconds = float(budget.get("phase_seconds", -1.0))
    native_before = int(budget.get("native_J_successes_before", -1))
    phase_limit = int(budget.get("phase_native_J_limit", -1))
    if (not math.isfinite(charged) or charged < 0 or not math.isfinite(phase_seconds)
            or phase_seconds <= 0 or charged + phase_seconds > TOTAL_WALL_CAP_SECONDS
            or native_before < 0 or phase_limit < 1
            or type(plan.get("armijo_iterations")) is not int or plan["armijo_iterations"] < 1
            or type(plan.get("armijo_backtracks")) is not int or plan["armijo_backtracks"] < 1):
        raise ValueError("two-face cumulative 3,600-second phase budget is invalid")
    if plan.get("fixed_problem_sha256") != FIXED_PROBLEM_SHA:
        raise ValueError("two-face response fixed-problem hash mismatch")
    if plan.get("runtime_identity") != {"python": sys.version.split()[0],
            "torch": torch.__version__, "device": "CPU FP64"}:
        raise ValueError("two-face phase runtime differs from its freeze")
    return plan


def _twoface_phase_plan(path: Path, expected_sha: str, qualification: dict[str, Any],
        control: Tensor, parameters: Tensor, chart_metadata: dict[str, Any],
        qualification_plan_sha: str) -> dict[str, Any]:
    if (path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve())
            or _sha_bytes(path.read_bytes()) != expected_sha):
        raise ValueError("two-face active-response phase plan seal mismatch")
    plan = _read(path)
    qsha = _sha_bytes(json.dumps(qualification, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode())
    required = {
        "phase_kind": "active_face_reoptimized_parameter_response",
        "geometry_mode": "two_face_intersection_24d",
        "status": "frozen_active_face_response_plan",
        "root_launch_clearance": True,
        "faces": [list(face) for face in FACES],
        "base_control_sha256": _tensor_sha(control),
        "base_parameters_sha256": _tensor_sha(parameters),
        "qualification_sha256": qsha,
        "qualification_plan_sha256": qualification_plan_sha,
        "chart": chart_metadata,
        "runner_sha256": _sha_bytes(SELF.read_bytes()),
        "geometry_helper_sha256": _sha_bytes(DIAGNOSTIC_SOURCE.read_bytes()),
        "common_runner_sha256": _sha_bytes(COMMON_SOURCE.read_bytes()),
    }
    for key, expected in required.items():
        if plan.get(key) != expected:
            raise ValueError(f"two-face phase plan mismatch: {key}")
    if plan.get("fixed_problem_sha256") != FIXED_PROBLEM_SHA:
        raise ValueError("two-face phase fixed-problem identity mismatch")
    transition = plan.get("source_transition")
    if (not isinstance(transition, dict) or transition.get("path") != TRANSPORT_PATH
            or transition.get("before") != SUPPORT_BEFORE_SHA
            or transition.get("after") != SUPPORT_AFTER_SHA):
        raise ValueError("two-face phase must pin c9dde24→f6c680da tuple-selection source transition")
    source_files, archive_files = plan.get("source_files"), plan.get("archive_files")
    if not isinstance(source_files, dict) or not isinstance(archive_files, dict):
        raise ValueError("two-face phase requires explicit source/archive hash maps")
    if source_files.get(TRANSPORT_PATH) != SUPPORT_AFTER_SHA:
        raise ValueError("two-face phase source map omits current transport API")
    source_plan_relative = plan.get("qualification_source_plan_path")
    if not isinstance(source_plan_relative, str):
        raise ValueError("two-face phase plan must identify the final A source plan")
    source_plan_path = ROOT / source_plan_relative
    if (source_plan_path.is_symlink() or not source_plan_path.is_file()
            or _sha_bytes(source_plan_path.read_bytes()) != qualification_plan_sha):
        raise ValueError("two-face qualification source-plan path/hash mismatch")
    _validate_twoface_qualification_plan(source_plan_path, qualification_plan_sha)
    if (source_files.get(source_plan_relative) != qualification_plan_sha
            and archive_files.get(source_plan_relative) != qualification_plan_sha):
        raise ValueError("two-face phase hash map omits the exact A qualification source plan")
    for name, expected in {**source_files, **archive_files}.items():
        file_path = ROOT / name
        if (file_path.is_symlink() or not file_path.is_file()
                or not file_path.resolve().is_relative_to(ROOT.resolve())
                or _sha_bytes(file_path.read_bytes()) != expected):
            raise ValueError(f"two-face active phase pin mismatch: {name}")
    budget = plan.get("budget")
    if not isinstance(budget, dict):
        raise ValueError("two-face active phase is missing cumulative budget")
    charged = float(budget.get("charged_seconds_before", -1.0))
    phase_seconds = float(budget.get("phase_seconds", -1.0))
    count_before = int(budget.get("native_J_successes_before", -1))
    phase_limit = int(budget.get("phase_native_J_limit", -1))
    if (not math.isfinite(charged) or charged < 0 or not math.isfinite(phase_seconds)
            or phase_seconds <= 0 or charged + phase_seconds > TOTAL_WALL_CAP_SECONDS
            or count_before < 0 or phase_limit < 1
            or type(plan.get("armijo_iterations")) is not int or plan["armijo_iterations"] < 1
            or type(plan.get("armijo_backtracks")) is not int or plan["armijo_backtracks"] < 1):
        raise ValueError("two-face active phase cumulative 3,600-second budget is invalid")
    runtime = {"python": sys.version.split()[0], "torch": torch.__version__, "device": "CPU FP64"}
    if plan.get("runtime_identity") != runtime:
        raise ValueError("two-face phase runtime differs from its frozen plan")
    return plan


def _load_two_face_geometry() -> Any:
    return diagnose


def _two_face_chart(control: Tensor, problem: Any) -> tuple[Any, Tensor, dict[str, Any]]:
    diagnostic = _load_two_face_geometry()
    spec = problem.frozen.fv_transport
    if spec is None:
        raise ValueError("two-face chart requires pinned FV transport")
    matrix, pivots, free, metadata = diagnostic._chart(spec.coefficient_limits, spec.psi_basis)
    lift, initial, projected, _original_flow = diagnostic._make_intersection_chart(
        control, matrix, pivots, free)
    if _tensor_sha(projected) != _tensor_sha(control):
        raise ValueError("qualified two-face control does not exactly restore through Γ₂₄")
    chart_matrix = torch.func.jacrev(lift)(initial)
    if chart_matrix.shape != (26, 24) or not bool(torch.isfinite(chart_matrix).all()):
        raise ValueError("two-face Γ₂₄ Jacobian is malformed")
    return lift, initial, metadata






def _two_face_flux_vector(control: Tensor, problem: Any) -> Tensor:
    diagnostic = _load_two_face_geometry()
    spec = problem.frozen.fv_transport
    if spec is None:
        raise ValueError("two-face flux check requires pinned FV transport")
    return torch.stack([diagnostic._face_value(control, spec.psi_basis,
        spec.coefficient_limits, face) for face in FACES])


def _sector_normal_coefficients(normal_matrix: Tensor,
        gradients: list[Tensor]) -> list[dict[str, Any]]:
    """Resolve oblique two-normal cone coefficients without unit-normal shortcuts."""
    if (normal_matrix.ndim != 2 or normal_matrix.shape[0] != 2 or not gradients
            or any(gradient.shape != (normal_matrix.shape[1],) for gradient in gradients)):
        raise ValueError("two-face normal cone requires N[2,n] and matching gradients")
    singular = torch.linalg.svdvals(normal_matrix)
    if (not bool(torch.isfinite(singular).all())
            or float(singular[-1]) <= ROUNDING_MULTIPLIER * torch.finfo(normal_matrix.dtype).eps
                * float(singular[0])):
        raise ValueError("two-face normals are unresolved or dependent")
    gram = normal_matrix @ normal_matrix.T
    row_norms = torch.linalg.vector_norm(normal_matrix, dim=1).clamp_min(
        torch.finfo(normal_matrix.dtype).tiny)
    result = []
    for sides, gradient in zip(SECTORS, gradients):
        eta = torch.linalg.solve(gram, normal_matrix @ gradient)
        margin = torch.tensor(sides, dtype=gradient.dtype) * eta
        budget = ROUNDING_MULTIPLIER * torch.finfo(gradient.dtype).eps * gradient.norm() / row_norms
        result.append({"sides": list(sides), "eta": eta.tolist(),
            "eta_signed": margin.tolist(), "eta_roundoff_budget": budget.tolist(),
            "strict_sign_match": (margin > budget).tolist()})
    return result


def _sector_extension(problem: Any, sector: tuple[int, int]):
    from advar import transport
    selected = tuple((face[0], face[1], face[2], side) for face, side in zip(FACES, sector))
    return transport.selected_face_extensions(*selected)


def _sector_matrices(record: dict[str, Any], weights: list[float]) -> tuple[list[Tensor], Tensor]:
    rows = record.get("final_hessians")
    if not isinstance(rows, dict):
        raise ValueError("qualified two-face base is missing four exact 24x24 sector Hessians")
    hessians: list[Tensor] = []
    keys = ("-1,-1", "-1,+1", "+1,-1", "+1,+1")
    for key in keys:
        entry = rows.get(key)
        if not isinstance(entry, dict) or entry.get("positive_definite") is not True:
            raise ValueError(f"two-face base sector {key} lacks a positive exact Hessian")
        raw = entry.get("hessian")
        if not isinstance(raw, list):
            raise ValueError(f"two-face base sector {key} lacks its matrix")
        matrix = torch.as_tensor(raw, dtype=torch.float64, device="cpu")
        if matrix.shape != (24, 24) or _tensor_sha(matrix) != entry.get("hessian_sha256"):
            raise ValueError(f"two-face sector {key} Hessian shape/digest mismatch")
        hessians.append(matrix)
    reference = torch.tensor(weights, dtype=torch.float64)
    mixed = _weighted_tensor_sum(reference, hessians)
    mixed = 0.5 * (mixed + mixed.T)
    return hessians, mixed


def _mixed_gradient(problem: Any, lift: Any, tangent: Tensor, parameters: Tensor,
        weights: Tensor, *, count_objective: Callable[[], None] | None = None) -> Tensor:
    sectors: list[Tensor] = []
    for sector in SECTORS:
        if count_objective is not None:
            count_objective()
        def phi(z: Tensor, p: Tensor, sides: tuple[int, int] = sector) -> Tensor:
            control = lift(z)
            with _sector_extension(problem, sides):
                return problem.objective(control, p)
        sectors.append(torch.func.grad(phi, argnums=0)(tangent, parameters))
    return _weighted_tensor_sum(weights, sectors)


def _twoface_exact_base_response(problem: Any, lift: Any, tangent: Tensor,
        parameters: Tensor, weights: Tensor, hessian: Tensor, parameter_index: int,
        charge_objective: Callable[[], None], charge_score: Callable[[], None]) -> dict[str, float]:
    def phi_sector(z: Tensor, p: Tensor, sector: tuple[int, int]) -> Tensor:
        from advar import transport
        control = lift(z)
        selected = tuple((face[0], face[1], face[2], side) for face, side in zip(FACES, sector))
        with transport.selected_face_extensions(*selected):
            return problem.objective(control, p)

    def mixed_gradient(z: Tensor, p: Tensor) -> Tensor:
        sector_gradients = []
        for index, sector in enumerate(SECTORS):
            charge_objective()
            sector_gradients.append(torch.func.grad(
                lambda zz, pp, _sector=sector: phi_sector(zz, pp, _sector), argnums=0)(z, p))
        return _weighted_tensor_sum(weights, sector_gradients)

    def score(z: Tensor, p: Tensor) -> Tensor:
        return problem.score(lift(z), p)

    p = parameters.detach().clone().requires_grad_(True)
    charge_score()
    ez = torch.func.grad(score, argnums=0)(tangent, p)
    charge_score()
    ep = torch.func.grad(score, argnums=1)(tangent, p)[parameter_index]
    parameter_direction = torch.zeros_like(p)
    parameter_direction[parameter_index] = 1.0
    fp = torch.func.jvp(lambda pp: mixed_gradient(tangent, pp),
        (p,), (parameter_direction,))[1]
    multiplier = torch.linalg.solve(hessian.T, ez)
    prediction = ep - torch.dot(multiplier, fp)
    if not bool(torch.isfinite(prediction)):
        raise ValueError("two-face exact response derivative is nonfinite")
    return {"E_p": float(ep.detach()),
        "lambda_F_zp": float(torch.dot(multiplier, fp).detach()),
        "predicted_dE_dp": float(prediction.detach()),
        "linear_system_residual_norm": float(torch.linalg.vector_norm(hessian.T @ multiplier - ez).detach())}


def _qualified_twoface_perturbation(value: Any, control: Tensor, parameters: Tensor,
        chart_metadata: dict[str, Any], base_signature: str) -> bool:
    if not isinstance(value, dict):
        return False
    final = value.get("final_point", value)
    if not isinstance(final, dict):
        return False
    branch = final.get("branch")
    if not isinstance(branch, dict):
        return False
    if (value.get("status") != "twoface_first_order_qualified"
            or value.get("input_parameters_sha256") != _tensor_sha(parameters)
            or final.get("control_sha256", final.get("event_control_sha256")) != _tensor_sha(control)
            or branch.get("stage_count") != 360
            or branch.get("classification") != "multiplefaces"
            or branch.get("unique_nearzero_face_entities") != ["y:2:1", "y:4:3"]
            or branch.get("unique_near_tie_slope_entities") != []
            or branch.get("actual_selector_signature_sha256") != base_signature
            or final.get("branch_supported_exactly_two_faces") is not True
            or final.get("first_order_pass") is not True):
        return False
    hull = final.get("full_clarke_hull")
    tangential = final.get("tangent_hull")
    if not isinstance(hull, dict) or not isinstance(tangential, dict):
        return False
    try:
        full_inf = float(torch.as_tensor(hull["vector"], dtype=torch.float64).abs().max())
        tangent_inf = float(torch.as_tensor(tangential["vector"], dtype=torch.float64).abs().max())
    except (KeyError, TypeError, ValueError):
        return False
    sectors = final.get("sectors")
    normals = final.get("normal_coefficients")
    reference_weights = hull.get("weights")
    return (full_inf < STATIONARITY_TOL and tangent_inf < STATIONARITY_TOL
        and isinstance(sectors, list) and len(sectors) == 4
        and all(row.get("sides") == list(sides)
            for row, sides in zip(sectors, SECTORS))
        and isinstance(reference_weights, list) and len(reference_weights) == 4
        and all(math.isfinite(float(weight)) and float(weight) >= 0 for weight in reference_weights)
        and abs(sum(float(weight) for weight in reference_weights) - 1.0) <= 256 * torch.finfo(torch.float64).eps
        and isinstance(normals, list) and len(normals) == 4
        and all(isinstance(row, dict) and row.get("sides") == list(sides)
            and row.get("strict_sign_match") == [True, True]
            for row, sides in zip(normals, SECTORS))
        and value.get("same_actual_branch") is True
        and final.get("chart", chart_metadata) == chart_metadata)


def qualify_two_face_point(problem: Any, control: Tensor, parameters: Tensor,
        budget: dict[str, Any], *, lift: Any, chart_metadata: dict[str, Any],
        base_parameters: Tensor, base_branch_signature: str) -> dict[str, Any]:
    """Actual first-order 24D/4-sector face audit; computes no Hessian or score."""
    if (control.shape != (26,) or parameters.shape != (13,) or base_parameters.shape != (13,)
            or control.dtype != torch.float64 or parameters.dtype != torch.float64
            or base_parameters.dtype != torch.float64 or control.device.type != "cpu"
            or parameters.device.type != "cpu" or base_parameters.device.type != "cpu"
            or not bool(torch.isfinite(control).all())
            or not bool(torch.isfinite(parameters).all())
            or not bool(torch.isfinite(base_parameters).all())):
        raise ValueError("two-face qualification requires finite CPU FP64 control[26], parameters[13]")
    parameter_index = budget.get("declared_parameter_index")
    delta = budget.get("declared_parameter_delta")
    if (parameter_index not in (0, 12) or not isinstance(delta, (float, int))
            or not math.isfinite(float(delta))):
        raise ValueError("two-face callback requires one declared observation/beta transition")
    expected = base_parameters.clone()
    expected[parameter_index] += float(delta)
    if not torch.equal(expected, parameters) or budget.get("base_parameters_sha256") != _tensor_sha(base_parameters):
        raise ValueError("two-face callback received an undeclared parameter transition")
    deadline = float(budget["phase_deadline_monotonic"])
    phase_limit = int(budget["max_phase_native_J_successes"])

    def check() -> None:
        if time.monotonic() >= deadline or int(budget["native_J_successes_completed"]) >= phase_limit:
            raise TimeoutError("two-face qualification exhausted shared phase budget")

    problem.contract(parameters)
    from advar import transport
    # The separately frozen two-face phase plan, not continuation3's older
    # single-face source pin, authorizes this current tuple-extension API.
    current_transport_sha = _sha_bytes((ROOT / TRANSPORT_PATH).read_bytes())
    if current_transport_sha != SUPPORT_AFTER_SHA:
        raise ValueError("tuple-based selected_face_extensions source pin is not active")
    q = _two_face_flux_vector(control, problem)
    qx, qy = transport.face_volume_fluxes(torch.einsum("k,kij->ij",
        problem.frozen.fv_transport.coefficient_limits * torch.tanh(control[20:25]),
        problem.frozen.fv_transport.psi_basis))
    flux_scale = max(float(qx.abs().max()), float(qy.abs().max()), torch.finfo(control.dtype).tiny)
    q_budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * flux_scale
    if float(q.abs().max()) > q_budget:
        raise ValueError("two-face chart point does not realize both Q=0 events")
    tangent = next((value for value in budget.get("chart_coordinates", [])), None)
    if tangent is None:
        # Chart coordinates are reconstructed from the diagnostic's exact
        # 24D transform using its pivot/free metadata.
        diagnostic = _load_two_face_geometry()
        spec = problem.frozen.fv_transport
        matrix, pivots, free, _meta = diagnostic._chart(spec.coefficient_limits, spec.psi_basis)
        _lift, tangent, projected, _flow = diagnostic._make_intersection_chart(control, matrix, pivots, free)
        if _tensor_sha(projected) != _tensor_sha(control):
            raise ValueError("two-face callback control does not round-trip through Γ₂₄")
    if tangent.shape != (TANGENT_SIZE,) or _tensor_sha(lift(tangent)) != _tensor_sha(control):
        raise ValueError("two-face callback control does not match its 24D chart coordinates")
    jacobian = torch.func.jacrev(lift)(tangent)
    if jacobian.shape != (26, 24) or not bool(torch.isfinite(jacobian).all()):
        raise ValueError("two-face tangent Jacobian must be finite 26x24")

    check()
    budget["native_J_successes_started"] += 1
    red = _load_red_qualifier()
    branch = red._analysis_branch(problem, control, parameters, deadline=deadline)
    if branch.get("objective_finite") is not True or branch.get("native_J_on_audited_path") is None:
        raise ValueError("two-face actual 360-stage branch audit returned invalid native J")
    budget["native_J_successes_completed"] += 1
    signature = branch.get("actual_selector_signature_sha256")
    if (branch.get("classification") != "multiplefaces" or branch.get("stage_count") != 360
            or branch.get("unique_nearzero_face_entities") != ["y:2:1", "y:4:3"]
            or branch.get("unique_near_tie_slope_entities") != []
            or signature != base_branch_signature):
        return {"status": "unsupported_twoface_branch", "event_control": control.tolist(),
            "event_control_sha256": _tensor_sha(control), "input_parameters_sha256": _tensor_sha(parameters),
            "native_J": float(branch["native_J_on_audited_path"]), "branch": branch,
            "same_actual_branch": False, "selection_use": False}

    full_gradients: list[Tensor] = []
    tangent_gradients: list[Tensor] = []
    sector_rows: list[dict[str, Any]] = []
    for sides in SECTORS:
        check()
        budget["native_J_successes_started"] += 1
        selected = tuple((face[0], face[1], face[2], side) for face, side in zip(FACES, sides))
        with transport.selected_face_extensions(*selected):
            gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("two-face sector gradient is nonfinite")
        budget["native_J_successes_completed"] += 1
        tangent_gradient = jacobian.T @ gradient
        full_gradients.append(gradient)
        tangent_gradients.append(tangent_gradient)
        sector_rows.append({"sides": list(sides),
            "full_gradient_inf": float(gradient.abs().max()),
            "tangent_gradient_inf": float(tangent_gradient.abs().max()),
            "full_gradient_sha256": _tensor_sha(gradient),
            "tangent_gradient": tangent_gradient.tolist()})

    diagnostic = _load_two_face_geometry()
    normal_matrix = torch.func.jacrev(lambda c: _two_face_flux_vector(c, problem))(control)
    normal_coefficients = _sector_normal_coefficients(normal_matrix, full_gradients)
    geometry = _load_two_face_geometry()
    full_hull = geometry.geometry.minimum_sampled_hull(torch.stack(full_gradients))
    tangent_hull = geometry.geometry.minimum_sampled_hull(torch.stack(tangent_gradients))
    tangents = torch.stack(tangent_gradients)
    tangent_gap = float((tangents - tangents[:1]).abs().max())
    tangent_scale = max(float(tangents.norm()), torch.finfo(control.dtype).tiny)
    tangent_budget = ROUNDING_MULTIPLIER * torch.finfo(control.dtype).eps * tangent_scale
    full_mix_inf = float(torch.as_tensor(full_hull["vector"], dtype=control.dtype).abs().max())
    tangent_mix_inf = float(torch.as_tensor(tangent_hull["vector"], dtype=control.dtype).abs().max())
    tangent_stationary = (full_mix_inf < STATIONARITY_TOL and tangent_mix_inf < STATIONARITY_TOL
        and tangent_gap <= tangent_budget)
    normal_pass = all(all(row["strict_sign_match"]) for row in normal_coefficients)
    first_order = tangent_stationary and normal_pass
    return {
        "status": "twoface_first_order_qualified" if first_order else "twoface_first_order_not_qualified",
        "event_control": control.tolist(), "event_control_sha256": _tensor_sha(control),
        "input_parameters_sha256": _tensor_sha(parameters),
        "native_J": float(branch["native_J_on_audited_path"]),
        "event_faces": [list(face) for face in FACES], "event_face_values": q.tolist(),
        "chart": chart_metadata, "branch": branch,
        "actual_selector_signature_sha256": signature,
        "same_actual_branch": signature == base_branch_signature,
        "branch_supported_exactly_two_faces": True,
        "sector_gradients": sector_rows, "sectors": sector_rows,
        "full_side_gradient_inf_by_sector": [row["full_gradient_inf"] for row in sector_rows],
        "tangent_sector_gradients": [row["tangent_gradient"] for row in sector_rows],
        "tangent_gradient_agreement_inf": tangent_gap,
        "tangent_gradient_agreement_budget": tangent_budget,
        "full_clarke_hull": full_hull, "tangent_hull": tangent_hull,
        "normal_matrix": normal_matrix.tolist(), "normal_coefficients": normal_coefficients,
        "minimum_mixture_weights": full_hull["weights"],
        "tangent_stationary": tangent_stationary, "normal_signs_strict": normal_pass,
        "first_order_pass": first_order, "curvature_computed": False,
        "selection_use": False,
    }




def run_two_face_response(problem: Any, parameters: Tensor, qualification: dict[str, Any], *,
        phase_plan_path: Path, phase_plan_sha256: str, output: Path) -> dict[str, Any]:
    """Run actual four-sector 24D reoptimized response after both launch gates."""
    if output.exists():
        raise ValueError("two-face response output must be fresh")
    invocation_start = time.monotonic()
    runner_sha_before = _sha_bytes(SELF.read_bytes())
    parameters_sha_before = _tensor_sha(parameters)
    try:
        base_control, reference_weights, base_signature, qualification_plan_sha, point = _qualified_twoface_base(
            qualification, parameters)
    except (ValueError, KeyError, TypeError) as error:
        return _gate(output, "gate_pending_base", str(error), plan_sha=phase_plan_sha256)
    if (parameters.shape != (13,) or parameters.dtype != torch.float64
            or parameters.device.type != "cpu" or not bool(torch.isfinite(parameters).all())):
        raise ValueError("two-face active response requires fixed CPU FP64 parameters[13]")
    identity = getattr(problem, "identity", None)
    if not isinstance(identity, dict) or identity.get("fixed_problem_sha256") != FIXED_PROBLEM_SHA:
        return _gate(output, "gate_pending_fixed_case",
            "problem identity differs from the frozen fixed case",
            control_sha=_tensor_sha(base_control), plan_sha=phase_plan_sha256)
    base_control_sha_before = _tensor_sha(base_control)
    lift, tangent, chart_metadata = _two_face_chart(base_control, problem)
    try:
        plan = _twoface_phase_plan(phase_plan_path, phase_plan_sha256,
            qualification, base_control, parameters, chart_metadata, qualification_plan_sha)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return _gate(output, "gate_pending_launch_clearance", str(error),
            control_sha=_tensor_sha(base_control), plan_sha=phase_plan_sha256)
    hessians, mixed_hessian = _two_face_base_hessians(qualification, reference_weights)
    try:
        factor = torch.linalg.cholesky(mixed_hessian)
    except RuntimeError:
        return _gate(output, "gate_pending_base", "reference-weighted four-sector Hessian is not SPD",
            control_sha=_tensor_sha(base_control), plan_sha=phase_plan_sha256)
    del hessians
    budget_plan = plan["budget"]
    phase_deadline = invocation_start + float(budget_plan["phase_seconds"])
    native_j_before = int(budget_plan["native_J_successes_before"])
    phase_limit = int(budget_plan["phase_native_J_limit"])
    budget: dict[str, Any] = {"native_J_successes_before": native_j_before,
        "native_J_successes_started": 0, "native_J_successes_completed": 0,
        "score_evaluations_started": 0, "score_evaluations_completed": 0,
        "max_phase_native_J_successes": phase_limit,
        "phase_deadline_monotonic": phase_deadline}

    def write_budget_failure(reason: str) -> dict[str, Any]:
        return _write(output, {"status": "partial_budget_exhausted", "reason": reason,
            "base_control_sha256": _tensor_sha(base_control),
            "base_parameters_sha256": _tensor_sha(parameters),
            "phase_plan_sha256": phase_plan_sha256, "budget": budget,
            "minimum_claim": False, "response_claim": False, "selection_use": False,
            "elapsed_seconds": time.monotonic() - invocation_start})

    def check_wall_budget() -> None:
        if time.monotonic() >= phase_deadline:
            write_budget_failure("two-face original 3,600-second cap exhausted")
            raise TimeoutError("two-face response shared wall deadline reached")

    def check_budget() -> None:
        check_wall_budget()
        if int(budget["native_J_successes_completed"]) >= phase_limit:
            write_budget_failure("two-face phase native-J allowance exhausted")
            raise TimeoutError("two-face response phase J cap reached")

    def charge_objective() -> None:
        check_budget()
        budget["native_J_successes_started"] += 1
        budget["native_J_successes_completed"] += 1

    def charge_score() -> None:
        check_wall_budget()
        budget["score_evaluations_started"] += 1
        budget["score_evaluations_completed"] += 1

    def phi_sector(z: Tensor, p: Tensor, sides: tuple[int, int]) -> Tensor:
        from advar import transport
        control = lift(z)
        selected = tuple((face[0], face[1], face[2], side) for face, side in zip(FACES, sides))
        with transport.selected_face_extensions(*selected):
            check_budget()
            budget["native_J_successes_started"] += 1
            value = problem.objective(control, p)
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError("two-face native-J sector value is nonfinite")
        budget["native_J_successes_completed"] += 1
        return value

    def phi_for_ad(z: Tensor, p: Tensor, sides: tuple[int, int]) -> Tensor:
        from advar import transport
        selected = tuple((face[0], face[1], face[2], side) for face, side in zip(FACES, sides))
        with transport.selected_face_extensions(*selected):
            return problem.objective(lift(z), p)

    def mixed_gradient(z: Tensor, p: Tensor, *, count: bool) -> Tensor:
        gradients = []
        for sector in SECTORS:
            if count:
                check_budget()
                budget["native_J_successes_started"] += 1
            gradient = torch.func.grad(
                lambda zz, pp, _sector=sector: phi_for_ad(zz, pp, _sector), argnums=0)(z, p)
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError("two-face tangent native gradient is nonfinite")
            if count:
                budget["native_J_successes_completed"] += 1
            gradients.append(gradient)
        weights = torch.as_tensor(reference_weights, dtype=z.dtype, device="cpu")
        return _weighted_tensor_sum(weights, gradients)

    base_native_j = float(point["native_J"])
    rows: list[dict[str, Any]] = []
    for parameter_index, label in PARAMETERS:
        h = (float(problem.observation_std_dbz.reshape(-1)[0]) / 100.0
            if parameter_index == 0 else 1.0e-3)
        try:
            exact = _twoface_exact_base_response(problem, lift, tangent, parameters,
                torch.as_tensor(reference_weights, dtype=tangent.dtype), mixed_hessian,
                parameter_index, charge_objective, charge_score)
        except TimeoutError:
            return _read(output)
        except (ValueError, RuntimeError) as error:
            return _write(output, {"status": "base_twoface_adjoint_failed", "reason": str(error),
                "parameter_index": parameter_index, "base_control_sha256": _tensor_sha(base_control),
                "phase_plan_sha256": phase_plan_sha256, "budget": budget,
                "minimum_claim": False, "response_claim": False, "selection_use": False})
        point_rows: list[dict[str, Any]] = []
        for scale in (1.0, -1.0, 0.5, -0.5):
            p = parameters.clone()
            delta = scale * h
            p[parameter_index] += delta
            z = tangent.clone()
            budget["base_parameters_sha256"] = _tensor_sha(parameters)
            budget["declared_parameter_index"] = parameter_index
            budget["declared_parameter_delta"] = delta
            budget["chart_coordinates"] = z.tolist()
            trace: list[dict[str, Any]] = []
            converged = False
            try:
                for iteration in range(int(plan["armijo_iterations"])):
                    check_budget()
                    value = phi_sector(z, p, SECTORS[-1])
                    gradient = mixed_gradient(z, p, count=True)
                    gradient_inf = float(gradient.abs().max())
                    trace.append({"iteration": iteration, "native_J": float(value),
                        "tangent_gradient_inf": gradient_inf})
                    if gradient_inf < STATIONARITY_TOL:
                        converged = True
                        break
                    direction = torch.cholesky_solve((-gradient).reshape(-1, 1), factor).reshape(-1)
                    slope = float(torch.dot(gradient, direction))
                    if not math.isfinite(slope) or slope >= 0:
                        break
                    accepted = False
                    for backtrack in range(int(plan["armijo_backtracks"])):
                        alpha = 0.5 ** backtrack
                        trial = z + alpha * direction
                        try:
                            trial_value = phi_sector(trial, p, SECTORS[-1])
                        except (ValueError, RuntimeError):
                            continue
                        if float(trial_value) <= float(value) + 1e-4 * alpha * slope:
                            z = trial
                            trace[-1]["accepted_alpha"] = alpha
                            accepted = True
                            break
                    if not accepted:
                        break
            except TimeoutError:
                return _read(output)
            if not converged:
                return _write(output, {"status": "twoface_reoptimization_not_stationary",
                    "parameter_index": parameter_index, "scale": scale, "trace": trace,
                    "budget": budget, "minimum_claim": False, "response_claim": False,
                    "selection_use": False, "base_control_sha256": _tensor_sha(base_control)})
            candidate_control = lift(z)
            check_budget()
            try:
                qualification_point = qualify_two_face_point(problem, candidate_control, p, budget,
                    lift=lift, chart_metadata=chart_metadata,
                    base_parameters=parameters, base_branch_signature=base_signature)
            except TimeoutError:
                return _read(output)
            except (ValueError, RuntimeError, KeyError, TypeError) as error:
                return _write(output, {"status": "twoface_perturbed_first_order_gate_failed",
                    "reason": str(error), "parameter_index": parameter_index, "scale": scale,
                    "candidate_control_sha256": _tensor_sha(candidate_control),
                    "parameters_sha256": _tensor_sha(p), "trace": trace, "budget": budget,
                    "minimum_claim": False, "response_claim": False, "selection_use": False})
            if not _qualified_twoface_perturbation(qualification_point,
                    candidate_control, p, chart_metadata, base_signature):
                return _write(output, {"status": "twoface_perturbed_first_order_gate_failed",
                    "parameter_index": parameter_index, "scale": scale,
                    "qualification": qualification_point, "trace": trace, "budget": budget,
                    "minimum_claim": False, "response_claim": False, "selection_use": False})
            check_wall_budget()
            budget["score_evaluations_started"] += 1
            score = problem.score(candidate_control, p)
            if score.ndim != 0 or not bool(torch.isfinite(score)):
                raise ValueError("two-face terminal score must be a finite scalar")
            budget["score_evaluations_completed"] += 1
            point_rows.append({"scale": scale, "parameter_delta": delta,
                "optimizer_tangent_sha256": _tensor_sha(z),
                "qualified_control_sha256": _tensor_sha(candidate_control),
                "parameters_sha256": _tensor_sha(p),
                "native_J": float(qualification_point["native_J"]), "score": float(score),
                "full_clarke_gradient_inf": float(qualification_point["full_clarke_hull"]["vector"]
                    and torch.as_tensor(qualification_point["full_clarke_hull"]["vector"]).abs().max()),
                "tangent_gradient_inf": float(torch.as_tensor(
                    qualification_point["tangent_hull"]["vector"]).abs().max()),
                "branch_signature": base_signature, "armijo_trace": trace,
                "perturbed_curvature_certified": False})
        by_scale = {float(item["scale"]): item for item in point_rows}
        derivative_h = (float(by_scale[1.0]["score"]) - float(by_scale[-1.0]["score"])) / (2.0 * h)
        derivative_half = (float(by_scale[0.5]["score"]) - float(by_scale[-0.5]["score"])) / h
        rows.append({"parameter_index": parameter_index, "parameter": label, "h": h,
            "exact_twoface_adjoint": exact, "reoptimized_dE_dp_h": derivative_h,
            "reoptimized_dE_dp_h_over_2": derivative_half,
            "central_difference_refinement_change": derivative_half - derivative_h,
            "points": point_rows})

    source_after_ok = True
    for name, digest in {**plan["source_files"], **plan["archive_files"]}.items():
        if _sha_bytes((ROOT / name).read_bytes()) != digest:
            source_after_ok = False
            break
    if (not source_after_ok or _sha_bytes(SELF.read_bytes()) != runner_sha_before
            or _tensor_sha(base_control) != base_control_sha_before
            or _tensor_sha(parameters) != parameters_sha_before):
        return _write(output, {"status": "twoface_source_or_input_closure_failed",
            "base_control_sha256": _tensor_sha(base_control), "budget": budget,
            "minimum_claim": False, "response_claim": False, "selection_use": False})
    elapsed = time.monotonic() - invocation_start
    return _write(output, {"status": "completed", "phase_plan_sha256": phase_plan_sha256,
        "phase_plan_path": str(phase_plan_path),
        "runtime_identity": {"python": sys.version.split()[0], "torch": torch.__version__,
            "device": "CPU FP64"},
        "qualification_plan_sha256": qualification_plan_sha,
        "qualification_sha256": _sha_bytes(json.dumps(qualification, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()),
        "base_control_sha256": _tensor_sha(base_control),
        "base_parameters_sha256": _tensor_sha(parameters),
        "base_native_J": base_native_j, "base_faces": [list(face) for face in FACES],
        "reference_sector_weights": reference_weights,
        "chart_metadata": chart_metadata, "source_transition": plan["source_transition"],
        "source_hashes": plan["source_files"], "archive_hashes": plan["archive_files"],
        "phase_budget": {**budget, "charged_seconds_before": plan["budget"]["charged_seconds_before"],
            "phase_seconds": plan["budget"]["phase_seconds"], "elapsed_seconds": elapsed,
            "global_elapsed_after_seconds": float(plan["budget"]["charged_seconds_before"]) + elapsed,
            "native_J_successes_after": int(plan["budget"]["native_J_successes_before"])
                + int(budget["native_J_successes_completed"]),
            "no_new_3600_second_grant": True},
        "implicit_equation":"base-reference-weighted four-sector 24D tangent gradient; weights fixed at base",
        "perturbed_curvature_certified": False, "minimum_claim": False,
        "response_claim": True, "selection_use": False, "parameters": rows})
