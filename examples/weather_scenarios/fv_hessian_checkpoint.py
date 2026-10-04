"""Receipt-bound, resumable matrix-free Hessian checkpointing for 26 controls."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, cast

import torch
from torch import Tensor

from examples.weather_scenarios import fv_diagnostic_guard
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks

NC = 26
_SCHEMA = "advar.hessian-checkpoint.v1"
_RECEIPT_KEYS = {"objective", "problem", "coordinates", "source", "runtime", "branch", "analysis_context"}


class CheckpointRefusal(RuntimeError):
    """A typed resume/qualification refusal carrying committed progress."""

    def __init__(self, message: str, state: dict[str, Any]) -> None:
        super().__init__(message)
        self.state = state


class BudgetRefusal(CheckpointRefusal):
    """The absolute deadline, attempt cap or attempt quota stopped progress."""


CheckpointBudgetRefusal = BudgetRefusal


class ReceiptRefusal(CheckpointRefusal):
    """Control, parameters, gradient or caller identity changed across an HVP."""


class NumericalRefusal(CheckpointRefusal):
    """A finite, symmetry or independent eigenpair qualification failed."""


def tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _validate_tensor(name: str, value: Tensor) -> None:
    if (value.shape != (NC,) or value.dtype != torch.float64 or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be finite CPU FP64 length 26")


def _validate_parameters(value: Tensor) -> None:
    if (value.ndim != 1 or value.numel() < 1 or value.dtype != torch.float64
            or value.device.type != "cpu" or not bool(torch.isfinite(value).all())):
        raise ValueError("parameters must be finite CPU FP64 rank-1")


def _receipt(provider: Any, control: Tensor, parameters: Tensor, gradient: Tensor) -> dict[str, Any]:
    """Bind caller-attested objective/data identity to live numerical inputs.

    The provider must describe the exact objective callable, including its
    closure/data and source identity. The kernel records and rechecks that
    attestation; it does not generically authenticate arbitrary Python callables.
    """
    context = provider()
    if not isinstance(context, dict) or not _RECEIPT_KEYS <= set(context):
        raise ValueError("receipt_provider must return objective/problem/coordinates/source/runtime/branch/analysis_context")
    data = {"control": {"values": control.tolist(), "sha256": tensor_sha(control)},
            "parameters": {"values": parameters.tolist(), "sha256": tensor_sha(parameters)},
            "gradient": {"values": gradient.tolist(), "sha256": tensor_sha(gradient)},
            "context": context}
    return json.loads(json.dumps(data, sort_keys=True, allow_nan=False))


def _progress(checkpoint: dict[str, Any], directory: Path, status: str,
              receipt: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"status": status, "completed_columns": len(checkpoint["columns"]),
            "hvp_columns": len(checkpoint["columns"]),
            "hvp_calls_total": len(checkpoint["columns"]) + int(checkpoint["audit_hvp"] is not None),
            "attempts_reserved": len(checkpoint["attempts"]), "receipt": receipt,
            "checkpoint_dir": str(directory), **extra}


def _save(path: Path, checkpoint: dict[str, Any]) -> None:
    fv_diagnostic_guard.atomic_write_text(
        path, json.dumps(checkpoint, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _header(control: Tensor, parameters: Tensor, gradient: Tensor,
            receipt: dict[str, Any], max_attempts: int,
            attempt_seconds: float) -> dict[str, Any]:
    return {"max_attempts": max_attempts, "attempt_seconds": attempt_seconds,
            "control": {"values": control.tolist(), "sha256": tensor_sha(control)},
            "parameters": {"values": parameters.tolist(), "sha256": tensor_sha(parameters)},
            "gradient": {"values": gradient.tolist(), "sha256": tensor_sha(gradient)},
            "receipt": receipt}


class _WriterClaim:
    """Exclusive lock; only this process's clean claim is removed."""

    def __init__(self, path: Path) -> None:
        with path.open("x") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stat = os.fstat(stream.fileno())
        self.path = path
        self.identity = (stat.st_dev, stat.st_ino)

    def release(self) -> None:
        try:
            stat = self.path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return
        if (stat.st_dev, stat.st_ino) == self.identity:
            self.path.unlink()


def _checkpoint_state(checkpoint: dict[str, Any], status: str) -> None:
    checkpoint["status"] = status
    checkpoint["last_status"] = status


def _validate_checkpoint(checkpoint: dict[str, Any], header: dict[str, Any],
                         max_attempts: int, attempt_seconds: float) -> None:
    if checkpoint.get("schema") != _SCHEMA or checkpoint.get("header") != header:
        raise ValueError("checkpoint endpoint identity or attempt-budget header changed")
    attempts = checkpoint.get("attempts")
    columns = checkpoint.get("columns")
    if (not isinstance(attempts, list) or len(attempts) > max_attempts
            or not isinstance(columns, list) or len(columns) > NC):
        raise ValueError("checkpoint attempt/column ledger is malformed")
    attempt_states = {"reserved", "columns_pending", "audit_pending", "audit_validation_pending",
                      "schur_pending", "completed", "budget_refused", "receipt_refused",
                      "numerical_refused", "programming_error"}
    for index, item in enumerate(attempts, start=1):
        if (not isinstance(item, dict) or item.get("attempt") != index
                or item.get("reserved_seconds") != attempt_seconds
                or item.get("status") not in attempt_states):
            raise ValueError("checkpoint attempt ledger lost its full-cap reservation")
    for index, row in enumerate(columns):
        if not isinstance(row, dict):
            raise ValueError("checkpoint column entry must be an object")
        direction = torch.tensor(row.get("direction"), dtype=torch.float64)
        product = torch.tensor(row.get("product"), dtype=torch.float64)
        expected = torch.zeros(NC, dtype=torch.float64)
        expected[index] = 1.0
        if (row.get("index") != index or direction.shape != (NC,) or product.shape != (NC,)
                or not bool(torch.isfinite(direction).all() & torch.isfinite(product).all())
                or not torch.equal(direction, expected)
                or tensor_sha(direction) != row.get("direction_sha256")
                or tensor_sha(product) != row.get("product_sha256")
                or row.get("receipt_before") != header["receipt"]
                or row.get("receipt_after") != header["receipt"]):
            raise ValueError("checkpoint HVP column is malformed or lost its endpoint receipt")
    audit = checkpoint.get("audit_hvp")
    if audit is not None:
        if not isinstance(audit, dict):
            raise ValueError("checkpoint audit HVP must be an object")
        direction = torch.tensor(audit.get("direction"), dtype=torch.float64)
        product = torch.tensor(audit.get("product"), dtype=torch.float64)
        if (len(columns) != NC or direction.shape != (NC,) or product.shape != (NC,)
                or not bool(torch.isfinite(direction).all() & torch.isfinite(product).all())
                or tensor_sha(direction) != audit.get("direction_sha256")
                or tensor_sha(product) != audit.get("product_sha256")
                or audit.get("receipt_before") != header["receipt"]
                or audit.get("receipt_after") != header["receipt"]):
            raise ValueError("checkpoint independent HVP is malformed or lost its endpoint receipt")


def _make_state(checkpoint: dict[str, Any], directory: Path, status: str,
                receipt: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return _progress(checkpoint, directory, status, receipt, **extra)


def _record_refusal(checkpoint: dict[str, Any], path: Path, directory: Path,
                    status: str, receipt: dict[str, Any], *,
                    attempt_status: str | None = None, **extra: Any) -> dict[str, Any]:
    _checkpoint_state(checkpoint, status)
    if attempt_status is not None and checkpoint["attempts"]:
        checkpoint["attempts"][-1]["status"] = attempt_status
    state = _make_state(checkpoint, directory, status, receipt, **extra)
    checkpoint["last_refusal"] = state
    _save(path, checkpoint)
    return state


def _attempt_status(status: str) -> str:
    if "receipt" in status:
        return "receipt_refused"
    if status in {"invalid_hvp_product", "hessian_invalid", "hessian_symmetry_refused",
                  "eigensolver_refused", "eigensystem_nonfinite", "audit_hvp_invalid",
                  "eigenpair_refused", "schur_refused"}:
        return "numerical_refused"
    return "budget_refused"


def _basis(index: int) -> Tensor:
    direction = torch.zeros(NC, dtype=torch.float64)
    direction[index] = 1.0
    return direction


def _apply_hvp(objective: Any, parameters: Tensor, control: Tensor,
               direction: Tensor) -> Tensor:
    grad_fn = torch.func.grad(objective, argnums=0)
    return torch.func.jvp(lambda value: grad_fn(value, parameters), (control,), (direction,))[1]


def _build_final(checkpoint: dict[str, Any], directory: Path, receipt: dict[str, Any],
                 control: Tensor, gradient: Tensor) -> dict[str, Any]:
    columns = [torch.tensor(row["product"], dtype=torch.float64) for row in checkpoint["columns"]]
    hessian = torch.stack(columns, dim=1)
    norm = torch.linalg.matrix_norm(hessian)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        state = _make_state(checkpoint, directory, "hessian_invalid", receipt)
        raise NumericalRefusal("fresh Hessian has no positive finite scale", state)
    symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T) / norm)
    if not math.isfinite(symmetry) or symmetry > blocks.SYMMETRY_TOL:
        state = _make_state(checkpoint, directory, "hessian_symmetry_refused", receipt,
                            symmetry_relative=symmetry)
        raise NumericalRefusal("fresh Hessian symmetry gate failed", state)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (hessian + hessian.T))
    except torch.linalg.LinAlgError as error:
        state = _make_state(checkpoint, directory, "eigensolver_refused", receipt)
        raise NumericalRefusal(f"full-H eigensolver failed: {error}", state) from error
    if not bool(torch.isfinite(eigenvalues).all() & torch.isfinite(eigenvectors).all()):
        state = _make_state(checkpoint, directory, "eigensystem_nonfinite", receipt)
        raise NumericalRefusal("fresh Hessian eigensystem is nonfinite", state)
    return {"hessian_tensor": hessian, "hessian": hessian.tolist(),
            "hessian_sha256": tensor_sha(hessian), "norm": norm,
            "symmetry_relative": symmetry, "eigenvalues_tensor": eigenvalues,
            "eigenvectors_tensor": eigenvectors, "eigenvalues": eigenvalues.tolist(),
            "control": control, "gradient": gradient}


def _verify_completed(checkpoint: dict[str, Any], receipt: dict[str, Any],
                      gradient: Tensor) -> dict[str, Any]:
    """Authenticate cached completion against committed columns and audit HVP."""
    result = checkpoint.get("final_result")
    audit = checkpoint.get("audit_hvp")
    if (not isinstance(result, dict) or len(checkpoint["columns"]) != NC
            or not isinstance(audit, dict) or audit.get("status") != "passed"
            or not checkpoint["attempts"] or checkpoint["attempts"][-1].get("status") != "completed"):
        raise ValueError("completed checkpoint is missing its full HVP/eigenpair evidence")
    hessian = torch.stack([torch.tensor(row["product"], dtype=torch.float64)
                           for row in checkpoint["columns"]], dim=1)
    if (result.get("hessian") != hessian.tolist()
            or result.get("hessian_sha256") != tensor_sha(hessian)
            or result.get("hvp_columns") != NC or result.get("hvp_calls_total") != NC + 1
            or result.get("checkpoint_status") != "completed"
            or result.get("attempts_reserved") != len(checkpoint["attempts"])
            or result.get("receipt") != receipt):
        raise ValueError("completed result differs from its committed Hessian columns")
    norm = torch.linalg.matrix_norm(hessian)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise ValueError("completed checkpoint Hessian has no valid scale")
    symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T) / norm)
    if not math.isfinite(symmetry) or symmetry > blocks.SYMMETRY_TOL:
        raise ValueError("completed checkpoint Hessian symmetry no longer qualifies")
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (hessian + hessian.T))
    if not bool(torch.isfinite(eigenvalues).all() & torch.isfinite(eigenvectors).all()):
        raise ValueError("completed checkpoint eigensystem is nonfinite")
    direction = torch.tensor(audit["direction"], dtype=torch.float64)
    product = torch.tensor(audit["product"], dtype=torch.float64)
    if (not torch.equal(direction, eigenvectors[:, 0])
            or audit.get("direction_sha256") != tensor_sha(eigenvectors[:, 0])
            or result.get("symmetry_relative") != symmetry
            or result.get("eigenvalues") != eigenvalues.tolist()):
        raise ValueError("completed checkpoint eigensystem/audit direction is inconsistent")
    eigenpair = blocks.eigenpair_audit(product, eigenvalues[0], eigenvectors[:, 0], norm)
    if (eigenpair != audit.get("minimum_eigenpair_audit")
            or result.get("minimum_eigenpair_audit") != eigenpair):
        raise ValueError("completed checkpoint eigenpair audit is inconsistent")
    schur = blocks.schur_diagnostic(hessian, gradient)
    if schur != result.get("block_schur"):
        raise ValueError("completed checkpoint Schur record is inconsistent")
    return result


def checkpoint_hessian(objective: Any, control: Tensor, parameters: Tensor, gradient: Tensor, *,
                       receipt_provider: Any, checkpoint_dir: Path, deadline: float,
                       max_attempts: int, attempt_seconds: float,
                       max_new_columns: int | None = None) -> dict[str, Any]:
    """Resume a receipt-bound 26-column Hessian and independent 27th HVP audit.

    ``receipt_provider`` is a trusted caller attestation and must bind the exact
    objective callable, including its closure/data, as well as source/runtime
    identity. This kernel rechecks that attestation across HVPs; it does not
    generically authenticate arbitrary Python callables.
    """
    _validate_tensor("control", control)
    _validate_parameters(parameters)
    _validate_tensor("gradient", gradient)
    if (isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0
            or isinstance(attempt_seconds, bool) or not isinstance(attempt_seconds, (int, float))
            or not math.isfinite(float(attempt_seconds)) or attempt_seconds <= 0
            or isinstance(deadline, bool) or not isinstance(deadline, (int, float))
            or not math.isfinite(float(deadline))):
        raise ValueError("max_attempts/attempt_seconds/deadline are invalid")
    if (max_new_columns is not None and (isinstance(max_new_columns, bool)
            or not isinstance(max_new_columns, int) or not 0 <= max_new_columns <= NC)):
        raise ValueError("max_new_columns must be an integer in [0,26]")
    c, p, g = control.clone(), parameters.clone(), gradient.clone()
    receipt = _receipt(receipt_provider, c, p, g)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "hessian_checkpoint.json"
    expected_header = _header(c, p, g, receipt, max_attempts, float(attempt_seconds))
    lock = _WriterClaim(directory / ".writer.lock")
    checkpoint: dict[str, Any] = {}
    attempt_active = False
    try:
        if checkpoint_path.is_symlink():
            raise ValueError("checkpoint path cannot be a symlink")
        if checkpoint_path.exists():
            loaded: Any = json.loads(checkpoint_path.read_text())
            if not isinstance(loaded, dict):
                raise ValueError("checkpoint file must contain a JSON object")
            checkpoint = cast(dict[str, Any], loaded)
            saved_header = checkpoint.get("header")
            if not isinstance(saved_header, dict):
                raise ValueError("checkpoint has no immutable endpoint header")
            # A caller cannot extend an already-reserved attempt budget. Check
            # the immutable policy before recording any refusal or reserving work.
            if (saved_header.get("max_attempts") != expected_header["max_attempts"]
                    or saved_header.get("attempt_seconds") != expected_header["attempt_seconds"]):
                raise ValueError("checkpoint attempt-budget header changed")
            _validate_checkpoint(checkpoint, saved_header, max_attempts, float(attempt_seconds))
            if saved_header != expected_header:
                # A mismatched query does not own this checkpoint. Keep its
                # committed history and status untouched, including completion.
                state = _make_state(checkpoint, directory, "receipt_changed",
                                    cast(dict[str, Any], saved_header["receipt"]),
                                    observed_receipt=receipt)
                raise ReceiptRefusal("control/parameter/gradient/context changed since checkpoint", state)
            header = expected_header
        else:
            header = expected_header
            checkpoint = {"schema": _SCHEMA, "header": header, "attempts": [],
                          "columns": [], "audit_hvp": None, "status": "columns_pending"}
            _save(checkpoint_path, checkpoint)
        if checkpoint.get("status") == "completed":
            return _verify_completed(checkpoint, receipt, g)
        if checkpoint.get("status") == "programming_error":
            raise ValueError("checkpoint has a terminal programming error; start a new checkpoint explicitly")
        if time.monotonic() >= deadline:
            state = _make_state(checkpoint, directory, "deadline_expired", receipt)
            raise BudgetRefusal("absolute deadline expired before attempt reservation", state)
        if len(checkpoint["attempts"]) >= max_attempts:
            state = _record_refusal(checkpoint, checkpoint_path, directory,
                                    "attempt_limit_exhausted", receipt,
                                    attempt_status="budget_refused")
            raise BudgetRefusal("nonrefundable attempt quota is exhausted", state)

        attempt_start = time.monotonic()
        attempt_deadline = min(float(deadline), attempt_start + float(attempt_seconds))
        attempt = {"attempt": len(checkpoint["attempts"]) + 1,
                   "reserved_seconds": float(attempt_seconds),
                   "basis_start_column": len(checkpoint["columns"]), "status": "reserved"}
        checkpoint["attempts"].append(attempt)
        checkpoint["status"] = "attempt_running"
        _save(checkpoint_path, checkpoint)
        attempt_active = True
        new_columns = 0
        try:
            while len(checkpoint["columns"]) < NC:
                if time.monotonic() >= attempt_deadline:
                    raise BudgetRefusal("deadline expired during basis-column construction",
                                        _make_state(checkpoint, directory, "columns_pending", receipt))
                if max_new_columns is not None and new_columns >= max_new_columns:
                    raise BudgetRefusal("per-invocation basis-column cap reached",
                                        _make_state(checkpoint, directory, "columns_pending", receipt))
                index = len(checkpoint["columns"])
                direction = _basis(index)
                direction_sha = tensor_sha(direction)
                before = _receipt(receipt_provider, c, p, g)
                if before != receipt:
                    raise ReceiptRefusal("receipt changed before basis HVP",
                                         _make_state(checkpoint, directory, "receipt_changed", receipt,
                                                     receipt_before=before))
                tensor_hashes = (tensor_sha(c), tensor_sha(p), tensor_sha(g))
                hvp_started = time.monotonic()
                direction_argument = direction.clone()
                product = _apply_hvp(objective, p, c, direction_argument)
                hvp_elapsed = time.monotonic() - hvp_started
                after = _receipt(receipt_provider, c, p, g)
                attempt_elapsed = time.monotonic() - attempt_start
                if ((tensor_sha(c), tensor_sha(p), tensor_sha(g)) != tensor_hashes
                        or tensor_sha(direction) != direction_sha
                        or tensor_sha(direction_argument) != direction_sha
                        or before != after or after != receipt):
                    raise ReceiptRefusal("receipt/vector changed across basis HVP",
                                         _make_state(checkpoint, directory, "receipt_changed", receipt,
                                                     receipt_before=before, receipt_after=after))
                if (product.shape != (NC,) or product.dtype != torch.float64
                        or product.device.type != "cpu" or not bool(torch.isfinite(product).all())):
                    raise NumericalRefusal("basis HVP must be finite CPU FP64 length 26",
                                           _make_state(checkpoint, directory, "invalid_hvp_product", receipt,
                                                       attempted_column=index))
                checkpoint["columns"].append({
                    "index": index, "direction": direction.tolist(), "direction_sha256": direction_sha,
                    "product": product.tolist(), "product_sha256": tensor_sha(product),
                    "hvp_elapsed_seconds": hvp_elapsed,
                    "attempt_elapsed_seconds_at_commit": attempt_elapsed,
                    "receipt_before": before, "receipt_after": after,
                })
                new_columns += 1
                checkpoint["status"] = "columns_pending"
                _save(checkpoint_path, checkpoint)
                if time.monotonic() >= attempt_deadline:
                    phase = "audit_pending" if len(checkpoint["columns"]) == NC else "columns_pending"
                    raise BudgetRefusal("deadline expired after committed basis HVP",
                                        _make_state(checkpoint, directory, phase, receipt))
        except (BudgetRefusal, ReceiptRefusal, NumericalRefusal) as error:
            checkpoint["attempts"][-1]["status"] = _attempt_status(error.state["status"])
            checkpoint["status"] = error.state["status"]
            _save(checkpoint_path, checkpoint)
            error.state["attempts_reserved"] = len(checkpoint["attempts"])
            raise

        if len(checkpoint["columns"]) != NC:
            state = _record_refusal(checkpoint, checkpoint_path, directory,
                                    "columns_pending", receipt,
                                    attempt_status="budget_refused")
            raise BudgetRefusal("basis Hessian remains incomplete", state)
        if time.monotonic() >= attempt_deadline:
            state = _record_refusal(checkpoint, checkpoint_path, directory,
                                    "audit_pending", receipt,
                                    attempt_status="budget_refused")
            raise BudgetRefusal("26 basis columns are committed; 27th audit HVP pending", state)

        try:
            final = _build_final(checkpoint, directory, receipt, c, g)
        except NumericalRefusal as error:
            state = _record_refusal(checkpoint, checkpoint_path, directory,
                                    error.state["status"], receipt,
                                    attempt_status="numerical_refused", **{
                                        k: v for k, v in error.state.items()
                                        if k not in {"status", "completed_columns", "hvp_columns",
                                                     "hvp_calls_total", "attempts_reserved", "receipt",
                                                     "checkpoint_dir"}})
            raise NumericalRefusal(str(error), state) from error
        hessian = final["hessian_tensor"]
        eigenvalues = final["eigenvalues_tensor"]
        eigenvectors = final["eigenvectors_tensor"]
        audit_row = cast(dict[str, Any] | None, checkpoint.get("audit_hvp"))
        if audit_row is not None:
            direction = torch.tensor(audit_row["direction"], dtype=torch.float64)
            if (not torch.equal(direction, eigenvectors[:, 0])
                    or tensor_sha(direction) != audit_row["direction_sha256"]):
                raise ValueError("saved independent HVP direction differs from the fresh minimum eigenvector")
        else:
            if time.monotonic() >= attempt_deadline:
                state = _record_refusal(checkpoint, checkpoint_path, directory,
                                        "audit_pending", receipt,
                                        attempt_status="budget_refused")
                raise BudgetRefusal("deadline expired before independent 27th HVP", state)
            direction = eigenvectors[:, 0].contiguous()
            direction_sha = tensor_sha(direction)
            before = _receipt(receipt_provider, c, p, g)
            if before != receipt:
                state = _record_refusal(checkpoint, checkpoint_path, directory,
                                        "receipt_changed", receipt,
                                        attempt_status="receipt_refused", receipt_before=before)
                raise ReceiptRefusal("receipt changed before independent audit HVP", state)
            tensor_hashes = (tensor_sha(c), tensor_sha(p), tensor_sha(g))
            hvp_started = time.monotonic()
            direction_argument = direction.clone()
            product = _apply_hvp(objective, p, c, direction_argument)
            hvp_elapsed = time.monotonic() - hvp_started
            after = _receipt(receipt_provider, c, p, g)
            attempt_elapsed = time.monotonic() - attempt_start
            if ((tensor_sha(c), tensor_sha(p), tensor_sha(g)) != tensor_hashes
                    or tensor_sha(direction) != direction_sha
                    or tensor_sha(direction_argument) != direction_sha
                    or before != after or after != receipt):
                state = _record_refusal(checkpoint, checkpoint_path, directory,
                                        "receipt_changed", receipt, attempt_status="receipt_refused",
                                        receipt_before=before, receipt_after=after)
                raise ReceiptRefusal("receipt/vector changed across independent audit HVP", state)
            if (product.shape != (NC,) or product.dtype != torch.float64
                    or product.device.type != "cpu" or not bool(torch.isfinite(product).all())):
                state = _record_refusal(checkpoint, checkpoint_path, directory,
                                        "audit_hvp_invalid", receipt,
                                        attempt_status="numerical_refused")
                raise NumericalRefusal("independent 27th HVP is invalid/nonfinite", state)
            audit_row = {"direction": direction.tolist(), "direction_sha256": direction_sha,
                         "product": product.tolist(), "product_sha256": tensor_sha(product),
                         "receipt_before": before, "receipt_after": after,
                         "hvp_elapsed_seconds": hvp_elapsed,
                         "attempt_elapsed_seconds_at_commit": attempt_elapsed,
                         "status": "product_saved_validation_pending"}
            checkpoint["audit_hvp"] = audit_row
            checkpoint["status"] = "audit_validation_pending"
            _save(checkpoint_path, checkpoint)
            if time.monotonic() >= attempt_deadline:
                state = _record_refusal(checkpoint, checkpoint_path, directory,
                                        "audit_validation_pending", receipt,
                                        attempt_status="budget_refused")
                raise BudgetRefusal("deadline expired after saved independent 27th HVP", state)
        product = torch.tensor(audit_row["product"], dtype=torch.float64)
        eigenpair = blocks.eigenpair_audit(product, eigenvalues[0], eigenvectors[:, 0], final["norm"])
        if not eigenpair["passed"]:
            state = _record_refusal(checkpoint, checkpoint_path, directory,
                                    "eigenpair_refused", receipt, attempt_status="numerical_refused",
                                    minimum_eigenpair_audit=eigenpair)
            raise NumericalRefusal("independent 27th minimum-eigenpair HVP residual failed", state)
        audit_row["minimum_eigenpair_audit"] = eigenpair
        audit_row["status"] = "passed"
        checkpoint["audit_hvp"] = audit_row
        checkpoint["status"] = "schur_pending"
        _save(checkpoint_path, checkpoint)
        if time.monotonic() >= attempt_deadline:
            state = _record_refusal(checkpoint, checkpoint_path, directory,
                                    "schur_pending", receipt, attempt_status="budget_refused",
                                    minimum_eigenpair_audit=eigenpair)
            raise BudgetRefusal("deadline expired after independent HVP; Schur diagnostic pending", state)
        try:
            schur = blocks.schur_diagnostic(hessian, g)
        except blocks.DiagnosticRefusal as error:
            state = _record_refusal(checkpoint, checkpoint_path, directory,
                                    "schur_refused", receipt,
                                    attempt_status="numerical_refused", refusal=str(error))
            raise NumericalRefusal(f"block/Schur diagnostic refused: {error}", state) from error
        result = {"hessian": final["hessian"], "hessian_sha256": final["hessian_sha256"],
                  "hvp_columns": NC, "hvp_calls_total": NC + 1,
                  "symmetry_relative": final["symmetry_relative"],
                  "eigenvalues": final["eigenvalues"], "minimum_eigenpair_audit": eigenpair,
                  "block_schur": schur, "checkpoint_status": "completed",
                  "attempts_reserved": len(checkpoint["attempts"]), "receipt": receipt,
                  "checkpoint_dir": str(directory)}
        if time.monotonic() >= attempt_deadline:
            state = _record_refusal(checkpoint, checkpoint_path, directory,
                                    "schur_completed_over_deadline", receipt,
                                    attempt_status="budget_refused",
                                    block_schur=schur, minimum_eigenpair_audit=eigenpair)
            raise BudgetRefusal("deadline expired after Schur diagnostic", state)
        checkpoint["status"] = "completed"
        checkpoint["final_result"] = result
        checkpoint["attempts"][-1]["status"] = "completed"
        _save(checkpoint_path, checkpoint)
        attempt_active = False
        return result
    except Exception as error:
        if attempt_active and not isinstance(error, CheckpointRefusal):
            checkpoint["status"] = "programming_error"
            checkpoint["last_status"] = "programming_error"
            attempt = checkpoint["attempts"][-1]
            attempt["status"] = "programming_error"
            attempt["exception_type"] = type(error).__name__
            attempt["exception_message"] = str(error)
            checkpoint["last_refusal"] = _make_state(
                checkpoint, directory, "programming_error", receipt,
                exception_type=type(error).__name__, exception_message=str(error))
            _save(checkpoint_path, checkpoint)
        raise
    finally:
        lock.release()
