"""Research-only common envelope and differentiable chart composition helpers."""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor
from advar.local_response import LocalResponse

ResponseKind = Literal["smooth_stationary", "selected_face_conditional"]
_EXECUTION = {"not_recorded", "not_started", "completed", "failed", "resource_limited", "refused"}
_NUMERICAL = {"not_evaluated", "eligible", "ineligible", "validation_failed"}
_PHYSICAL = {"not_validated", "validated", "validation_failed"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def compose_full_control(function: Callable[[Tensor, Tensor], Tensor],
                         lift: Callable[[Tensor, Tensor], Tensor]
                         ) -> Callable[[Tensor, Tensor], Tensor]:
    """Compose J(c,p) or E(c,p) with c=Γ(t,p), retaining autograd through Γ."""
    if not callable(function) or not callable(lift):
        raise TypeError("full-control function and chart lift must be callable")

    def composed(tangent: Tensor, parameters: Tensor) -> Tensor:
        full_control = lift(tangent, parameters)
        if not isinstance(full_control, Tensor):
            raise TypeError("chart lift must return a tensor")
        return function(full_control, parameters)

    return composed


def local_response_payload(result: LocalResponse) -> dict[str, Any]:
    """Serialize the common numeric and adjoint fields from ``LocalResponse``."""
    scalar_maps = ("direct", "indirect", "total")
    vector_maps = ("mixed_gradients",)
    payload: dict[str, Any] = {
        name: {key: float(value) for key, value in getattr(result, name).items()}
        for name in scalar_maps
    }
    payload.update({name: {key: value.tolist() for key, value in getattr(result, name).items()}
                    for name in vector_maps})
    for name in ("direct_gradient", "indirect_gradient", "total_gradient",
                 "score_control_gradient", "adjoint"):
        payload[name] = getattr(result, name).tolist()
    for name in ("gradient_max", "true_adjoint_residual", "true_adjoint_relative_residual",
                 "pcg_relative_residual"):
        payload[name] = float(getattr(result, name))
    for name in ("pcg_iterations", "hvp_count"):
        payload[name] = int(getattr(result, name))
    payload.update(branch_signature=result.branch_signature, scope=result.scope,
                   input_identity=dict(result.input_identity))
    return payload


def _identity(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not _SHA256.fullmatch(str(value.get("sha256", ""))):
        raise ValueError(f"{name} identity requires a lowercase SHA-256")
    return dict(value)


def _point_evidence(row: Mapping[str, Any], point_sha: str, parameters_sha: str) -> dict[str, Any]:
    evidence = _identity("normal evidence", row)
    if (evidence.get("control_sha256") != point_sha
            or evidence.get("parameters_sha256") != parameters_sha or not evidence.get("scope")):
        raise ValueError("normal evidence must identify the response control, parameters, and scope")
    return evidence


def _validation(row: Mapping[str, Any], point_sha: str, parameters_sha: str) -> dict[str, Any]:
    result = dict(row)
    status = result.get("status")
    if (result.get("control_sha256") != point_sha
            or not _SHA256.fullmatch(str(result.get("direction_sha256", "")))
            or not result.get("scope")
            or status not in {"not_run", "passed", "failed", "inconclusive"}):
        raise ValueError("validation requires a direction, point, scope, and explicit status")
    base_parameters_sha = result.get("parameters_sha256")
    if status != "not_run":
        if base_parameters_sha != parameters_sha:
            raise ValueError("performed validation must identify the response base parameters")
    elif base_parameters_sha is not None and base_parameters_sha != parameters_sha:
        raise ValueError("unrun validation parameter identity differs from the response base")
    steps = result.get("step_sizes", [])
    endpoints = result.get("endpoint_control_sha256", [])
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise ValueError("validation step sizes must be a sequence")
    if any(not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0 for step in steps):
        raise ValueError("validation step sizes must be positive and finite")
    if not isinstance(endpoints, Sequence) or isinstance(endpoints, (str, bytes)):
        raise ValueError("validation endpoint identities must be a sequence")
    if any(not _SHA256.fullmatch(str(value)) for value in endpoints):
        raise ValueError("validation endpoint identities must be SHA-256 values")
    if status != "not_run":
        if not steps or not endpoints:
            raise ValueError("performed validation requires step sizes and endpoint identities")
        if not isinstance(result.get("method"), str) or not result["method"]:
            raise ValueError("performed validation method must be explicit")
        if not isinstance(result.get("criterion"), Mapping) or not result["criterion"]:
            raise ValueError("performed validation criterion must be explicit")
        if not isinstance(result.get("error"), Mapping) or not result["error"]:
            raise ValueError("performed validation error evidence must be explicit")
        if result.get("method") == "central_finite_difference":
            parameter_endpoints = result.get("endpoint_parameters_sha256")
            expected = 2 * len(steps)
            if (len(endpoints) != expected
                    or not isinstance(parameter_endpoints, Sequence)
                    or isinstance(parameter_endpoints, (str, bytes))
                    or len(parameter_endpoints) != expected):
                raise ValueError("central finite differences require two control and parameter endpoints per step")
            if any(not _SHA256.fullmatch(str(value)) for value in parameter_endpoints):
                raise ValueError("central finite-difference parameter endpoint identities must be SHA-256 values")
    else:
        if (steps or endpoints or result.get("criterion") or result.get("error")):
            raise ValueError("not_run validation cannot carry observed steps, endpoints, criteria, or errors")
        result.setdefault("criterion", {})
        result.setdefault("error", {})
        result.setdefault("step_sizes", [])
        result.setdefault("endpoint_control_sha256", [])
    return result


def response_envelope(*, kind: ResponseKind, execution_status: str,
                      numerical_status: str, physical_status: str,
                      identities: Mapping[str, Mapping[str, Any]],
                      objective_identity: Mapping[str, Any],
                      prior_identity: Mapping[str, Any],
                      coordinate_map: Mapping[str, Any],
                      stationarity: Mapping[str, Any],
                      ambient_diagnostics: Mapping[str, Any],
                      curvature: Mapping[str, Any],
                      validations: Sequence[Mapping[str, Any]],
                      response: Mapping[str, Any], resource: Mapping[str, Any],
                      claims: Mapping[str, bool],
                      selected_face: Mapping[str, Any] | None = None,
                      chart: Mapping[str, Any] | None = None,
                      normal_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build an explicit-status response envelope; it never infers qualification."""
    if kind not in ("smooth_stationary", "selected_face_conditional"):
        raise ValueError("unsupported response kind")
    if execution_status not in _EXECUTION or numerical_status not in _NUMERICAL or physical_status not in _PHYSICAL:
        raise ValueError("execution, numerical, and physical statuses must be explicit supported values")
    required = {"problem", "control", "parameters", "code", "branch", "source", "runtime", "plan"}
    if not isinstance(identities, Mapping) or not required <= set(identities):
        raise ValueError("identities must include problem/control/parameters/code/branch/source/runtime/plan")
    ids = {key: _identity(key, value) for key, value in identities.items()}
    point_sha = str(ids["control"]["sha256"])
    objective = _identity("objective", objective_identity)
    prior = dict(prior_identity)
    if (not _SHA256.fullmatch(str(prior.get("sha256", "")))
            or not prior.get("contract")
            or prior.get("restored_full_objective") is not True
            or type(prior.get("control_dimension")) is not int or prior["control_dimension"] < 1):
        raise ValueError("the original full-control prior contract/hash must be explicit and restored")
    coordinates = dict(coordinate_map)
    if (not coordinates or type(coordinates.get("ambient_dimension")) is not int
            or type(coordinates.get("tangent_dimension")) is not int
            or not _SHA256.fullmatch(str(coordinates.get("lift_sha256", "")))):
        raise ValueError("coordinate map requires ambient/tangent dimensions and a lift identity")
    if (coordinates["ambient_dimension"] != prior["control_dimension"]
            or not 0 < coordinates["tangent_dimension"] <= coordinates["ambient_dimension"]):
        raise ValueError("coordinate dimensions must match the original prior and define a nonempty tangent")
    if not isinstance(stationarity, Mapping) or not {"space", "tolerance", "value", "status"} <= set(stationarity):
        raise ValueError("stationarity space, tolerance, value, and status must be explicit")
    if not isinstance(ambient_diagnostics, Mapping) or not ambient_diagnostics:
        raise ValueError("ambient stationarity diagnostics must be separately recorded")
    if not isinstance(curvature, Mapping) or not {"space", "status"} <= set(curvature):
        raise ValueError("curvature scope and status must be explicit")
    if kind == "smooth_stationary":
        if (coordinates["tangent_dimension"] != coordinates["ambient_dimension"]
                or stationarity["space"] != "full_control" or curvature["space"] != "full_control"):
            raise ValueError("smooth stationary response requires full-control stationarity and curvature")
    elif (coordinates["tangent_dimension"] >= coordinates["ambient_dimension"]
          or stationarity["space"] != "tangent" or curvature["space"] != "tangent"):
        raise ValueError("selected-face response requires lower-dimensional tangent stationarity and curvature")
    if not isinstance(response, Mapping) or not response:
        raise ValueError("LocalResponse numeric payload must be explicit")
    if not isinstance(resource, Mapping) or not resource:
        raise ValueError("execution resource evidence must be explicit")
    if (not isinstance(claims, Mapping) or not {"full_root_supported", "physical_validation_supported"} <= set(claims)
            or any(not isinstance(value, bool) for value in claims.values())):
        raise ValueError("root/physical claims must be explicit booleans; no claims are inferred")
    if claims["physical_validation_supported"] and physical_status != "validated":
        raise ValueError("physical support cannot be claimed without an explicit validated status")
    if kind == "selected_face_conditional":
        if claims["full_root_supported"]:
            raise ValueError("selected-face conditional response cannot claim full-root support")
        if not isinstance(selected_face, Mapping) or not selected_face or not isinstance(chart, Mapping) or not chart:
            raise ValueError("selected-face response requires its face and chart identities")
        if (not isinstance(normal_evidence, Mapping)
                or normal_evidence.get("status") not in {"not_run", "measured", "refused"}
                or normal_evidence.get("control_sha256") != point_sha
                or not normal_evidence.get("scope")):
            raise ValueError("selected-face response requires an explicit normal-evidence status and scope")
        normal_parameters_sha = normal_evidence.get("parameters_sha256")
        if normal_evidence["status"] == "measured":
            if normal_parameters_sha != ids["parameters"]["sha256"]:
                raise ValueError("measured normal evidence must identify the response base parameters")
        elif normal_parameters_sha is not None and normal_parameters_sha != ids["parameters"]["sha256"]:
            raise ValueError("unmeasured normal evidence parameter identity differs from the response base")
        if normal_evidence["status"] in {"not_run", "refused"}:
            allowed = {"status", "control_sha256", "parameters_sha256", "scope", "references", "plan"}
            if set(normal_evidence) - allowed:
                raise ValueError("unmeasured normal evidence cannot carry inline measured values")
        refs = normal_evidence.get("references")
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            raise ValueError("normal evidence references must be an explicit sequence")
        if normal_evidence["status"] == "measured" and not refs:
            raise ValueError("measured normal evidence requires finite-point references")
        if normal_evidence["status"] != "measured" and refs:
            raise ValueError("unmeasured normal evidence cannot carry result references")
        checked_refs = []
        for item in refs:
            if not isinstance(item, Mapping):
                raise ValueError("normal evidence entries must be mappings")
            checked_refs.append(_point_evidence(item, point_sha, str(ids["parameters"]["sha256"])))
        geometry = {"selected_face": dict(selected_face), "chart": dict(chart),
                    "normal_evidence": {**dict(normal_evidence),
                        "references": checked_refs}}
    else:
        if selected_face is not None or chart is not None or normal_evidence is not None:
            raise ValueError("smooth full-control response cannot carry selected-face geometry")
        geometry = {"selected_face": None, "chart": None, "normal_evidence": None}
    if not isinstance(validations, Sequence) or not validations:
        raise ValueError("at least one independent direction/step validation record is required")
    checked_validations = [_validation(row, point_sha, str(ids["parameters"]["sha256"])) for row in validations]
    return {
        "schema": "advar.response-envelope.v1",
        "kind": kind,
        "assessment": {"execution": execution_status, "numerical": numerical_status,
                       "physical": physical_status},
        "identities": ids,
        "objective": objective,
        "prior": prior,
        "coordinates": coordinates,
        "stationarity": dict(stationarity),
        "ambient_diagnostics": dict(ambient_diagnostics),
        "curvature": dict(curvature),
        "geometry": geometry,
        "independent_validation": checked_validations,
        "response": dict(response),
        "resource": dict(resource),
        "claims": dict(claims),
    }
