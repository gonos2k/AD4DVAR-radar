"""Render immutable archived FV response reports into the common envelope."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from examples.weather_scenarios.fv_response_contract import response_envelope

_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE = Path("graphify-out/fv-root-cause-20260919")
_ARCHIVE_SHA256 = {
    "point_single_qx_response_attempt2/response.json":
        "cb3283e0ee70518700a6ee5e2ade3f118683a42bbede26433e9e96aed231a59f",
    "partial_active_face_response_attempt1/response.json":
        "9095a44201a82765315fbac9bbb58783e415c4f50f373d75ba2de40eb9f6bfdc",
    "point_centered_response_attempt1/point_centered_response.json":
        "e0996a5081e20037d39f95049768d15d67e768b8aeda133a00fd638ff39e47aa",
}
_PROFILES = {
    "point_single_qx_response_attempt2/response.json": {
        "name": "point_single_qx_response_attempt2", "kind": "selected_face_conditional",
        "face": {"axis": "x", "row": 3, "column": 4, "value": 0.0},
        "chart": {"pivot_flow_index": 0, "ambient_dimension": 26, "tangent_dimension": 25},
        "direction_name": "middle_four_qx_face_plus1", "parameters_dimension": 13,
        "code_path": "examples/weather_scenarios/fv_point_single_qx_response_probe.py",
        "binding_path": "examples/weather_scenarios/fv_point_active_face_binding.py",
        "normal_path": "graphify-out/fv-root-cause-20260919/point_single_qx_normal_attempt2/normal.json",
        "gate_path": "examples/weather_scenarios/fv_point_single_qx_response_probe.py",
        "direction_scope": "four middle-time point-observation values in dBZ; tested finite endpoints only",
    },
    "partial_active_face_response_attempt1/response.json": {
        "name": "partial_active_face_response_attempt1", "kind": "selected_face_conditional",
        "face": {"axis": "y", "row": 2, "column": 0, "value": 0.0},
        "chart": {"pivot_flow_index": 1, "ambient_dimension": 26, "tangent_dimension": 25},
        "direction_name": "middle_time_common_bias_plus1", "parameters_dimension": 61,
        "code_path": "examples/weather_scenarios/fv_active_face_response_probe.py",
        "binding_path": "examples/weather_scenarios/fv_active_face_binding.py",
        "normal_path": "graphify-out/fv-root-cause-20260919/partial_active_face_normal_attempt3/normal.json",
        "gate_path": "examples/weather_scenarios/fv_active_face_stationarity_probe.py",
        "direction_scope": "19 middle-time common-bias parameters; tested finite endpoints only",
    },
    "point_centered_response_attempt1/point_centered_response.json": {
        "name": "point_centered_response_attempt1", "kind": "smooth_stationary",
        "face": None,
        "chart": {"ambient_dimension": 26, "tangent_dimension": 26},
        "direction_name": "single_archived_local_direction", "parameters_dimension": 13,
        "code_path": "fv_point_centered_prior_response_probe.py.txt",
        "gate_path": "fv_point_centered_prior_response_probe.py.txt",
        "direction_scope": "one local direction on the constructed centered-prior profile",
    },
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), _sha(raw)


def _archive_path(path: str | Path, root: Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = absolute.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("archive must be inside the repository root") from exc
    if not relative.startswith(_EVIDENCE.as_posix() + "/"):
        raise ValueError("unknown FV response archive")
    profile_key = relative[len(_EVIDENCE.as_posix()) + 1:]
    profile = _PROFILES.get(profile_key)
    if profile is None:
        raise ValueError("unknown FV response archive")
    return absolute, profile


def _source_records(archive_dir: Path, raw: dict[str, Any], profile: dict[str, Any],
                    root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    if profile["name"] == "point_centered_response_attempt1":
        manifest, _ = _read_json(archive_dir / "manifest.json")
        snapshots = {name: digest for name, digest in manifest["sha256"].items()
                     if name.endswith(".py.txt")}
        for name, digest in snapshots.items():
            if _sha((archive_dir / name).read_bytes()) != digest:
                raise ValueError(f"archived source snapshot hash mismatch: {name}")
        source_refs = {str(archive_dir.relative_to(root) / name): digest
                       for name, digest in snapshots.items()}
        probe_name = profile["code_path"]
        code_key = "fv_point_centered_prior_response_probe.py.txt"
        code_sha = snapshots[code_key]
        unchanged = raw.get("source_before") == raw.get("source_after")
    else:
        preflight, _ = _read_json(archive_dir / "preflight.json")
        source_refs = dict(preflight["source_sha256"])
        probe_name = profile["code_path"]
        code_sha = source_refs[probe_name]
        if profile["name"] == "point_single_qx_response_attempt2":
            parent, _ = _read_json(archive_dir / "parent.json")
            unchanged = (raw.get("source_before") == raw.get("source_after")
                         and parent.get("source_before") == parent.get("source_after"))
            if parent.get("source_before") != preflight["source_sha256"]:
                raise ValueError("archived source hashes disagree across preflight and parent")
            if any(key not in source_refs or source_refs[key] != digest
                   for key, digest in raw["source_before"].items()):
                raise ValueError("response source snapshot disagrees with archived preflight hashes")
        else:
            unchanged = (raw.get("original_source_before") == raw.get("original_source_after")
                         and raw.get("diagnostic_source_before") == raw.get("diagnostic_source_after"))
            raw_refs = {**raw["original_source_before"], **raw["diagnostic_source_before"]}
            overlap = raw_refs.keys() & source_refs.keys()
            if not overlap or any(raw_refs[key] != source_refs[key] for key in overlap):
                raise ValueError("archived preflight and before/after source hashes disagree")
            source_refs = {**raw_refs, **source_refs}
    if not unchanged:
        raise ValueError("archive does not establish unchanged source before/after")
    binding = profile.get("binding_path")
    if binding:
        if binding not in source_refs:
            raise ValueError("archived binding-source SHA reference is absent")
        profile["chart"]["lift_sha256"] = source_refs[binding]
        profile["chart"]["binding_source_path"] = binding
    profile["code_identity"] = {"path": probe_name, "sha256": code_sha}
    return source_refs, {"code": profile["code_identity"],
                         "source": {"sha256": _canonical_sha(source_refs), "references": source_refs,
                                    "before_after_equal": True}}


def _stationarity_gate(archive_dir: Path, raw: dict[str, Any], profile: dict[str, Any],
                       source_refs: dict[str, str]) -> dict[str, Any]:
    """Read the fixed gate from the archived source identity; never infer a new tolerance."""
    local_evidence_path: Path | None = None
    local_evidence_sha: str | None = None
    if profile["name"] == "point_centered_response_attempt1":
        manifest, _ = _read_json(archive_dir / "manifest.json")
        source_name = profile["gate_path"]
        snapshot_name = source_name
        gate_sha = manifest["sha256"][snapshot_name]
        source_text = (archive_dir / snapshot_name).read_text()
        local_evidence_path = _ROOT / _EVIDENCE / "FV_POINT_CENTERED_PRIOR_RESPONSE_EVIDENCE.json"
        local_evidence, local_evidence_sha = _read_json(local_evidence_path)
        local_source_name = "src/advar/local_response.py"
        local_sha = local_evidence["sha256"][local_source_name]
        source_ref_name = str(archive_dir.relative_to(_ROOT) / snapshot_name)
        local_ref_name = local_source_name
        expected_expression = "gradient_max >= 1e-10"
    elif profile["name"] == "point_single_qx_response_attempt2":
        source_name = profile["gate_path"]
        preflight, _ = _read_json(archive_dir / "preflight.json")
        source_text = preflight["source_snapshots"][source_name]
        gate_sha = hashlib.sha256(source_text.encode()).hexdigest()
        local_name = "src/advar/local_response.py"
        local_sha = source_refs[local_name]
        source_ref_name, local_ref_name = source_name, local_name
        expected_expression = "STATIONARITY_TOLERANCE = 1e-10"
        if source_text.count("gradient_max >= STATIONARITY_TOLERANCE") < 1:
            raise ValueError("archived qx gate does not show the fixed comparison")
    else:
        source_name = profile["gate_path"]
        source_text = (Path(_ROOT) / source_name).read_text()
        gate_sha = hashlib.sha256(source_text.encode()).hexdigest()
        if raw["diagnostic_source_before"].get(source_name) != gate_sha:
            raise ValueError("partial-face stationarity source differs from the archived source SHA")
        local_name = "src/advar/local_response.py"
        local_sha = source_refs[local_name]
        source_ref_name, local_ref_name = source_name, local_name
        expected_expression = "STATIONARITY_TOLERANCE = 1e-10"
        if source_text.count("fresh_gradient_max >= STATIONARITY_TOLERANCE") != 1:
            raise ValueError("archived partial-face gate does not show the fixed comparison")
    if gate_sha != source_refs.get(source_ref_name):
        raise ValueError("stationarity gate text does not match its archived source SHA")
    if (profile["name"] != "point_centered_response_attempt1"
            and local_ref_name not in source_refs):
        raise ValueError("local response source SHA is absent from archived source evidence")
    if profile["name"] == "point_centered_response_attempt1":
        gate_ok = expected_expression in source_text
    else:
        gate_ok = expected_expression in source_text
    if not gate_ok or not re.search(r"1e-10", source_text):
        raise ValueError("archived source does not establish the fixed 1e-10 stationarity gate")
    local_evidence_reference: dict[str, Any] = {}
    if local_evidence_path is not None and local_evidence_sha is not None:
        local_evidence_reference = {"evidence_record_path": str(local_evidence_path.relative_to(_ROOT)),
                                   "evidence_record_sha256": local_evidence_sha}
    return {"tolerance": 1e-10, "comparison": "pass iff gradient_max < tolerance",
            "objective_control_units": "archived profile units; no rescaling applied",
            "gate_source": {"path": source_ref_name, "sha256": gate_sha,
                            "evidence": "archived source snapshot" if profile["name"] != "partial_active_face_response_attempt1"
                            else "archived source-before SHA verified against matching source bytes"},
            "local_response_source": {"path": local_ref_name, "sha256": local_sha,
                                      "evidence": ("archived evidence source map" if profile["name"] == "point_centered_response_attempt1"
                                                   else "archived source SHA"),
                                      **local_evidence_reference}}


def _validation_rows(raw: dict[str, Any], profile: dict[str, Any], point_sha: str,
                     direction_sha: str) -> list[dict[str, Any]]:
    pairs = raw.get("score_finite_difference", raw.get("pairs", []))
    endpoints = raw.get("endpoints", [])
    rows = []
    for pair in pairs:
        h = float(pair.get("step", pair.get("h")))
        matched = [endpoint for endpoint in endpoints
                   if abs(float(endpoint.get("step", endpoint.get("h", 0.0)))) == h]
        rows.append({
            "control_sha256": point_sha, "direction_sha256": direction_sha,
            "scope": profile["direction_scope"], "status": "passed" if pair["passed"] else "failed",
            "method": "central_finite_difference",
            "step_sizes": [h],
            "endpoint_control_sha256": [row["control_sha256"] for row in matched],
            "endpoint_parameters_sha256": [row["parameters_sha256"] for row in matched
                                             if row.get("parameters_sha256")],
            "criterion": {"tolerance": pair.get("tolerance", pair.get("limit")),
                          "source_field": "archived tolerance/limit"},
            "error": {key: pair[key] for key in pair
                      if key in {"absolute_error", "relative_error", "central_score_difference",
                                 "central_difference", "predicted_response", "adjoint_directional"}},
        })
    return rows


def _response_payload(raw: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    name = profile["direction_name"]
    if profile["kind"] == "smooth_stationary":
        nominal, base = raw["nominal"], raw["response"]
        payload: dict[str, Any]
        names = ("direct", "indirect", "total")
        payload = {key: {name: base[key]} for key in names}
        for key in ("direct_gradient", "indirect_gradient", "total_gradient", "adjoint"):
            payload[key] = base[key]
        payload["mixed_gradients"] = {}
        payload.update({key: base[key] for key in ("pcg_iterations", "pcg_relative_residual",
                      "true_adjoint_relative_residual", "hvp_count") if key in base})
        payload.update(direction=nominal["direction"], direction_name=name,
                       total_projection=base.get("total_projection"),
                       archived_response_seconds=base.get("seconds"))
        return payload
    base = raw["response"] if "response" in raw else raw["base_response"]
    payload: dict[str, Any] = {key: {name: base[key]} for key in ("direct", "indirect", "total")}
    for key in ("direct_gradient", "indirect_gradient", "total_gradient", "score_control_gradient", "adjoint"):
        payload[key] = base[key]
    payload["mixed_gradients"] = {name: base.get("mixed_gradient", [])}
    payload.update({key: base[key] for key in ("gradient_max", "pcg_iterations", "pcg_relative_residual",
                  "true_adjoint_relative_residual", "true_adjoint_residual", "hvp_count") if key in base})
    payload.update(direction=raw["direction"], direction_name=name,
                   tangent_predictor=raw.get("tangent_predictor"), scope=base.get("scope"),
                   missing_parameter_gradient_triplets=base.get("missing_parameter_gradient_triplets"))
    return payload


def render_archived_response(path: str | Path, *, root: str | Path = _ROOT) -> dict[str, Any]:
    """Adapt one of the three registered immutable archives; performs no model work."""
    root_path = Path(root).resolve()
    archive_path, original_profile = _archive_path(path, root_path)
    profile = json.loads(json.dumps(original_profile))
    raw, archive_sha = _read_json(archive_path)
    profile_key = archive_path.resolve().relative_to(root_path).as_posix()[len(_EVIDENCE.as_posix()) + 1:]
    if archive_sha != _ARCHIVE_SHA256[profile_key]:
        raise ValueError("registered archived response SHA-256 mismatch")
    archive_dir = archive_path.parent
    source_refs, source_ids = _source_records(archive_dir, raw, profile, root_path)
    name = profile["name"]
    if name == "point_centered_response_attempt1":
        nominal = raw["nominal"]
        point_sha, parameter_sha = nominal["control_sha256"], nominal["parameters_sha256"]
        direction_sha = nominal["direction_sha256"]
        problem_sha = _canonical_sha(raw["input_before"]["problem"])
        branch_sha = nominal["branch"].get("signature_sha256", _canonical_sha(nominal["branch"]))
        prior_sha = nominal["dynamics_prior_mean_sha256"]
        run, run_sha = _read_json(archive_dir / "point_centered_response.run.json")
        resource, resource_sha = _read_json(archive_dir / "point_centered_response.resource.json")
        plan_sha = raw["plan_sha256"]
        runtime = raw["environment"]
        objective = {"sha256": _canonical_sha({"problem": problem_sha, "objective": "constructed centered dynamics prior"}),
                     "name": "constructed centered dynamics prior; full-control objective"}
        prior = {"sha256": prior_sha, "contract": "constructed centered dynamics prior; not original zero-prior profile",
                 "restored_full_objective": True, "control_dimension": 26,
                 "mean_dimension": len(nominal["dynamics_prior_mean"]),
                 "mean": nominal["dynamics_prior_mean"]}
        gradient_value, stationarity_scope = nominal["gradient_max"], "full_control"
        curvature = {"space": "full_control", "status": "reported archived Hessian audit",
                     **nominal["hessian_audit"]}
        branch = nominal["branch"]
        direction = nominal["direction"]
        validations = _validation_rows(raw, profile, point_sha, direction_sha)
        exec_status = "completed" if run.get("execution_status") == "completed" else run.get("execution_status", "not_recorded")
        numerical = ("eligible" if run.get("response_validation") == "passed"
                     and run.get("numerical_status") == "eligible" else "not_evaluated")
        resource_out = {**resource, "wrapper_status": exec_status, "wrapper_record_sha256": run_sha,
                        "resource_record_sha256": resource_sha, "source": "archived wrapper and resource records"}
        runtime_sha = _canonical_sha(runtime)
        coordinate = {**profile["chart"], "lift_sha256": _canonical_sha({"identity_chart": 26})}
    else:
        point_sha, parameter_sha = raw["base_control_sha256"], raw["parameters_sha256"]
        direction_sha = raw["direction_sha256"]
        ids_in = raw["input_identity"]
        problem_sha = (ids_in["problem_identity"]["fixed_problem_sha256"] if name == "point_single_qx_response_attempt2"
                       else ids_in["current_problem_identity"]["fixed_problem_sha256"])
        branch = raw["base_branch"]
        branch_sha = branch["signature_sha256"]
        prior_sha = _canonical_sha({"contract": "zero-centered full-control prior retained", "dimension": 26})
        plan_sha = raw["plan_sha256"]
        runfile = "parent.json" if name == "point_single_qx_response_attempt2" else None
        if runfile:
            run, run_sha = _read_json(archive_dir / runfile)
            exec_status = run.get("execution_status", "not_recorded")
            success_status = "restricted_active_face_response_numerically_supported_at_tested_points"
            numerical = "eligible" if run.get("response_validation") == success_status else "not_evaluated"
            runtime = raw.get("runtime", {})
        else:
            run_sha, exec_status = None, "not_recorded"
            success_status = "restricted_active_face_response_numerically_supported_at_tested_points"
            numerical = "eligible" if raw.get("response_validation") == success_status else "not_evaluated"
            runtime = {}
        resource, resource_sha = _read_json(archive_dir / "response.resource.json")
        resource_out = {**resource, "wrapper_status": exec_status,
                        "resource_record_sha256": resource_sha,
                        "wrapper_record_sha256": run_sha,
                        "wrapper_status_source": "archived parent wrapper" if runfile else "not_recorded; child record only"}
        runtime_sha = _canonical_sha(runtime if runtime else {"status": "not_recorded"})
        objective = {"sha256": _canonical_sha({"problem": problem_sha, "full_objective": "archived full-control objective"}),
                     "name": "original full-control objective with retained prior"}
        prior = {"sha256": prior_sha, "contract": "original full-control zero-centered prior restored by the archived chart binding",
                 "restored_full_objective": True, "control_dimension": 26}
        gradient_value, stationarity_scope = raw["base_gradient_max"], "tangent"
        curvature_raw = raw.get("base_curvature", raw.get("base_hessian", {}))
        curvature = {"space": "tangent", "status": "reported archived curvature audit",
                     **curvature_raw}
        validations = _validation_rows(raw, profile, point_sha, direction_sha)
        coordinate = {**profile["chart"], "lift_sha256": profile["chart"]["lift_sha256"]}
    identities = {
        "problem": {"sha256": problem_sha, "scope": "archived fixed input identity"},
        "control": {"sha256": point_sha, "dimension": profile["chart"]["ambient_dimension"]},
        "parameters": {"sha256": parameter_sha, "dimension": profile["parameters_dimension"]},
        "code": source_ids["code"],
        "branch": {"sha256": branch_sha},
        "source": source_ids["source"],
        "runtime": {"sha256": runtime_sha, "recorded": runtime or None},
        "plan": {"sha256": plan_sha},
    }
    gate = _stationarity_gate(archive_dir, raw, profile, source_refs)
    gate_passed = math.isfinite(float(gradient_value)) and float(gradient_value) < gate["tolerance"]
    payload = _response_payload(raw, profile)
    payload["archive"] = {"path": archive_path.relative_to(root_path).as_posix(), "sha256": archive_sha,
                          "source_artifact_sha256": source_refs}
    tangent_dimension = profile["chart"]["tangent_dimension"]
    normal_evidence = None
    if profile["kind"] == "selected_face_conditional":
        normal_block = raw.get("base_normal", raw.get("base_normal_recomputed"))
        if not isinstance(normal_block, dict):
            raise ValueError("response archive lacks its inline base normal block")
        normal_sha = _canonical_sha(normal_block)
        normal_evidence = {"status": "measured", "control_sha256": point_sha,
                           "scope": normal_block.get("scope", "inline normal at archived base point"),
                           "references": [{"sha256": normal_sha, "control_sha256": point_sha,
                                           "archive_sha256": archive_sha,
                                           "json_pointer": ("/base_normal" if "base_normal" in raw
                                                            else "/base_normal_recomputed"),
                                           "parameters_sha256": parameter_sha,
                                           "tangent_sha256": raw["base_tangent_sha256"],
                                           "scope": "inline normal block bound to exact archived base control/parameters/tangent"}]}
        historical_path = root_path / profile["normal_path"]
        historical_sha = _sha(historical_path.read_bytes())
        if historical_sha != raw["normal_result_sha256"]:
            raise ValueError("historical prequalification normal reference hash mismatch")
        payload["historical_prequalification_normal_reference"] = {
            "status": "historical_prequalification_only_not_bound_to_current_base_point",
            "path": profile["normal_path"], "sha256": historical_sha,
            "input_sha256": raw.get("normal_input_sha256"),
            "scope": "kept separate from the current inline base normal; no current control binding asserted",
        }
    output = response_envelope(
        kind=profile["kind"], execution_status=exec_status, numerical_status=numerical,
        physical_status="not_validated", identities=identities, objective_identity=objective,
        prior_identity=prior, coordinate_map=coordinate,
        stationarity={"space": stationarity_scope, "tolerance": gate["tolerance"], "value": gradient_value,
                      "status": "reported_pass_at_archived_fixed_gate" if gate_passed
                      else "reported_fail_at_archived_fixed_gate",
                      "criterion": gate},
        ambient_diagnostics={"status": raw.get("original_full26_gradient_gate_passed", False),
                             "gradient_max": raw.get("ambient_full_gradient_max_diagnostic"),
                             "scope": "reported only; not promoted to root/stationarity claim"},
        curvature=curvature, validations=validations, response=payload, resource=resource_out,
        claims={"full_root_supported": False, "physical_validation_supported": False},
        selected_face=profile["face"],
        chart=coordinate if profile["kind"] == "selected_face_conditional" else None,
        normal_evidence=normal_evidence,
    )
    if tangent_dimension != output["coordinates"]["tangent_dimension"]:
        raise AssertionError("profile coordinate mapping changed during adaptation")
    output["profile"] = {"name": name, "archive_sha256": archive_sha,
                         "source_before_after_equal": True,
                         "profile_scope": raw.get("scope"),
                         "provenance": "archived report adaptation; no FV recomputation"}
    return output
