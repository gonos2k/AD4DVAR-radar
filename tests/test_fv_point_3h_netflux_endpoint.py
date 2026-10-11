from __future__ import annotations

import copy
import json

import pytest

from examples.weather_scenarios import fv_point_3h_netflux_endpoint as endpoint


def _receipt_fixture():
    plan = endpoint._load_json(endpoint._read(endpoint.PLAN, endpoint.PLAN_SHA), "plan")
    manifest = endpoint._load_json(
        endpoint._read(endpoint.RECEIPT_MANIFEST, endpoint.RECEIPT_MANIFEST_SHA), "manifest")
    members = endpoint._bundle_members(
        endpoint._read(endpoint.RECEIPTS_TAR, endpoint.RECEIPTS_TAR_SHA), set(manifest["files"]))
    raw = json.loads(members["attempt1/step.json"])
    run = json.loads(members["attempt1/step.run.json"])
    resource = json.loads(members["attempt1/step.resource.json"])
    case_manifest = endpoint._load_json(endpoint._read(endpoint.CASE_MANIFEST), "case manifest")
    return plan, raw, run, resource, members, case_manifest


def test_endpoint_admission_is_receipt_only_and_returns_current_version_prefix(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("endpoint admission must not prepare seed data or evaluate FV")

    monkeypatch.setattr(endpoint.cycle.old.previous.seed, "_prepare_fixed_seed", forbidden)
    monkeypatch.setattr(endpoint.old.tangent, "_observe", forbidden)
    loaded = endpoint.load_endpoint()
    assert endpoint.tangent._tensor_sha(loaded["control"]) == endpoint.BASE_SHA
    assert loaded["raw"]["numerical_version"] == endpoint.NUMERICAL_VERSION
    assert loaded["raw"]["base_requalified"] is True
    assert loaded["raw"]["candidate_committed"] is True
    assert loaded["raw"]["optimizer_steps_applied"] == 1
    assert loaded["raw"]["current_control"] == loaded["raw"]["last_confirmed_control"]
    assert loaded["raw"]["current_full_G"] == loaded["raw"]["last_confirmed_full_G"]
    assert loaded["repeat"]["control"] == loaded["raw"]["current_control"]
    assert loaded["parent_plan"]["numerical_change"]["version"] == endpoint.NUMERICAL_VERSION
    assert set(loaded["receipt_files"]) == {
        path.relative_to(endpoint.ROOT).as_posix() for path in endpoint.RECEIPTS}


def test_endpoint_does_not_read_untracked_attempt_paths(monkeypatch):
    original_read = endpoint._read
    local_attempt = (endpoint.WORK / "attempt1").resolve()

    def guarded_read(path, expected_sha=None):
        if path.resolve().is_relative_to(local_attempt):
            raise AssertionError("endpoint must consume raw attempt members from receipts.tar.gz")
        return original_read(path, expected_sha)

    monkeypatch.setattr(endpoint, "_read", guarded_read)
    assert endpoint.load_endpoint()["raw"]["current_control_sha256"] == endpoint.BASE_SHA


def test_endpoint_checks_pinned_receipt_bundle_before_parsing(monkeypatch):
    monkeypatch.setattr(endpoint, "RECEIPTS_TAR_SHA", "0" * 64)
    with pytest.raises(ValueError, match="pinned receipt changed"):
        endpoint.load_endpoint()


def test_endpoint_rejects_a_broken_p2_prefix_without_fv():
    plan, raw, run, resource, members, case_manifest = _receipt_fixture()
    damaged = copy.deepcopy(raw)
    damaged["current_point"]["minimum_mixture_P2_match"]["passed"] = False
    with pytest.raises(ValueError, match="accepted-prefix/current-P2"):
        endpoint._validate_child(plan, damaged, run, resource, members, case_manifest)


@pytest.mark.parametrize("metric", ["J", "R"])
def test_endpoint_requires_final_repeat_to_satisfy_recorded_armijo_bound(metric):
    plan, raw, run, resource, members, case_manifest = _receipt_fixture()
    damaged = copy.deepcopy(raw)
    proposal = damaged["closed_steps"][0]["proposal"]
    repeat = damaged["closed_steps"][0]["final_repeat"]
    if metric == "J":
        proposal["J_armijo_passed"] = False
        proposal["J_armijo_bound"] = float(repeat["objective"]) - 1e-12
    else:
        proposal["F_squared_armijo_passed"] = False
        proposal["F_squared_armijo_bound"] = float(repeat["F_squared"]) - 1e-12
    with pytest.raises(ValueError, match="accepted candidate/repeat/full-G"):
        endpoint._validate_child(plan, damaged, run, resource, members, case_manifest)
