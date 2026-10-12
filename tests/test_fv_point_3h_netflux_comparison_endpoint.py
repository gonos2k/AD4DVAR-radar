from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "graphify-out/fv-root-cause-20260919/netflux_subspace_20261012"
PLAN = WORK / "PLAN.json"
PLAN_SHA = "a0fef5718772032b4567d24ca75fbaf39cff8b58e30f7e4b249dbabb4a5c7d26"
SOURCE_TAR = WORK / "source.tar.gz"
SOURCE_TAR_SHA = "b332268a74f0c5ed9f9e05b3d890288d2e953215b1422253eec2426cb183cab3"
RECEIPT_MANIFEST_SHA = "879b35fd94d9e77036badf3bad47dba8f4885794a3a4841206bc3be3104a960c"
RECEIPTS_TAR_SHA = "cb05abcc41fd0b3f1a3cb86208caec566eb3fb5f363014094fd3b6c9a3f7bfc0"
PR301_SOURCE_COMMIT = "5e23be9"
SOURCE_PATH = "graphify-out/fv-root-cause-20260919/netflux_subspace_20261012/source.tar.gz"
RECEIPT_MANIFEST_PATH = "graphify-out/fv-root-cause-20260919/netflux_subspace_20261012/RECEIPT_MANIFEST.json"
RECEIPTS_TAR_PATH = "graphify-out/fv-root-cause-20260919/netflux_subspace_20261012/receipts.tar.gz"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _restore_pinned_tree(destination: Path) -> None:
    """Restore PR301 source and archived inputs from their actual pinned bytes."""
    plan_bytes = PLAN.read_bytes()
    assert _sha(plan_bytes) == PLAN_SHA
    assert (WORK / "PLAN_SHA256.txt").read_text().strip() == PLAN_SHA
    plan = json.loads(plan_bytes)

    destination.mkdir()
    # The bundle pins changed sources; its historical import closure also needs
    # unchanged project modules, restored from the PR301 merge tree first.
    source_tree = subprocess.run(
        ["git", "archive", "--format=tar", PR301_SOURCE_COMMIT],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(source_tree), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())

    source_bytes = SOURCE_TAR.read_bytes()
    assert _sha(source_bytes) == SOURCE_TAR_SHA
    with tarfile.open(fileobj=io.BytesIO(source_bytes), mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        assert set(members) == set(plan["source_files"])
        for name, expected in plan["source_files"].items():
            data = archive.extractfile(members[name]).read()
            assert _sha(data) == expected, name
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    for name, expected in plan["archive_files"].items():
        data = (ROOT / name).read_bytes()
        assert _sha(data) == expected, name
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    # These admission inputs are pinned alongside, rather than inside, the tarballs.
    for relative in (
        "graphify-out/fv-root-cause-20260919/netflux_subspace_20261012/PLAN.json",
        "graphify-out/fv-root-cause-20260919/netflux_subspace_20261012/PLAN_SHA256.txt",
        RECEIPT_MANIFEST_PATH,
        SOURCE_PATH,
        RECEIPTS_TAR_PATH,
    ):
        source = ROOT / relative
        expected = {
            "graphify-out/fv-root-cause-20260919/netflux_subspace_20261012/PLAN.json": PLAN_SHA,
            RECEIPT_MANIFEST_PATH: RECEIPT_MANIFEST_SHA,
            SOURCE_PATH: SOURCE_TAR_SHA,
            RECEIPTS_TAR_PATH: RECEIPTS_TAR_SHA,
        }.get(relative)
        if expected is not None:
            assert _sha(source.read_bytes()) == expected
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


_ADMISSION_SCRIPT = r'''
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
sys.path[:0] = [str(root / "src"), str(root)]
from examples.weather_scenarios import fv_point_3h_netflux_comparison_endpoint as endpoint

def forbidden(*args, **kwargs):
    raise AssertionError("receipt-only endpoint invoked production evaluation")

endpoint.comparison.CurrentBackend = forbidden
endpoint.comparison.tangent._observe = forbidden
if mode == "admission":
    admitted = endpoint.load_endpoint()
    assert endpoint.comparison.tangent._tensor_sha(admitted["control"]) == endpoint.BASE_SHA
    assert admitted["full_G"].shape == (27,)
    assert admitted["repeat"]["control"] == admitted["control"].tolist()
    assert admitted["theta"] == admitted["repeat"]["theta"]
    assert admitted["input_after"]["control_sha256"] == endpoint.BASE_SHA
    assert admitted["parent_plan"]["experiment_kind"] == "same_point_fixed_version_gradient_robust_GN"
    assert admitted["receipt_files"][endpoint.PLAN.relative_to(endpoint.ROOT).as_posix()] == endpoint.PLAN_SHA
elif mode == "tamper":
    original = endpoint._read
    def changed_manifest(path, digest=None):
        if path == endpoint.RECEIPT_MANIFEST:
            return b"{}"
        return original(path, digest)
    endpoint._read = changed_manifest
    try:
        endpoint.load_endpoint()
    except ValueError as error:
        assert re.search("manifest|pinned", str(error))
    else:
        raise AssertionError("endpoint accepted a changed receipt manifest")
else:
    raise AssertionError("unknown historical endpoint test")
'''


@pytest.fixture(scope="module")
def historical_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pr301-source") / "tree"
    _restore_pinned_tree(root)
    return root


def _run_historical_admission(root: Path, mode: str) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", _ADMISSION_SCRIPT, str(root), mode],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_receipt_endpoint_admits_d19e_without_model_evaluation(historical_root: Path) -> None:
    _run_historical_admission(historical_root, "admission")


def test_endpoint_rejects_a_changed_receipt_manifest(historical_root: Path) -> None:
    _run_historical_admission(historical_root, "tamper")
