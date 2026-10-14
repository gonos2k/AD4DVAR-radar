from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012"
               / "qualify_candidate.py")
SPEC = importlib.util.spec_from_file_location("native_reference_qualification", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def _smooth_stage() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = torch.arange(4, dtype=torch.float64)[:, None]
    x = torch.arange(5, dtype=torch.float64)[None, :]
    q = 10.0 + 2.0 * y + y.square() + x.square() + 0.2 * x * y
    qx = torch.ones((4, 6), dtype=torch.float64)
    qy = -torch.ones((5, 5), dtype=torch.float64)
    return q, qx, qy


def _complete_one_stage(audit: object, q: torch.Tensor, qx: torch.Tensor,
                        qy: torch.Tensor) -> dict[str, object]:
    audit.stages = qualification.ANALYSIS_STAGES - 1
    audit(q, qx, qy)
    return audit.report()


def test_analysis_audit_reports_actual_regular_selectors_and_margins() -> None:
    q, qx, qy = _smooth_stage()
    report = _complete_one_stage(qualification.AnalysisBranchAudit(), q, qx, qy)

    assert report["classification"] == "smooth_regular"
    assert report["stage_count"] == qualification.ANALYSIS_STAGES
    assert len(report["actual_selector_signature_sha256"]) == 64
    assert report["selector_counts"]["x"]["left"] + report["selector_counts"]["x"]["right"] > 0
    assert report["normalized_branch_margins"]["x_operand_over_q_scale"]["value"] > 0


def test_analysis_audit_distinguishes_single_and_multiple_zero_faces() -> None:
    q, qx, qy = _smooth_stage()
    qx[1, 2] = 0.0
    single = _complete_one_stage(qualification.AnalysisBranchAudit(), q, qx, qy)
    assert single["classification"] == "singleface"
    assert single["unique_nearzero_face_entities"] == ["x:1:2"]

    qx[2, 4] = 0.0
    multiple = _complete_one_stage(qualification.AnalysisBranchAudit(), q, qx, qy)
    assert multiple["classification"] == "multiplefaces"
    assert multiple["unique_nearzero_face_entities"] == ["x:1:2", "x:2:4"]


def test_analysis_audit_retains_limiter_ties_as_unsupported_branch_data() -> None:
    q, qx, qy = _smooth_stage()
    q[1, 1] = 0.5 * (q[1, 0] + q[1, 2])
    report = _complete_one_stage(qualification.AnalysisBranchAudit(), q, qx, qy)

    assert report["classification"] in {"limiter_or_mixed_nonsmooth", "multiplefaces"}
    assert report["unique_near_tie_slope_entities"]
    assert report["strict_pointwise_branch"] is False

