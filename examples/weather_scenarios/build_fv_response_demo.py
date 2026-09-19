"""Embed actual shared-solver/FV/GN response fields in a standalone replay page."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy

import torch

from advar.fv_sensitivity import compute_fv_observation_response
from advar.physics import dbz_to_echo, echo_to_dbz
from advar.transport import finite_volume_trajectory
from advar.variational import forecast_fv_analysis, solve_analysis

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build_data():
    make_case = runpy.run_path(str(HERE / "fv_sensitivity_probe.py"))["make_case"]
    observations, frozen, future, support = make_case()
    config = replace(
        frozen.analysis_config,
        maximum_outer_iterations=16,
        maximum_pcg_iterations=96,
        gradient_tolerance=1e-10,
        step_tolerance=1e-12,
        pcg_relative_tolerance=1e-10,
    )
    frozen = replace(frozen, analysis_config=config)
    result = solve_analysis(observations, frozen)
    forecast = forecast_fv_analysis(
        result.control,
        frozen,
        leads=2,
        boundary_start_interval=2,
        boundary_echo=future,
        boundary_support=support,
    )
    spec = frozen.fv_transport
    assert spec is not None
    # Held-out synthetic truth only: these coefficients never enter solve_analysis.
    initial = dbz_to_echo(observations.dbz[0], min_dbz=-10.0)
    truth, _ = finite_volume_trajectory(
        initial,
        torch.ones_like(initial),
        spec.coefficient_limits * initial.new_tensor([0.2, -0.1, 0.15]).tanh(),
        initial.new_tensor(0.01),
        psi_basis=spec.psi_basis,
        leads=4,
        substeps_per_interval=2,
        interval_seconds=60.0,
        spacing_yx=spec.spacing_yx,
        boundary_echo=spec.boundary_echo + future,
        boundary_support=spec.boundary_support + support,
        reconstruction="donorcell",
    )
    truth_dbz = echo_to_dbz(truth, min_dbz=-10.0)
    forecast_dbz = echo_to_dbz(forecast.frames_linear, min_dbz=-10.0)
    verification = truth_dbz[3:]
    response = compute_fv_observation_response(
        result.control,
        observations,
        frozen,
        verification_dbz=verification,
        metric_weight=torch.ones_like(verification),
        leads=2,
        boundary_start_interval=2,
        boundary_echo=future,
        boundary_support=support,
        background_dependency="first_observation",
    )
    mae = (forecast_dbz[1:] - verification).abs().mean(dim=(-2, -1))
    persistence = (observations.dbz[-1] - verification).abs().mean(dim=(-2, -1))
    checks = []

    def check(label, passed, detail):
        checks.append(
            dict(label=label, status="pass" if passed else "fail", detail=detail)
        )

    check(
        "실제 분석 비용 감소",
        result.final_objective < result.initial_objective,
        f"{result.initial_objective:.6g} → {result.final_objective:.6g}; {result.outer_iterations}회 외부 반복",
    )
    check(
        "분석 정상성 수치 기준",
        response.gradient_max <= 1e-8,
        f"제어 기울기 최대 {response.gradient_max:.3e} ≤ 1e-8; 전역 정상성 인증은 아님",
    )
    check(
        "GN 수반 실제 잔차",
        response.adjoint_relative_residual <= 1e-10,
        f"{response.adjoint_relative_residual:.3e} ≤ 1e-10; 정상 연산 {response.normal_products}/128회",
    )
    check(
        "예측 에코 비음수",
        bool(
            torch.isfinite(forecast.frames_linear).all()
            & (forecast.frames_linear >= 0).all()
        ),
        "실제 FV 반환장의 모든 화소 확인",
    )
    check(
        "고정 영역 persistence 비교",
        bool((mae < persistence).all()),
        "+1·+2분 각각 20화소 전체에서 MAE 비교; 같은 모형의 합성 사례",
    )
    check(
        "관측 민감도 유한성",
        bool(torch.isfinite(response.sensitivity_dbz).all()),
        "B=y[0] 직접·암시적 경로 포함; IRLS–GN 역행렬 근사",
    )
    for label in (
        "일반 기상 시나리오의 성공",
        "정확한 implicit FSO/FSOI 제품 적격성",
        "전체 P1·FSO/FSOI D7",
        "FSO/FSOI 기반 지속 학습·재적재",
    ):
        checks.append(
            dict(
                label=label,
                status="unverified",
                detail="이번 데모의 검증 범위 밖 · 완료로 판정하지 않음",
            )
        )
    source_files = [
        Path("src/advar") / name
        for name in (
            "variational.py",
            "transport.py",
            "matrix_free.py",
            "physics.py",
            "fv_sensitivity.py",
        )
    ]
    source_files += [
        Path(__file__).resolve().relative_to(ROOT),
        (HERE / "fv_sensitivity_probe.py").relative_to(ROOT),
    ]
    hashes = {
        str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in source_files
    }
    return dict(
        scope="4×5 CPU FP64 · 같은 FV 모형의 합성 실증 · 알려진 미래 경계 · 실자료 성능 아님",
        grid=list(initial.shape),
        times_minutes=[-2, -1, 0, 1, 2],
        observations_dbz=observations.dbz.tolist(),
        forecast_dbz=forecast_dbz.tolist(),
        truth_dbz=truth_dbz.tolist(),
        persistence_dbz=observations.dbz[-1].tolist(),
        sensitivity_dbz=response.sensitivity_dbz.tolist(),
        checks=checks,
        metrics=dict(
            forecast_mae=mae.tolist(),
            persistence_mae=persistence.tolist(),
            gradient_max=response.gradient_max,
            adjoint_relative_residual=response.adjoint_relative_residual,
            normal_products=response.normal_products,
            curvature=response.curvature,
        ),
        source=dict(
            generated_at=datetime.now(timezone.utc).isoformat(),
            files=hashes,
            sha256=hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode()
            ).hexdigest(),
        ),
    )


def render(data, output):
    template = (HERE / "fv_response_template.html").read_text()
    if template.count("__FV_RESPONSE_DATA__") != 1:
        raise ValueError("template must contain one data placeholder")
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(template.replace("__FV_RESPONSE_DATA__", encoded))
    temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "fv_response.html")
    parser.add_argument("--data-output", type=Path, required=True)
    args = parser.parse_args()
    data = build_data()
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.data_output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    render(data, args.output)
    print(
        json.dumps(
            dict(html=str(args.output), checks=data["checks"], metrics=data["metrics"]),
            ensure_ascii=False,
            indent=2,
        )
    )
