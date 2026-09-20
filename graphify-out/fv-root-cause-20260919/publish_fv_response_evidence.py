"""Publish saved 240x240 FV response evidence without changing demo arrays.

This publisher is deliberately fail-closed: it requires the completed ``--impacts``
report before touching either HTML file.  The sensitivity image is rendered from
the saved response tensor, not recomputed here.
"""
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import re
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[2]
# Source hashes independently checked against c192c0df (the measured revision).
# Pin the fingerprint so offline/shallow checkouts can validate archived reports.
TRANSPORT_SOURCE_FINGERPRINT = "e619a8bf0fdf4d8da492ca4beec84ede1cb06242ec0aa92ce998a91011889cfb"
MEASURED_REVISION = "ce6e36a2dcfb2f823647e7cabb5369dca3c9f200"
HERE = Path(__file__).resolve().parent
REPORT_PATH = HERE / "rotation240_stable_response_18.json"
RESPONSE_PATH = HERE / "rotation240_stable_response_18.pt"
CENTRAL_PATH = HERE / "rotation240_stable_central_0.0005.json"
LONG_HORIZON_PATH = HERE / "fv_grid_convergence_180min.json"
MAP_PATHS = tuple(HERE / f"rotation240_stable_response_18_sensitivity_{i}.svg" for i in range(3))


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_final_report():
    if not REPORT_PATH.is_file() or not RESPONSE_PATH.is_file():
        raise SystemExit("final response JSON/PT are required; run fv_large_response_probe.py --impacts first")
    report = json.loads(REPORT_PATH.read_text())
    if hashlib.sha256(RESPONSE_PATH.read_bytes()).hexdigest() != report["response_tensor_sha256"]:
        raise SystemExit("response tensor differs from the measured artifact")
    if hashlib.sha256((HERE / "rotation240_stable_refined.pt").read_bytes()).hexdigest() != report["refined_checkpoint_sha256"]:
        raise SystemExit("refined checkpoint differs from the measured artifact")
    for name, expected in report["source_hashes"].items():
        measured_source = subprocess.check_output(
            ["git", "show", f"{MEASURED_REVISION}:{name}"], cwd=ROOT
        )
        if hashlib.sha256(measured_source).hexdigest() != expected:
            raise SystemExit(f"measured revision does not match response source: {name}")
    if report.get("leads") != 18 or report.get("curvature") != "exact_robust_hessian":
        raise SystemExit("response report is not the requested 18-lead exact-Hessian result")
    impacts = report.get("impacts")
    if not isinstance(impacts, list) or len(impacts) < 2:
        raise SystemExit("response report is preliminary; at least two final Taylor impacts are required")
    required = ("step", "actual_change", "linear_prediction", "taylor_error", "gradient_norm", "gradient_max", "face_margin")
    for impact in impacts:
        if any(not _finite(impact.get(key)) for key in required):
            raise SystemExit("response report has incomplete or nonfinite impact evidence")
    for key in ("gradient_max", "adjoint_relative_residual", "face_margin", "score", "sensitivity_norm"):
        if not _finite(report.get(key)):
            raise SystemExit(f"response report has no finite {key}")
    return report


def _load_optional_central(report):
    """Load only a central check bound to the same measured response."""
    if not CENTRAL_PATH.is_file():
        return None
    central = json.loads(CENTRAL_PATH.read_text())
    if central.get("source_hashes") != report.get("source_hashes"):
        raise SystemExit("central check source hashes differ from the primary response")
    for name in ("response_tensor_sha256", "refined_checkpoint_sha256"):
        if central.get(name) != report.get(name):
            raise SystemExit(f"central check {name} differs from the primary response")
    required = ("directional_slope", "central_slope")
    if any(not _finite(central.get(name)) for name in required):
        raise SystemExit("central check has incomplete or nonfinite slope evidence")
    for side in ("positive", "negative"):
        for name in ("actual_change", "linear_prediction", "taylor_error", "gradient_max", "face_margin"):
            if not _finite(central.get(side, {}).get(name)):
                raise SystemExit(f"central check has no finite {side} {name}")
    if not _finite(central.get("step")) or central["step"] <= 0:
        raise SystemExit("central check requires a positive step")
    adjoint = float(central["directional_slope"])
    central_slope = float(central["central_slope"])
    relative_error = (
        abs(central_slope - adjoint) / abs(adjoint)
        if adjoint
        else None
    )
    return {
        "adjoint_slope": adjoint,
        "central_slope": central_slope,
        "relative_error": relative_error,
        "positive_gradient_max": float(central["positive"]["gradient_max"]),
        "negative_gradient_max": float(central["negative"]["gradient_max"]),
        "negative_impact": {**central["negative"], "step": -central["step"]},
    }


def _validate_transport_comparison(reports):
    """Validate this saved comparison before assigning scheme labels in HTML."""
    cases = {"translation", "rotation", "area_preserving_strain"}
    sizes = [32, 64, 128]
    expected_rows = {(case, size) for case in cases for size in sizes}
    baseline = reports["donorcell"]
    for scheme, data in reports.items():
        expected = {
            "status": "complete", "scheme": f"current {scheme} SSPRK2",
            "domain_side_m": 48000.0, "interval_seconds": 600.0,
            "leads": 18, "sizes": sizes, "max_courant": 0.5,
            "device": "cpu", "dtype": "float64",
            "boundary_condition": "known-zero exterior with complete known initial field",
        }
        if any(data.get(key) != value for key, value in expected.items()):
            raise SystemExit("transport comparison scheme/domain/time/boundary mismatch")
        if len(data.get("cases", [])) != 3 or set(data["cases"]) != cases:
            raise SystemExit("transport comparison must contain the three unique cases")
        for key in ("source_sha256", "oracle", "shape_policy", "python", "torch", "device", "dtype"):
            if not data.get(key) or data[key] != baseline.get(key):
                raise SystemExit(f"transport comparison has different {key}")
        rows = data.get("results", [])
        keys = [(row["case"], row["size"]) for row in rows]
        if len(keys) != len(expected_rows) or set(keys) != expected_rows:
            raise SystemExit("transport comparison has missing or duplicate case/grid rows")
        for row in rows:
            if (row.get("reconstruction") != scheme
                    or row.get("spacing_m") != 48000 / row["size"]
                    or row.get("echo_area_threshold") != 0.1
                    or [lead.get("lead_minutes") for lead in row["leads"]] != list(range(10, 181, 10))):
                raise SystemExit("transport comparison row scheme/grid/time/threshold mismatch")
            if (type(row.get("substeps_per_lead")) is not int or row["substeps_per_lead"] <= 0
                    or not _finite(row.get("actual_max_cfl")) or not 0 <= row["actual_max_cfl"] <= 0.5):
                raise SystemExit("transport comparison has an invalid CFL schedule")
            final = row["leads"][-1]
            values = (final["relative_echo_l2"], row["derivative"]["jvp_relative_to_independent_oracle"],
                      final["predicted_shape"]["maximum_echo"], final["reference_shape"]["maximum_echo"])
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values) or values[-1] == 0:
                raise SystemExit("transport comparison has invalid displayed metrics")
    donor_rows = {(r["case"], r["size"]): r for r in baseline["results"]}
    for row in reports["minmod"]["results"]:
        donor = donor_rows[(row["case"], row["size"])]
        if any(row.get(key) != donor.get(key) for key in ("substeps_per_lead", "actual_max_cfl")):
            raise SystemExit("transport comparison uses different CFL schedules")
    # These archived runs predate tensor input hashes. Bind their common
    # deterministic initial/flow/boundary generators to the measured revision.
    sources = baseline["source_sha256"]
    hashes = {Path(name).name: value for name, value in sources.items()}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if len(hashes) != len(sources) or fingerprint != TRANSPORT_SOURCE_FINGERPRINT:
        raise SystemExit("transport comparison sources do not match the measured revision")


def _validate_joint_inverse(joint):
    """Validate the archived 3x3 result, not a newly generated inverse report.

    Canonical report identity is from PR165/855aab5, including its producer
    hashes. It does not assert that today's producer is the measured source.
    """
    expected_rows = {(direction, h) for direction in ("observation", "background_parameter")
                     for h in (.001, .0005)}
    if (joint["controls"] != 12 or joint["substeps_per_interval"] != 9
            or joint["stationary_branch"]["euler_stages"] != 54
            or joint["general_minmod_response_eligible"] is not False
            or joint["finite_path_certified"] is not False
            or len(joint["reanalysis"]) != 4
            or {(r["direction"], r["h"]) for r in joint["reanalysis"]} != expected_rows):
        raise SystemExit("joint minmod report scope or reanalysis mismatch")
    # The oracle uses 1e-10 stationarity; require a resolved adjoint and strict
    # positive curvature/limiter margins before publishing this local result.
    for name in ("gradient_max", "adjoint_relative_residual"):
        value = joint.get(name)
        if not _finite(value) or not 0 <= value <= 1e-10:
            raise SystemExit(f"joint minmod invalid {name}")
    eigenvalue = joint.get("hessian_min_eigenvalue")
    margin = joint["stationary_branch"].get("minimum_scaled_slope_margin")
    if not _finite(eigenvalue) or eigenvalue <= 0:
        raise SystemExit("joint minmod invalid Hessian curvature")
    if not _finite(margin) or margin <= 128*math.ulp(1.0):
        raise SystemExit("joint minmod invalid branch margin")
    for row in joint["reanalysis"]:
        adjoint, central, error = (row.get(k) for k in
                                  ("adjoint", "central_reanalysis", "absolute_error"))
        if not all(_finite(v) for v in (adjoint, central, error)) or error < 0:
            raise SystemExit("joint minmod invalid reanalysis numbers")
        expected = abs(adjoint-central)
        if not math.isclose(error, expected, rel_tol=1e-12,
                            abs_tol=8*math.ulp(max(abs(adjoint), abs(central)))):
            raise SystemExit("joint minmod inconsistent absolute_error")
    fingerprint = hashlib.sha256(json.dumps(joint, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    if fingerprint != "eb1b676c9b80996d197ca344d3ff9a23d9aa7b83919c1c81172ac15f7a87d950":
        raise SystemExit("joint minmod archived report/source identity mismatch")


def _local_path_panel():
    path = HERE / "minmod_local_path.json"
    if not path.is_file():
        return ""
    data = json.loads(path.read_text())
    pairs = data["pairs"][-2:]
    if (data["status"] != "complete" or len(pairs) != 2
            or data["finite_path_certified"] is not False
            or data["general_minmod_response_eligible"] is not False
            or pairs[1]["j"] != pairs[0]["j"] + 1):
        raise SystemExit("local path scope or completion mismatch")
    for pair in pairs:
        if not pair["derivative_pass"] or not pair["predictors_ok"]:
            raise SystemExit("local path derivative/branch mismatch")
        for endpoint in pair["endpoints"]:
            g = endpoint["gradient_max"]
            if not _finite(g) or not 0 <= g < 1e-10 or not endpoint["same_local_branch"]:
                raise SystemExit("local path endpoint is not stationary on the branch")
    # This panel publishes one archived measurement, including source/input IDs.
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    if fingerprint != "db62cd14daa55af1e8a86b51a841eedb9535887fac3f81396ade48ac1e9cd998":
        raise SystemExit("local path archived identity mismatch")
    rows = "".join(
        f'<tr><td>{p["h"]:.6g}</td><td>{p["response_sensitivity"]:.8e}</td>'
        f'<td>{p["central_reanalysis"]:.8e}</td>'
        f'<td>{p["relative_error_adjoint_denominator"]:.4%}</td></tr>' for p in pairs)
    return (
        '<section id="fvMinmodLocalPath"><h3>4×5 minmod: 같은 국소 분기의 관측 응답</h3>'
        '<p>저장된 26제어 정상점에서 Hessian 기반 예측값을 보정했습니다. '
        '동일한 raw sin(k), k=0…59 관측 방향·고정 배경 파라미터로 양·음 재분석을 수행했습니다. '
        '최대 gradient &lt; 1e-10, 54개 RK 단계의 셀별 limiter·면별 유량 부호를 대조했습니다.</p>'
        '<div style="max-width:100%;overflow-x:auto"><table style="border-spacing:12px 6px;white-space:nowrap">'
        '<thead><tr><th>h (dBZ)</th><th>수반 방향미분</th><th>중앙 재분석 차분</th><th>수반 대비 차이</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
        '<p>기존 h=0.001의 유한 영향 검증은 여전히 미완료입니다. '
        '이 결과는 합성 조건부 점수의 국소 미분 검증이며, 전체 경로 인증·일반 minmod FSOI·'
        '신경망 학습 개선·독립 예측 성능을 뜻하지 않습니다. 기존 애니메이션은 유지합니다.</p>'
        '<p><a href="../../graphify-out/fv-root-cause-20260919/minmod_local_path.json">원시 결과·Hessian·경로 예측값</a> · '
        '<a href="../../graphify-out/fv-root-cause-20260919/MINMOD_LOCAL_PATH_REVIEW.md">범위·실행 기록</a></p></section>'
    )


def _validate_direction_result(data, direction, fingerprint):
    pairs = data["pairs"][-2:]
    if (data["status"] != "complete" or data["selected_direction"] != direction
            or data["finite_path_certified"] is not False
            or data["general_minmod_response_eligible"] is not False
            or len(pairs) != 2 or pairs[1]["j"] != pairs[0]["j"] + 1):
        raise SystemExit("direction result scope/completion mismatch")
    for pair in pairs:
        s, fd, error = (pair[k] for k in
                        ("response_sensitivity", "central_reanalysis", "absolute_error"))
        if (not all(_finite(x) for x in (s, fd, error, pair["h"]))
                or not s or pair["h"] != 1e-3*2.0**(-pair["j"]) or error < 0
                or not math.isclose(error, abs(s-fd), rel_tol=1e-12,
                                    abs_tol=8*math.ulp(max(abs(s), abs(fd))))
                or error > 1e-4*abs(s) or not pair["derivative_pass"]
                or not pair["predictors_ok"]
                or [e["sign"] for e in pair["endpoints"]] != [-1, 1]):
            raise SystemExit("direction result derivative evidence mismatch")
        endpoints = pair["endpoints"]
        if not math.isclose(fd, (endpoints[1]["score"]-endpoints[0]["score"])/(2*pair["h"]),
                            rel_tol=1e-12, abs_tol=8*math.ulp(abs(fd))):
            raise SystemExit("direction result endpoint score mismatch")
        for endpoint in endpoints:
            g = endpoint["gradient_max"]
            branch = endpoint["branch"]
            nominal = data["nominal_branch"]
            if (not _finite(g) or not 0 <= g < 1e-10
                    or not endpoint["same_local_branch"]
                    or branch["choices"] != nominal["choices"]
                    or branch["face_signs"] != nominal["face_signs"]):
                raise SystemExit("direction result endpoint branch/stationarity mismatch")
    actual = hashlib.sha256(json.dumps(data, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    if actual != fingerprint:
        raise SystemExit("direction result archived identity mismatch")


def _additional_directions_panel():
    # Fingerprints bind the displayed numbers to completed measured reports.
    measured = [('theta', '배경 θ', '9f01752de68a609e8497b5da59a96b6ea9fd90b79ed61235000b1dae7c1f7978'), ('middle_time_bias', '중간 시각 공통 편향', 'faba5a1846e963e6b3d2e6bd9d516320b34b1b60e15e66d2ac0ce9d58166f8a9')]
    if not any((HERE / f"minmod_{name}_final.json").exists() for name, _, _ in measured):
        return ""
    rows = []
    for direction, label, fingerprint in measured:
        data = json.loads((HERE / f"minmod_{direction}_final.json").read_text())
        _validate_direction_result(data, direction, fingerprint)
        for pair in data["pairs"][-2:]:
            rows.append(f'<tr><td>{label}</td><td>{pair["h"]:.6g}</td>'
                        f'<td>{pair["response_sensitivity"]:.8e}</td>'
                        f'<td>{pair["central_reanalysis"]:.8e}</td>'
                        f'<td>{100*pair["relative_error_adjoint_denominator"]:.3e}%</td></tr>')
    if not rows:
        return ""
    return (
        '<section id="fvMinmodAdditionalDirections"><h3>4×5 minmod: 배경 파라미터와 중간 시각 편향</h3>'
        '<p>같은 정상점·Hessian을 재사용하고 현재 점수의 직접항·제어 gradient를 다시 확인했습니다. '
        'θ는 배경 평균장만, 중간 시각 편향은 두 번째 관측장의 20개 화소만 바꿉니다.</p>'
        '<div style="max-width:100%;overflow-x:auto"><table style="border-spacing:12px 6px;white-space:nowrap">'
        '<thead><tr><th>방향</th><th>h</th><th>수반 방향미분</th><th>중앙 재분석 차분</th><th>수반 대비 차이</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        '<p>두 연속 크기의 양·음 재분석에서 최대 gradient &lt; 1e-10 및 54개 RK 단계의 분기를 확인했습니다. '
        '고정 합성 점수의 국소 응답 검증이며 신경망 학습 개선·일반 minmod FSOI·유한 경로 인증은 아닙니다.</p>'
        '<p><a href="../../graphify-out/fv-root-cause-20260919/MINMOD_CACHE_AND_DIRECTIONS_REVIEW.md">실행 기록·범위</a></p></section>'
    )


def _colour(value, scale):
    """Symmetric blue-paper-red colour for a normalized sensitivity value."""
    t = max(-1.0, min(1.0, float(value) / scale))
    if t < 0:
        a, b, u = (27, 76, 99), (237, 238, 228), t + 1.0
    else:
        a, b, u = (237, 238, 228), (122, 31, 43), t
    rgb = tuple(round(x + (y - x) * u) for x, y in zip(a, b))
    return "#%02x%02x%02x" % rgb


def _row_colour_runs(values, row, scale):
    """Return exact contiguous colour runs for one raster row."""
    colours = [_colour(float(values[row, col]), scale) for col in range(240)]
    runs = []
    column = 0
    for colour, group in groupby(colours):
        width = sum(1 for _ in group)
        runs.append((column, width, colour))
        column += width
    return runs


def _write_sensitivity_map():
    payload = torch.load(RESPONSE_PATH, weights_only=True)
    sensitivity = payload.get("sensitivity")
    if not isinstance(sensitivity, torch.Tensor) or sensitivity.ndim != 3 or tuple(sensitivity.shape) != (3, 240, 240):
        raise SystemExit("saved sensitivity tensor is not the expected [3,240,240] map")
    values = sensitivity.detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise SystemExit("saved sensitivity map contains nonfinite values")
    scale = float(values.abs().max())
    if not _finite(scale) or scale <= 0:
        raise SystemExit("saved sensitivity map has no finite nonzero scale")
    labels = ("첫 관측 −20분", "중간 관측 −10분", "최근 관측 0분")
    for pane, (label, map_path) in enumerate(zip(labels, MAP_PATHS)):
        cells = [f'<text x="4" y="12" font-family="monospace" font-size="9">{label}</text>']
        for row in range(240):
            for column, width, colour in _row_colour_runs(values[pane], row, scale):
                cells.append(
                    f'<rect x="{column}" y="{row + 18}" width="{width}" height="1" fill="{colour}"/>'
                )
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 258" preserveAspectRatio="xMidYMid meet" '
            f'role="img" aria-label="{label} 240 by 240 exact FV observation sensitivity map">'
            f'<title>Exact FV observation sensitivity: {label}</title>'
            '<desc>Symmetric normalized colour scale from negative blue through zero paper to positive red.</desc>'
            '<g shape-rendering="crispEdges">' + "".join(cells) + "</g></svg>\n"
        )
        map_path.write_text(svg)
    return scale


def _panel(report, scale, central):
    impact_rows = []
    impacts = list(report["impacts"])
    if central is not None:
        impacts.append(central["negative_impact"])
    for impact in impacts:
        actual = abs(impact["actual_change"])
        relative_error = abs(impact["actual_change"] - impact["linear_prediction"]) / actual if actual else None
        relative_cell = f"<td>{relative_error:.2%}</td>" if relative_error is not None else "<td>N/A (실제 변화 0)</td>"
        impact_rows.append(
            "<tr>"
            f"<td>{impact['step']:.3g}</td>"
            f"<td>{impact['actual_change']:.8e}</td>"
            f"<td>{impact['linear_prediction']:.8e}</td>"
            f"<td>{impact['taylor_error']:.3e}</td>"
            f"<td>{impact['gradient_max']:.3e}</td>"
            f"<td>{impact['face_margin']:.3e}</td>"
            + relative_cell
            + "</tr>"
        )
    taylor_ratio = (
        abs(report["impacts"][0]["taylor_error"])
        / abs(report["impacts"][1]["taylor_error"])
    )
    if central is None:
        central_text = "중앙 차분 확인 대기 중입니다."
        central_link = ""
    else:
        central_relative = (
            f"{central['relative_error']:.3e}"
            if central["relative_error"] is not None
            else "N/A"
        )
        central_text = (
            "중앙 차분 slope "
            f"{central['central_slope']:.8e}, 수반 slope "
            f"{central['adjoint_slope']:.8e}, 상대 오차 {central_relative}; "
            f"양의/음의 endpoint max |gradient| "
            f"{central['positive_gradient_max']:.3e} / "
            f"{central['negative_gradient_max']:.3e}."
        )
        central_link = ' · <a href="../../graphify-out/fv-root-cause-20260919/rotation240_stable_central_0.0005.json">중앙 차분 JSON</a>'
    long_horizon = ""
    if LONG_HORIZON_PATH.is_file():
        convergence = json.loads(LONG_HORIZON_PATH.read_text())
        if convergence["leads"] != 18 or convergence["sizes"] != [32, 64, 128]:
            raise SystemExit("long-horizon report must cover 18 leads and grids 32/64/128")
        labels = {"translation": "병진", "rotation": "회전", "area_preserving_strain": "면적보존 변형"}
        rows = "".join(
            f"<tr><td>{labels[row['case']]}</td>"
            f"<td>{row['leads'][-1]['relative_echo_l2']:.2%}</td>"
            f"<td>{row['derivative']['jvp_relative_to_independent_oracle']:.2%}</td></tr>"
            for row in convergence["results"] if row["size"] == 128
        )
        long_horizon = (
            '<h3>별도 3시간 처방 유동 정확도 검사</h3>'
            '<p>48km 영역·32/64/128 격자·알려진 영 경계의 donor-cell 시험입니다. '
            '격자를 세분하면 오차가 감소하지만 아래 128격자 오차가 남습니다. '
            '이산 수반의 일치가 연속 방정식에 대한 고정밀 민감도를 뜻하지 않습니다.</p>'
            '<table style="border-spacing:12px 6px"><thead><tr><th>유동</th><th>에코 상대 L₂ 오차</th>'
            f'<th>독립 기준 대비 JVP 오차</th></tr></thead><tbody>{rows}</tbody></table>'
            '<p>처방 유동의 장시간 수치확산 검사이며, 240×240 역문제나 전체 D7 비용 검증이 아닙니다. '
            '<a href="../../graphify-out/fv-root-cause-20260919/fv_grid_convergence_180min.json">3시간 원시 결과·소스 해시</a></p>'
        )
    comparison = ""
    paths = [HERE / f"fv_comparison_{scheme}_180min.json" for scheme in ("donorcell", "minmod")]
    if any(path.is_file() for path in paths):
        if not all(path.is_file() for path in paths):
            raise SystemExit("transport comparison requires both scheme reports")
        reports = {scheme: json.loads(path.read_text()) for scheme, path in zip(("donorcell", "minmod"), paths)}
        _validate_transport_comparison(reports)
        comparison_rows = []
        for scheme, data in reports.items():
            for row in data["results"]:
                if row["size"] != 128:
                    continue
                last = row["leads"][-1]
                peak_ratio = last["predicted_shape"]["maximum_echo"] / last["reference_shape"]["maximum_echo"]
                label = {"translation": "병진", "rotation": "회전", "area_preserving_strain": "변형"}[row["case"]]
                comparison_rows.append(
                    f"<tr><td>{label}</td><td>{scheme}</td>"
                    f"<td>{last['relative_echo_l2']:.2%}</td>"
                    f"<td>{row['derivative']['jvp_relative_to_independent_oracle']:.2%}</td>"
                    f"<td>{peak_ratio:.2%}</td></tr>"
                )
        comparison = (
            '<h3 id="fvTransportComparison">같은 3시간 조건의 저확산 후보 비교</h3>'
            '<p>48km·128격자·동일 초기장과 알려진 영 경계에서 계산했습니다. '
            '원래 애니메이션을 교체한 결과가 아닌 별도 처방 유동 시험입니다. '
            'max 비율은 기준해 최대 에코에 대한 예측 최대 에코의 비율입니다.</p>'
            '<div style="max-width:100%;overflow-x:auto"><table style="border-spacing:12px 6px;white-space:nowrap">'
            '<thead><tr><th>유동</th><th>수송</th><th>에코 L₂ 오차</th><th>기준 대비 AD JVP 오차</th><th>max 비율</th></tr></thead>'
            f'<tbody>{"".join(comparison_rows)}</tbody></table></div>'
            '<p>minmod의 형상 오차는 감소했지만 limiter 분기 안정성과 정확 FSO/FSOI 연결은 미검증입니다. '
            '검열 raw 숫자의 민감도 0은 검열 관측을 제거해도 영향이 없다는 뜻이 아닙니다.</p>'
            '<p><a href="../../graphify-out/fv-root-cause-20260919/fv_comparison_donorcell_180min.json">donor-cell 원시 결과</a> · '
            '<a href="../../graphify-out/fv-root-cause-20260919/fv_comparison_minmod_180min.json">minmod 원시 결과</a> · '
            '<a href="../../graphify-out/fv-root-cause-20260919/REGRESSION_AND_TRANSPORT_COMPARISON.md">검증 범위·폭·면적·비용</a></p>'
        )
    joint_inverse = ""
    joint_path = HERE / "minmod_joint_inverse_final.json"
    if joint_path.is_file():
        joint = json.loads(joint_path.read_text())
        _validate_joint_inverse(joint)
        rows = "".join(
            f"<tr><td>{row['direction']}</td><td>{row['h']:.4g}</td>"
            f"<td>{row['adjoint']:.8e}</td><td>{row['central_reanalysis']:.8e}</td>"
            f"<td>{row['absolute_error']:.3e}</td></tr>"
            for row in joint["reanalysis"]
        )
        joint_inverse = (
            '<h3 id="fvMinmodJointInverse">작은 minmod 공동 역문제: 실제 재분석 대조</h3>'
            '<p>3×3 초기장 9개 값·유동 계수 2개·성장을 같은 minmod 목적함수로 함께 추정했습니다. '
            '고정된 알려진 경계, FP64, 구간당 9 substep의 별도 국소 실험입니다. '
            '검증장은 기준 분석 예측에 합성 오차를 더한 뒤 고정했습니다. '
            '독립 예측 성능 검증이나 위 애니메이션을 바꾼 결과가 아닙니다.</p>'
            f'<p>정상점 max |gradient| {joint["gradient_max"]:.3e}, '
            f'수반 상대잔차 {joint["adjoint_relative_residual"]:.3e}. '
            '각 RK 단계의 엄격한 분기를 검사하고 관측과 배경 파라미터를 각각 바꿔 다시 풀었습니다.</p>'
            '<div style="max-width:100%;overflow-x:auto"><table style="border-spacing:12px 6px;white-space:nowrap">'
            '<thead><tr><th>방향</th><th>h</th><th>수반 방향미분</th><th>중앙 재분석 차분</th><th>절대 차이</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            '<p>유한 섭동 구간 전체 인증·일반 minmod FSOI·신경망 학습 완료를 뜻하지 않습니다. '
            '공개 정확 응답의 donor-cell 제한은 유지합니다. '
            '<a href="../../graphify-out/fv-root-cause-20260919/minmod_joint_inverse_final.json">원시 결과·측정 소스</a></p>'
        )
    return f'''  <details class="learn" id="fvResponseEvidence">
    <summary>240×240 FV 관측 민감도·유한 섭동 연구 검증</summary>
    <div class="lesson-body">
      <p><strong>기존 화면의 원래 forecast·analysis 자료와 새 응답 계산을 구분합니다.</strong> 원래 demo-data 배열은 그대로 두고, 저장된 refined 분석 제어값에서 exact robust Hessian 응답을 계산한 별도 진단입니다. 전역 최적점, 운영 FSO/FSOI 적격성, 일반 FV를 주장하지 않습니다.</p>
      <p>민감도 지도는 저장된 <strong>[3, 240, 240]</strong> 응답 텐서에서 만들었습니다. 값은 <strong>d(weighted mean dBZ²) / d(observation dBZ)</strong> 단위이며, 세 관측 시각에 하나의 최대 절대값 {scale:.6e}을 적용해 대칭 정규화한 [-1, 1]입니다. 파랑은 음수, 종이색은 0, 빨강은 양수입니다.</p>
      <figure style="margin:12px 0 18px;background:var(--paper2);padding:10px;border:1px solid var(--rule)">
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:10px;max-width:100%">
          <img src="../../graphify-out/fv-root-cause-20260919/rotation240_stable_response_18_sensitivity_0.svg" alt="첫 관측 −20분의 240×240 exact FV observation sensitivity map" style="display:block;max-width:100%;width:100%;height:auto;image-rendering:pixelated">
          <img src="../../graphify-out/fv-root-cause-20260919/rotation240_stable_response_18_sensitivity_1.svg" alt="중간 관측 −10분의 240×240 exact FV observation sensitivity map" style="display:block;max-width:100%;width:100%;height:auto;image-rendering:pixelated">
          <img src="../../graphify-out/fv-root-cause-20260919/rotation240_stable_response_18_sensitivity_2.svg" alt="최근 관측 0분의 240×240 exact FV observation sensitivity map" style="display:block;max-width:100%;width:100%;height:auto;image-rendering:pixelated">
        </div>
        <figcaption style="margin-top:8px;color:var(--muted);font-size:.82rem">첫 관측 −20분 · 중간 관측 −10분 · 최근 관측 0분 · 세 지도 공통 색상 정규화 · 실제 3D 값은 원시 tensor 파일에 보존</figcaption>
      </figure>
      <p>측정 소스 커밋 <code>{MEASURED_REVISION[:7]}</code>의 저장 결과입니다. 작은 성장량을 보존하는 계산식에서 분석 상태를 보정하고 수반을 계산했습니다. 소스·응답 텐서·분석 체크포인트 해시를 확인해 재사용했으며, 현재 코드에서 다시 계산한 결과는 아닙니다.</p>
      <p>응답 score {report['score']:.8e}, sensitivity norm {report['sensitivity_norm']:.8e}, gradient max {report['gradient_max']:.3e}, adjoint relative residual {report['adjoint_relative_residual']:.3e}, face margin {report['face_margin']:.3e}.</p>
      <div class="table-scroll" style="max-width:100%;overflow-x:auto"><table style="border-collapse:separate;border-spacing:10px 6px;white-space:nowrap"><thead><tr><th>h (dBZ)</th><th>실제 Δscore</th><th>1차 예측 h·s</th><th>절대 잔차</th><th>max |gradient|</th><th>분기 여유</th><th>실제 변화 대비 오차</th></tr></thead><tbody>{''.join(impact_rows)}</tbody></table></div>
      <p>상대오차 = |실제 변화 − 1차 예측| / |실제 변화|. 실제 변화가 작으면 상대오차가 커질 수 있으므로 절대 잔차와 함께 해석합니다. 허용 가능한 유한 섭동 범위는 아직 검증하지 않았습니다.</p>
      <p>각 actual 값은 perturbed observations를 다시 자료동화해 얻은 값이고, h·s는 저장 응답의 signed linear prediction입니다. 두 양의 h에서 Taylor error 비는 <strong>{taylor_ratio:.9f}</strong>로 측정됐습니다. h를 절반으로 줄이자 오차가 약 4분의 1이 되어 2차 Taylor 잔차와 일치합니다. 다만 이 크기의 섭동에서는 1차 예측에 비해 잔차가 큽니다. 국소 미분 검증과 유한 변화량의 근사 정확도는 구분해야 합니다.</p>
      <p>{central_text}</p>
      {long_horizon}
      {comparison}
      {joint_inverse}
      {_local_path_panel()}
      {_additional_directions_panel()}
      <p><a href="../../graphify-out/fv-root-cause-20260919/rotation240_stable_response_18.json">최종 응답 JSON</a> · <a href="../../graphify-out/fv-root-cause-20260919/rotation240_stable_response_18.pt">응답 tensor</a>{central_link} · <a href="../../graphify-out/fv-root-cause-20260919/rotation240_stable_response_18_sensitivity_0.svg">민감도 −20분</a> · <a href="../../graphify-out/fv-root-cause-20260919/rotation240_stable_response_18_sensitivity_1.svg">−10분</a> · <a href="../../graphify-out/fv-root-cause-20260919/rotation240_stable_response_18_sensitivity_2.svg">0분</a></p>
    </div>
  </details>
'''


def main():
    report = _load_final_report()
    central = _load_optional_central(report)
    scale = _write_sensitivity_map()
    block = _panel(report, scale, central)
    for name in ("template.html", "index.html"):
        path = ROOT / "examples/weather_scenarios" / name
        text = path.read_text()
        original_data = re.search(r'<script id="demo-data" type="application/json">(.*?)</script>', text, re.S).group(1)
        text = re.sub(r'  <details class="learn" id="fvResponseEvidence">.*?</details>\n', '', text, flags=re.S)
        text = text.replace('  <p class="foot">', block + '  <p class="foot">', 1)
        new_data = re.search(r'<script id="demo-data" type="application/json">(.*?)</script>', text, re.S).group(1)
        if new_data != original_data:
            raise SystemExit(f"{name}: demo-data changed")
        path.write_text(text)
        print(name, "frame data unchanged", hashlib.sha256(new_data.encode()).hexdigest())


if __name__ == "__main__":
    main()
