"""Publish cached experiment outcomes in the existing standalone FV replay page."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
OUT = EVIDENCE / "html_case_review_20261012"
START = "<!-- FV_CASE_REVIEW_START -->"
END = "<!-- FV_CASE_REVIEW_END -->"


def load(folder: str, name: str = "METRICS.json") -> dict:
    return json.loads((EVIDENCE / folder / name).read_text())


def evidence_paths(record: dict) -> list[str]:
    return record.get("sources", [record["source"]])


def case_records() -> list[dict]:
    comparison = load("netflux_direction_comparison_20261012")
    subspace = load("netflux_subspace_20261012")
    preparation = load("subspace_rank_fix_20261012")
    progress = load("subspace_active_qr_20261012")
    residual = load("residual_direction_20261012")
    toy = load("html_case_review_20261012", "TOY_REPLAY_METRICS.json")
    records = [dict(title="기존 4×5 · 조건부 FV 응답 데모", status="소형 수치 기준 통과 · 정확 FSOI 미검증",
        start="기존 4×5 저장 사례 · 26제어 경로와 별개", end="분석 후 +1·+2분 저장 프레임",
        demo=toy["metrics"], counts="동일 모형 합성 검증 · 알려진 미래 경계",
        note="비용 감소·gradient 기준·GN 선형 수반 잔차·두 리드의 persistence 대비 MAE 개선이 저장 검사에서 통과. GN 근사 응답이며 전체 원래 문제의 정확 수반·운영 예측성능 인증이 아님.",
        source="html_case_review_20261012/TOY_REPLAY_METRICS.json")]
    for arm in comparison["arms"]:
        records.append(dict(
            title="PR300 · " + ("gradient" if arm["name"] == "baseline_tangent" else "강건 GN"),
            status="실제 감소 확인" if arm["name"] == "baseline_tangent" else "국소 예측의 유한 실현 제한",
            start=comparison["base_control_sha256"],
            end=comparison["current_control_sha256"] if arm["name"] == "baseline_tangent" else None,
            J0=comparison["start"]["J"], J=arm["J"],
            R0=comparison["start"]["R"], R=arm["R"],
            counts=f"후보 {arm['candidate_count']}개 · 양쪽 독립 재평가 후 R 우선 선택",
            note=("gradient가 잔차 기준으로 선택됨. GN은 비용을 더 낮춰 J/R의 우열이 다름."
                  if arm["name"] == "baseline_tangent" else
                  "χ=87.468%지만 채택 배율은 모형 최소배율의 1/512. 앞선 9개 후보는 R 증가. 채택 후보의 벡터 모형오차/실제 변화≈100.073%. 비선택 후보이며 새 확정점이 아님."),
            source="netflux_direction_comparison_20261012/METRICS.json",
        ))
    for title, folder, data, baseJ, baseR, start, note in [
        ("PR301 · gradient + GN", "netflux_subspace_20261012", subspace,
         comparison["arms"][0]["J"], comparison["arms"][0]["R"], comparison["current_control_sha256"],
         "GN 계수 0. 두 경로의 제어점·J·R가 같아 추가 보정 이득은 미입증. 작은 이동의 벡터 오차≈0.004557%."),
        ("PR304 · gradient + 잔차 하강방향 s", "residual_direction_20261012", residual,
         progress["J"], progress["R"], progress["control_sha256"],
         "s 계수 0. 두 경로의 제어점·J·R가 같아 가속 미입증. 새 HVP 4회, GN row/풀이 0회. 당시 fb0a는 미평가였고 이번 후속의 별도 3점 미래 관찰에서 평가함."),
    ]:
        arm = data["arms"][0]
        records.append(dict(title=title, status="한 점 개선 · 추가 방향 이득 없음",
            start=start, end=data["control_sha256"], J0=baseJ, J=arm["J"],
            R0=baseR, R=arm["R"], counts="후보 2개 · 독립 최종 반복 2회 · 확정 1점",
            note=note, source=folder + "/METRICS.json"))
    records.insert(4, dict(title="PR302 · 현재점 반경/KKT 준비", status="준비 완료 · 후보 미평가",
        start=preparation["control_sha256"], end=None, J0=None, J=None, R0=None, R=None,
        counts="HVP 4회 · row VJP 24회 · 후보 0개 · 확정 0점",
        note="GN 진입 반경≈2.666배로 사전 2배 상한 밖. 기법 추가이득 조건에 따른 중단이며, 더 큰 gradient 이동의 실패나 최적화 수렴을 뜻하지 않음.",
        source="subspace_rank_fix_20261012/METRICS.json"))
    records.insert(5, dict(title="PR303 · gradient 연속 9단계", status="연속 개선 · 수렴 미완료",
        start=subspace["control_sha256"], end=progress["control_sha256"],
        J0=subspace["arms"][0]["J"], J=progress["J"], R0=subspace["final_R"], R=progress["R"],
        counts="후보 9개 · 확정 9점 · 새 HVP 완료 16회",
        note="2–9단계는 해당 방향의 선형 잔차모형 최소배율. 후반 χ≈0.437–0.453%로 방향 효율 제한. 8번째 벡터 모형오차≈31.046%, 나머지 8개는 ≤0.015889%; 모두 실제 J/R 승인. 예산 종료; 가산 child 607.160초로 내부 600초 초과, 외부 660초 내. 최종점 후속 JVP 미실행.",
        source="subspace_active_qr_20261012/METRICS.json"))
    for title, folder, name, start_key, end_key in [
        ("PR300 · 이전 11단계 미래점수", "netflux_future_observation_20261012", "METRICS.json", "start_control_sha256", "final_control_sha256"),
        ("PR301 · b647 → d19e 미래점수", "netflux_subspace_20261012", "OBSERVATION_VERIFICATION.json", "start_control_sha256", "final_control_sha256"),
        ("PR304 · 7839 → 2a4d 미래점수", "residual_direction_20261012", "OBSERVATION_VERIFICATION.json", "score_start_control", "score_final_control"),
    ]:
        raw = load(folder, name)
        data = raw.get("metrics", raw)
        records.append(dict(title=title, status="고정 합성 참값의 관찰 감소",
            start=data[start_key], end=data[end_key], E0=data["start_E"], E=data["final_E"],
            counts="미래점수 2회 · 최적화/후보 선택 0회",
            note="20분 분석창 이후 최종 +180분 dBZ MSE. 전체 lead 평균이 아님. 같은 고정 합성 참값에 대한 작은 변화이며 독립 예측성능·통계적 유의성 미검증.",
            source=folder + "/" + name))
    audit_path = EVIDENCE / "signed_span_20261012/AUDIT.json"
    if audit_path.exists():
        audit = load("signed_span_20261012", "AUDIT.json")
        cone, signed = audit["nonnegative_cone"], audit["best_signed_quadrant"]
        records.append(dict(title="PR304 후속 · 비음수 원뿔과 signed span", status="저장 선형모형 진단 · 실제 후보 성과와 구분",
            start=audit["base"]["control_sha256"], end=None,
            model_R0=cone["predicted_R"], model_R=signed["predicted_R"],
            counts="추가 FV/HVP/실제 후보 0회 · 기존 반경에서 4개 부호 영역 비교",
            note="비음수 해는 반경 내부이고 0인 s 계수의 KKT margin≈+0.34812. signed 결합은 반경 경계에서 양측 비용 미소 하강을 만족하며 예상 R를 더 낮춤. 무반경 signed 해는 반경의 약 100배로 실행 불가. 원래 비음수 문제의 solver 오류를 뜻하지 않음.",
            source="signed_span_20261012/AUDIT.json"))
    metrics_path = EVIDENCE / "signed_span_20261012/METRICS.json"
    if metrics_path.exists():
        data = load("signed_span_20261012")
        winner = next((arm for arm in data.get("arms", []) if arm["name"] == data.get("selected_direction")), None)
        if winner is not None:
            records.append(dict(title="PR304 후속 · signed 결합 실제 비교", status="실제 J/R 승인 · 적격 최소점 미완료",
                start=data["base_control_sha256"], end=data["control_sha256"],
                J0=data["base_J"], J=winner["J"], R0=data["base_R"], R=winner["R"],
                counts=f"후보 {data['candidates']}개 · 확정 {data['commits']}점 · 새 HVP 0회 · 저장 HVP 4개 재자격",
                note="동일 2a4d 점의 대안 경로이며 fb0a에서의 후속 스텝이 아님. 이전 준비 비용을 예산에 가산. signed 결합의 한 점 추가 J/R 이득을 확인함. 새점 연산·정확 응답은 미완료지만 미래점수와 독립 시간적분 비교는 아래 별도 실행에서 수행함.",
                source="signed_span_20261012/METRICS.json"))
    reference_path = EVIDENCE / "signed_span_20261012/independent_semidiscrete_reference.json"
    forecast_path = EVIDENCE / "signed_span_20261012/forecast_validation_attempt1/forecast_validation.json"
    if forecast_path.exists():
        scores = load("signed_span_20261012/forecast_validation_attempt1", "forecast_validation.json")["score_results"]
        records.append(dict(title="현재 3점 · 별도 최종 +180분 미래점수", status="평가 완료 · signed의 추가 감소는 매우 작음",
            start=scores["2a4d"]["control_sha256"], end=scores["96108"]["control_sha256"],
            custom_metrics={label + " MSE": f"{scores[label]['terminal_dbz_mse']:.12f} dBZ²" for label in ("2a4d", "fb0a", "96108")},
            counts="최적화/후보 선택/HVP 0회 · score/forward 3회 · 각 18개 미래장 저장",
            note="동일한 고정 참값의 전체 4×5 최종 검증장. signed vs fb0a MSE 감소≈0.000977%. 원시 기록의 off-grid라는 주석 오류는 정오표에서 바로잡았고 원시 값·파일은 보존함. 18리드 평균/독립 수송기준 결과와 따로 표시함.",
            source="signed_span_20261012/forecast_validation_attempt1/forecast_validation.json"))
    if reference_path.exists():
        ref = load("signed_span_20261012", "independent_semidiscrete_reference.json")
        values = ref["candidate_forecast_vs_independent_reference"]
        records.append(dict(title="Historical 96108 candidate · independent semidiscrete 18-lead reference", status="historical synthetic truth comparison · mean/late-lead persistence worse",
            start="분석창 끝 +20분 · 동일 분석장 persistence", end=ref["candidate_control_sha256"],
            custom_metrics={"18리드 평균 예측 MAE": f"{values['mean_mae_dbz_all_18_leads']:.9f} dBZ",
                            "18리드 평균 persistence MAE": f"{values['analyzed_frame_persistence_mae_dbz_all_leads']:.9f} dBZ",
                            "+180분 예측 MAE": f"{values['terminal_mae_dbz']:.9f} dBZ",
                            "+180분 persistence MAE": f"{values['per_lead_analyzed_frame_persistence_mae_dbz'][-1]:.9f} dBZ"},
            counts="NumPy 공간 flux 구현 + DOP853 시간적분 · 두 정밀도 비교 · 모든 18×20 화소 평가",
            note="+10–100분은 분석장 persistence보다 MAE가 낮지만 +110–180분은 높음. 평균·최종 오차도 더 높음. 관측장 persistence가 아니라 분석장을 고정한 기준선. 같은 공간 minmod 모형의 독립 구현/시간적분이며 연속 PDE 참값·보류 기상사례 검증을 뜻하지 않음.",
            source="signed_span_20261012/independent_semidiscrete_reference.json"))
        recovery = load("signed_span_20261012", "PHYSICAL_RECOVERY.json")
        records.append(dict(title="문제 정의 · 참제어 복원과 원래 J의 불일치", status="하한으로 확인 · 참제어는 이 J의 최소점 불가",
            start="동일 고정 입력의 알려진 생성 제어", end=recovery["candidate_control"],
            custom_metrics={"참제어 J 하한 · prior만": f"{recovery['generator_prior_lower_bound_on_J']:.12g}",
                            "현재 후보 전체 J": f"{recovery['candidate_J']:.12g}",
                            "유동 flux 상대 오차 norm": f"{100*recovery['relative_staggered_volume_flux_error_norm']:.4f}%",
                            "에코 log 성장/10분 · 참값→분석": f"{recovery['generating_growth_per_10min']:.5f} → {recovery['estimated_growth_per_10min']:.8f}"},
            counts="추가 FV/HVP/점수 0회 · 동일 identity prior와 봉인 제어의 재산술",
            note="비음수 나머지 손실을 더하기 전 참제어 prior가 이미 현재 전체 J보다 큼. 정확한 생성 제어 복원과 이 고정 J 최소화는 양립하지 않음. 생성 제어의 Q≈0.104도 선택 Q=0 면 밖. 모든 J 최소점의 예측이 나쁘다는 증명은 아니며 목적/관측/prior와 탐색면을 구분해야 함.",
            source="signed_span_20261012/PHYSICAL_RECOVERY.json"))
    native_dir = "native_reference_20261012"
    powell = load(native_dir + "/continuation3/attempt1", "CONTINUATION_METRICS.json")
    support = load(native_dir, "zero_powell_support_diagnosis.json")
    fb0a_hessian = load(native_dir + "/continuation1", "qualification_fb0a_hessian_attempt2.json")
    face_chart = load(native_dir + "/continuation3/attempt1", "face_chart_newton_resume.json")
    modified_face = load(native_dir + "/continuation3/attempt1", "modified_newton_face_qualification_attempt2.json")
    n24_dir = native_dir + "/two_face_intersection_20261013/continuation2_20261013"
    n24 = load(n24_dir, "native24_chord_extension_result.json")
    n24_closure = load(n24_dir, "native24_chord_readonly_terminal_closure.json")
    growth_ray = load(native_dir + "/growth_ray_20261013", "RESULT.json")
    coupled_ray = load(native_dir + "/coupled_ray_20261013", "RESULT.json")
    growth_rows = growth_ray["rows"]
    growth_base_row = next(row for row in growth_rows if row["delta"] == 0.0)
    growth_positive_edge = next(row for row in growth_rows if row["delta"] == 2e-7)
    growth_signatures = {row["branch"]["actual_selector_signature_sha256"] for row in growth_rows}
    coupled_rows = coupled_ray["rows"]
    coupled_base_signature = coupled_rows[0]["branch"]["actual_selector_signature_sha256"]
    first_coupled_branch_change = next(
        row for row in coupled_rows[1:]
        if row["branch"]["actual_selector_signature_sha256"] != coupled_base_signature)
    coupled_nonmonotone = next(row for row in coupled_rows if row["alpha"] == 0.0078125)
    if (n24.get("status") != "one_chord_nonstationary"
            or n24.get("source_archive_closure") is not True
            or n24_closure.get("qualification", {}).get("exact_terminal_hessian") is not False
            or n24_closure.get("qualification", {}).get("minimum_claim") is not False
            or n24_closure.get("qualification", {}).get("response_claim") is not False
            or n24_closure.get("point", {}).get("control_sha256") != n24.get("current_control_sha256")
            or n24_closure.get("point", {}).get("first_order_pass") is not False
            or growth_ray.get("status") != "completed"
            or growth_ray.get("source_unchanged") is not True
            or growth_ray.get("input_unchanged") is not True
            or growth_ray.get("parameters_unchanged") is not True
            or growth_ray.get("score_calls") != 0
            or len(growth_signatures) != 1
            or growth_base_row["control_sha256"] != n24.get("current_control_sha256")
            or coupled_ray.get("status") != "completed"
            or coupled_ray.get("input_unchanged") is not True
            or coupled_ray.get("metric_input_unchanged") is not True
            or coupled_ray.get("calls", {}).get("gradient") != 0
            or coupled_ray.get("calls", {}).get("Hessian") != 0
            or coupled_ray.get("calls", {}).get("score") != 0):
        raise ValueError("latest 6ca endpoint or fixed-ray diagnostics failed receipt closure")
    if (modified_face.get("status") != "two_corrections_exhausted_without_stationarity"
            or len(modified_face.get("accepted_steps", [])) != 2
            or modified_face.get("source_hashes_before") != modified_face.get("source_hashes_after")
            or modified_face.get("fixed_input_identity", {}).get("parameters_sha256") != fb0a_hessian["parameters_sha256"]
            or modified_face.get("cycles", [{}])[2].get("point", {}).get("first_order_pass") is not False):
        raise ValueError("completed modified face-chart continuation failed its closure checks")
    modified_first, modified_second = modified_face["accepted_steps"]
    modified_final = modified_face["cycles"][2]["point"]
    initial_face_lambda_min = modified_face["cycles"][0]["modified_step"]["original_H_lambda_min"]
    middle_face_lambda_min = modified_face["cycles"][1]["modified_step"]["original_H_lambda_min"]
    if (initial_face_lambda_min >= 0.0 or middle_face_lambda_min <= 0.0
            or modified_first["alpha"] != 1.0 or modified_second["alpha"] != 1.0
            or modified_final["control_sha256"] != modified_face["control_sha256"]):
        raise ValueError("modified Newton face-chart evidence does not match its accepted sequence")
    zero_callback = next(
        json.loads(line)
        for line in (EVIDENCE / native_dir / "continuation3/attempt1/continuation_evaluations.jsonl").read_text().splitlines()
        if json.loads(line).get("start") == "zero" and json.loads(line).get("powell_callback") == 700
    )
    later_cycle = face_chart["cycle_records"][1]
    n24_point = n24_closure["point"]
    n24_chord = n24["iterations"][0]
    n24_accepted = n24_chord["line_search"][-1]
    native_j_union = (n24["native_J_calls_completed_total"]
                      + 1 + growth_ray["native_J_calls"] + coupled_ray["plan"]["native_J_calls"])
    growth_minus_edge = next(row for row in growth_rows if row["delta"] == -2e-7)
    response_doc = native_dir + "/PERTURBATION_RESPONSE.md"
    records.append(dict(
        title="A · Native Powell, current two-face endpoint, and ray diagnostics",
        status="6ca actual-J chord accepted · full/tangent first-order gate still fails",
        start="zero J=34.4276342940 · fb0a J=0.0609888784",
        end=("Powell: zero best-so-far J=8.0591816671 (budget stop), fb0a J=0.0607084311 (1000-call cap) · "
             f"modified face chart: 8eff → 6a5d J={modified_face['native_J']:.12g} · "
             f"separate native24 chord: {n24['input_control_sha256'][:12]} → 6ca56498 J={n24['native_J']:.12g}"),
        custom_metrics={
            "callback 700 · support 수정 후 native J 재생산": f"{zero_callback['J']:.12g} · control {zero_callback['control_sha256'][:12]}",
            "두 start · Powell 제한 시점 최선 J": f"zero {powell['starts']['zero']['best_so_far_J']:.12g} · fb0a {powell['starts']['fb0a']['completed_powell_J']:.12g}",
            "support min / strict tolerance": f"{support['support_min']:.16f} / {support['support_threshold_tolerance']:.3e}",
            "support excess beyond tolerance": f"{support['excess_over_tolerance']:.3e} · boundary supports all 1 · CFL pass",
            "fb0a native gradient ∞ / gate": f"{fb0a_hessian['gradient_inf']:.6g} / {fb0a_hessian['gradient_tolerance']:.1e} · gate fail",
            "fb0a full Hessian": f"PD · λmin {min(fb0a_hessian['hessian_eigenvalues']):.6g} · nonstationary point",
            "later 25D face-chart Hessian": f"not PD · λmin {min(later_cycle['hessian_eigenvalues']):.6g} at {face_chart['control_sha256'][:12]}",
            "modified face-chart corrections": f"{len(modified_face['accepted_steps'])} · α=1 each · J {modified_first['native_J']:.12g} → {modified_face['native_J']:.12g}",
            "final face-chart gradient ∞ / gate": f"{modified_final['gradient_inf']:.6g} / 1e−10 · first-order gate fail",
            "face-chart Hessians along corrections": f"λmin {initial_face_lambda_min:.6g} (shifted descent), then {middle_face_lambda_min:.6g} · final point not curvature-qualified",
            "latest native24 chord": f"J {n24['input_native_J']:.12g} → {n24['native_J']:.12g} · α={n24_accepted['alpha']:.7g} · fixed-H chord",
            "latest 6ca finite-neighborhood gradient hull proxy ∞ / gate": f"{n24_point['full_gradient_inf']:.8g} / 1e−10 · first-order fail",
            "6ca face and terminal qualification": "two faces · strict normal signs · terminal Hessian/P2 not run",
            "growth-coordinate partial": f"same selector signature at h=±2e−7 · dJ/dc25 {growth_minus_edge['growth_partial']:.6g} / {growth_positive_edge['growth_partial']:.6g}; ΔJ(+2e−7)={growth_positive_edge['native_J']-growth_base_row['native_J']:.3e}",
            "stale-metric coupled ray": f"7 native-J / 0 AD · selector changes by α={first_coupled_branch_change['alpha']:.7g}; ΔJ(α=.0078125)={coupled_nonmonotone['native_J']-coupled_rows[0]['native_J']:+.3e} vs linear descent",
            "current sequence native-J union": f"{native_j_union}/512 · main493 + observer1 + growth5 + coupled7 · 6 remain, no new grant",
        },
        counts=(f"zero: {powell['starts']['zero']['Powell_callback_count']} callbacks then cumulative native-J cap · "
                f"fb0a: {powell['starts']['fb0a']['completed_function_evaluations']} calls then maxfev · "
                f"modified face corrections: 2 · native24 chord: 9 J / 4 gradient calls · "
                f"growth ray: 5 J / 4 gradients · coupled ray: 7 J / 0 AD · HVP=0 on these diagnostics"),
        note=("At callback 34, one propagated support value was 3.66e−15 below one, "
              "exceeding the 16-epsilon tolerance by 1.11e−16; all boundary support was one, "
              "there were no unknown incoming faces, and CFL passed. The sealed transport-only "
              "fix updates coverage as a material scalar so constant support stays exact on the "
              "validated divergence-free domain. It reproduces callback 700, then zero-start "
              "Powell stops at its cumulative objective-call cap. fb0a exhausts maxfev without a "
              "stationarity result. At distinct selected-face point 8eff, the exact chart Hessian "
              "had λmin<0; spectrally shifted directions passed actual native-J Armijo twice to 6a5d. "
              "A separate bounded fixed-H chord ended at 6ca. Its finite-gradient hull ∞ norm "
              "0.001145 is still far above 1e−10; terminal Hessian and P2 were not run. Growth is "
              "a fixed-flow/field scalar partial, not a stationarity certificate. The coupled ray uses "
              "an explicitly stale SPD metric and crosses selector signature at its first nonzero "
              "sample, so linear predictions are not actual-J guarantees. All results remain local "
              "and do not certify a minimum."),
        source=native_dir + "/continuation3/attempt1/CONTINUATION_FINDINGS.md",
        sources=[
            native_dir + "/continuation3/attempt1/CONTINUATION_METRICS.json",
            native_dir + "/continuation3/attempt1/CONTINUATION_FINDINGS.md",
            native_dir + "/continuation3/attempt1/continuation_evaluations.jsonl",
            native_dir + "/zero_powell_support_diagnosis.json",
            native_dir + "/continuation1/qualification_fb0a_hessian_attempt2.json",
            native_dir + "/continuation3/attempt1/face_chart_newton_resume.json",
            native_dir + "/continuation3/attempt1/modified_newton_face_qualification_attempt2.json",
            native_dir + "/two_face_intersection_20261013/continuation2_20261013/native24_chord_extension_result.json",
            native_dir + "/two_face_intersection_20261013/continuation2_20261013/native24_chord_readonly_terminal_closure.json",
            native_dir + "/growth_ray_20261013/PLAN.json",
            native_dir + "/growth_ray_20261013/RESULT.json",
            native_dir + "/coupled_ray_20261013/PLAN.json",
            native_dir + "/coupled_ray_20261013/RESULT.json",
            native_dir + "/coupled_ray_20261013/FLUX_SIGN_COMPARISON.json",
            native_dir + "/coupled_ray_20261013/executed_ray.py",
            native_dir + "/coupled_ray_20261013/POST_RUN_SOURCE_PROVENANCE.json",
            native_dir + "/continuation3/CONTINUATION_PLAN.json",
        ],
    ))
    records.append(dict(
        title="B · Native perturbation response gate at 6ca",
        status="Gated · measured finite-gradient hull proxy remains above tolerance",
        start=n24["current_control_sha256"], end=None,
        custom_metrics={
            "latest 6ca measured full/tangent finite-gradient hull ∞ / gate": f"{n24_point['full_gradient_inf']:.8g} / 1e−10 · fail",
            "terminal Hessian / P2": "not run at 6ca",
            "response routine": "wired, but exact qualification callback did not pass",
            "perturbed Powell grids / score": "not run / 0",
            "historical fb0a Hessian": f"PD, but ∞-gradient {fb0a_hessian['gradient_inf']:.6g} failed stationarity",
        },
        counts="Current endpoint: 2 faces / strict normals · 0 terminal Hessian/P2 · response evaluations 0",
        note=("The native response routine is wired but was not run. Its gate requires the current "
              "full/tangent first-order residual below 1e−10, terminal Hessian/P2 qualification, "
              "and a stable smooth branch. At 6ca the measured finite-gradient full/tangent hull proxy infinity norm "
              "is 0.00114502; no terminal Hessian or P2 was computed. The older fb0a full Hessian "
              "was positive definite at a different, nonstationary point and does not open this gate. "
              "No perturbed reoptimization, adjoint response, or score comparison is claimed."),
        source=response_doc,
        sources=[response_doc, native_dir + "/continuation1/qualification_fb0a_hessian_attempt2.json",
                 native_dir + "/two_face_intersection_20261013/continuation2_20261013/native24_chord_readonly_terminal_closure.json"],
    ))
    observer_dir = native_dir + "/current_endpoint_observer_20261013"
    observer_name = "final_native_forecast_20261013.json"
    observer_verify_name = "OBSERVER_VERIFICATION.json"
    observer_path = EVIDENCE / observer_dir / observer_name
    observer_verify_path = EVIDENCE / observer_dir / observer_verify_name
    chord_closure_name = native_dir + "/two_face_intersection_20261013/continuation2_20261013/native24_chord_readonly_terminal_closure.json"
    growth_result_name = native_dir + "/growth_ray_20261013/RESULT.json"
    coupled_result_name = native_dir + "/coupled_ray_20261013/RESULT.json"
    current_curve_path = OUT / "native_forecast_6ca56498_mae.json"
    if observer_path.exists() and observer_verify_path.exists() and current_curve_path.exists():
        observer = load(observer_dir, observer_name)
        observer_verify = load(observer_dir, observer_verify_name)
        current_curve = json.loads(current_curve_path.read_text())
        current_fields_path = EVIDENCE / observer_dir / observer["forecast_fields_npz"]
        fields_sha = hashlib.sha256(current_fields_path.read_bytes()).hexdigest()
        n24_record = load(native_dir + "/two_face_intersection_20261013/continuation2_20261013",
                          "native24_chord_extension_result.json")
        n24_readonly = load(native_dir + "/two_face_intersection_20261013/continuation2_20261013",
                            "native24_chord_readonly_terminal_closure.json")
        growth_data = load(native_dir + "/growth_ray_20261013", "RESULT.json")
        coupled_data = load(native_dir + "/coupled_ray_20261013", "RESULT.json")
        current_budget_union = (n24_record["native_J_calls_completed_total"]
                                + observer["evaluation_counts"]["native_J_completed"]
                                + growth_data["native_J_calls"] + coupled_data["calls"]["native_J"])
        if (observer_verify.get("status") != "passed"
                or observer_verify.get("metric_recomputed_from_saved_npz") is not True
                or observer_verify.get("result_sha256") != hashlib.sha256(observer_path.read_bytes()).hexdigest()
                or fields_sha != observer.get("forecast_fields_sha256")
                or observer.get("endpoint_control_sha256") != n24_record.get("current_control_sha256")
                or observer.get("candidate_native_J_independent") != observer.get("candidate_native_J_stored")
                or observer.get("selection_use") is not False
                or observer.get("minimum_claim") is not False
                or observer.get("response_claim") is not False
                or observer.get("score_calls") != 0
                or n24_readonly.get("qualification", {}).get("exact_terminal_hessian") is not False
                or n24_readonly.get("qualification", {}).get("minimum_claim") is not False
                or n24_readonly.get("qualification", {}).get("response_claim") is not False
                or current_curve.get("candidate_control_sha256") != observer.get("endpoint_control_sha256")
                or current_curve.get("selection_use") is not False
                or current_curve.get("first_lead_candidate_worse_than_persistence_minutes") != 110
                or current_budget_union != 506):
            raise ValueError("current 6ca native observer, one-chord closure, or unique-call union failed verification")
        forecast_metrics = observer["forecast_metrics"]
        native_gradient_inf = n24_readonly["point"]["full_gradient_inf"]
        records.append(dict(
            title="C · Current native forecast at endpoint 6ca",
            status="observer complete · mean MAE slightly worse · not selected",
            start=observer["endpoint_control_sha256"], end="+180min · 18-lead stored truth comparison",
            custom_metrics={
                "native J · independently reproduced": f"{observer['candidate_native_J_independent']:.12g}",
                "latest measured full/tangent finite-gradient hull proxy ∞ / stationarity gate": f"{native_gradient_inf:.8g} / 1e−10 · fail",
                "18-lead mean MAE · forecast / persistence": f"{forecast_metrics['forecast_mean_mae_dbz_all_18_leads']:.6f} / {forecast_metrics['persistence_mean_mae_dbz_all_18_leads']:.6f} dBZ",
                "forecast MAE excess · 18-lead mean": f"+{forecast_metrics['forecast_minus_persistence_mean_mae_dbz']:.6f} dBZ",
                "+180min MAE · forecast / persistence": f"{forecast_metrics['forecast_per_lead_mae_dbz'][-1]:.6f} / {forecast_metrics['persistence_per_lead_mae_dbz'][-1]:.6f} dBZ",
                "18-lead aggregate MSE · forecast / persistence": f"{forecast_metrics['forecast_aggregate_mse_dbz2_all_cells_leads']:.6f} / {forecast_metrics['persistence_aggregate_mse_dbz2_all_cells_leads']:.6f} dBZ²",
                "lead-wise MAE first worse than persistence": f"+{current_curve['first_lead_candidate_worse_than_persistence_minutes']}min (derived from all 18 rows)",
                "observer trajectory / score selection": "1 / disabled",
                "unique native-J sequence total": f"{current_budget_union}/512 · 6 calls remain · no budget reset",
            },
            counts="18 forecast leads · 1 native-J observer recheck · 0 score-selection/HVP calls",
            note=("This observer forecast is from the latest nonstationary 6ca endpoint; its measured "
                  "finite-gradient full/tangent hull proxy is 0.00114502 (not limiting-gradient certification), "
                  "and terminal Hessian/P2 was not run. Forecast "
                  "MAE is below analysis persistence at +10…+100min, then above from +110min; the "
                  "18-lead mean is worse by 0.018785 dBZ. E was observed after fixing 6ca and did not "
                  "qualify or select it. The independent reference is the fixed-case true generator "
                  "trajectory, not the forecast previously associated with control 96108. This is one "
                  "synthetic case, not held-out weather skill or a response/minimum result."),
            source=observer_dir + "/" + observer_name,
            sources=[
                observer_dir + "/" + observer_name,
                observer_dir + "/" + observer_verify_name,
                observer_dir + "/" + observer["forecast_fields_npz"],
                "signed_span_20261012/independent_semidiscrete_reference.npz",
                chord_closure_name,
                native_dir + "/two_face_intersection_20261013/continuation2_20261013/native24_chord_extension_result.json",
                growth_result_name,
                coupled_result_name,
                native_dir + "/coupled_ray_20261013/FLUX_SIGN_COMPARISON.json",
                native_dir + "/coupled_ray_20261013/executed_ray.py",
                native_dir + "/coupled_ray_20261013/POST_RUN_SOURCE_PROVENANCE.json",
            ],
        ))
    return records


def number_pair(before: float | None, after: float | None, unit: str = "") -> str:
    if before is None or after is None:
        return "미평가"
    decrease = 100 * (before - after) / before if before else None
    change = f" · {decrease:.6f}% 감소" if decrease is not None else ""
    return f"{before:.12g} → {after:.12g}{unit}{change}"


def render_section(records: list[dict]) -> str:
    plot_path = EVIDENCE / "signed_span_20261012/forecast_mae.svg"
    plot = ""
    if plot_path.exists():
        svg = re.sub(r"<\?xml.*?\?>|<!DOCTYPE[^>]*>", "", plot_path.read_text(), flags=re.S)
        svg = "\n".join(line.rstrip() for line in svg.splitlines())
        plot = ('<figure style="margin:20px 0">' + svg
                + '<figcaption>18개 미래 리드 · 같은 분석장의 persistence와 비교. +110분부터 예측 MAE가 더 높습니다. 독립 시간적분 기준이며 실제 기상사례 성능은 아닙니다. '
                + '<a href="../../graphify-out/fv-root-cause-20260919/signed_span_20261012/forecast_mae.svg">SVG source · SHA-pinned</a></figcaption></figure>')
    native_plots = []
    for endpoint in ("6ca56498", "8eff6b1f"):
        native_plot_path = OUT / f"native_forecast_{endpoint}_mae.svg"
        native_plot_data = OUT / f"native_forecast_{endpoint}_mae.json"
        if not (native_plot_path.exists() and native_plot_data.exists()):
            continue
        svg = re.sub(r"<\?xml.*?\?>|<!DOCTYPE[^>]*>", "", native_plot_path.read_text(), flags=re.S)
        svg = "\n".join(line.rstrip() for line in svg.splitlines())
        png_href = f"../../graphify-out/fv-root-cause-20260919/html_case_review_20261012/native_forecast_{endpoint}_mae.png"
        if endpoint == "6ca56498":
            caption = ("최신 6ca endpoint · 18 leads (+10…+180분). Original fixed-case generator truth and the analysis-end persistence field are aligned at each lead. "
                       "First worse than persistence: +110분; 18-lead mean MAE is 1.163358 vs 1.144573 dBZ. Observer only; not selected.")
        else:
            caption = ("Earlier 8eff face-chart point · 18 leads against the same fixed-case generator truth and analysis-end persistence. "
                       "Retained as historical context; it is not the current endpoint or a selected control.")
        native_plots.append(
            '<figure style="margin:20px 0">' + svg
            + f'<figcaption>{caption} <a href="{png_href}">PNG</a></figcaption></figure>'
        )
    native_plot = "".join(native_plots)
    cards = []
    for record in records:
        evidence_sources = record.get("sources", [record["source"]])
        source_links = " · ".join(
            f'<a href="{html.escape("../../graphify-out/fv-root-cause-20260919/" + item)}">{html.escape(Path(item).name)}</a>'
            for item in evidence_sources)
        if "demo" in record:
            data = record["demo"]
            metrics = [("gradient max", f"{data['gradient_max']:.6g}"),
                       ("GN 선형 수반 잔차", f"{data['adjoint_relative_residual']:.6g}"),
                       ("FV MAE +1/+2분", " / ".join(f"{x:.6g}" for x in data["forecast_mae"]) + " dBZ"),
                       ("persistence MAE", " / ".join(f"{x:.6g}" for x in data["persistence_mae"]) + " dBZ")]
        elif "E" in record:
            metrics = [("E · dBZ MSE", number_pair(record["E0"], record["E"], " dBZ²"))]
        elif "model_R" in record:
            metrics = [("예상 R · 원뿔→span", number_pair(record["model_R0"], record["model_R"]))]
        elif "custom_metrics" in record:
            metrics = list(record["custom_metrics"].items())
        else:
            metrics = [(key, number_pair(record[key + "0"], record[key])) for key in ("J", "R")]
        rows = "".join(f"<dt>{html.escape(key)}</dt><dd>{html.escape(value)}</dd>" for key, value in metrics)
        end = record["end"] or "없음 · 비선택 또는 후보 미평가"
        cards.append(f'''<details class="case-card"><summary>{html.escape(record['title'])}<span>{html.escape(record['status'])}</span></summary>
          <div class="case-detail"><dl>{rows}<dt>시작점</dt><dd>{html.escape(record['start'])}</dd><dt>확정/관찰 끝점</dt><dd>{html.escape(end)}</dd></dl>
          <p>{html.escape(record['counts'])}</p><p>{html.escape(record['note'])}</p><p>저장 근거: {source_links}</p></div></details>''')
    return f'''{START}
      <style>
        #session-runs {{ margin-top:22px; }}
        .case-review-body {{ padding:16px; }}
        .case-review-body h3 {{ font-size:1.12rem; margin:20px 0 8px; }}
        .case-review-body p {{ margin:8px 0 14px; }}
        .case-verdicts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
        .case-verdict {{ border-left:3px solid var(--blue); background:var(--paper2); padding:12px; }}
        .case-verdict.limited {{ border-color:var(--red); }}
        .case-verdict strong {{ display:block; margin-bottom:8px; }}
        .case-card {{ border-top:1px solid var(--rule); }}
        .case-card summary {{ cursor:pointer; padding:14px 0; min-height:44px; font-weight:600; }}
        .case-card summary span {{ display:block; color:var(--muted); font-size:.86rem; font-weight:400; }}
        .case-detail {{ padding:0 8px 16px; }}
        .case-detail dl {{ margin:0; display:grid; grid-template-columns:130px minmax(0,1fr); gap:6px 12px; }}
        .case-detail dt {{ color:var(--muted); }}
        .case-detail dd {{ margin:0; overflow-wrap:anywhere; font: .85rem/1.6 var(--mono); }}
        #session-runs a {{ color:var(--blue); overflow-wrap:anywhere; }}
        #session-runs svg {{ width:100%; height:auto; }}
        @media(max-width:520px) {{ .case-detail dl {{ grid-template-columns:1fr; }} }}
      </style>
      <section class="panel" id="session-runs" aria-labelledby="sessionRunsTitle">
        <h2 class="panel-title" id="sessionRunsTitle"><span>수행 사례 · 무엇이 되고, 무엇이 어려운가</span><small>2026-10-13 · PR300–304와 후속 저장 근거</small></h2>
        <div class="case-review-body">
          <p>아래 26제어·12점관측·3시간 연구와 기존 4×5·2분 응답 데모는 서로 다른 예시입니다. 비용 J, 정상성 잔차 R, 미래 오차 E를 각각 판정합니다. 사례 제목을 열면 수치와 계산점·근거를 확인할 수 있습니다.</p>
          <div class="case-verdicts">
            <div class="case-verdict"><strong>잘된 범위 · 실제 J/R 진행</strong>작은 gradient 이동은 반복 승인됐고 9단계에서 R 4.215677% 감소. 한 점의 계산·국소모형 검증은 작동합니다. 최종 최소점의 수렴은 아직 확인하지 못했습니다.</div>
            <div class="case-verdict limited"><strong>제한된 범위 · 방향 가속</strong>PR300의 큰 GN 후보는 J를 낮춰도 시작점보다 R가 증가했습니다. 비음수 gradient+GN 및 gradient+s는 추가 이득이 없었지만, 후속 signed 결합은 같은 점에서 더 낮은 실제 J/R를 얻었습니다. 한 점의 추가 개선과 종단 수렴 효율은 구분합니다.</div>
            <div class="case-verdict limited"><strong>Historical 96108 · synthetic future comparison</strong>이전 후보 96108의 독립 semidiscrete truth comparison이 저장돼 있습니다. 그 후보는 18리드 평균 MAE 1.1628 dBZ로 analysis persistence 1.1459보다 높고, +110분부터도 더 나쁩니다. 최신 6ca observer forecast는 아래 별도 receipt와 curve에 표시합니다. 실제 기상사례·정확 수반·섭동 재최적화로 일반화하지 않습니다.</div>
          </div>
          <h3>A/B/C · native reference, response gate, forecast</h3>
          <p>Native Powell 시작점 비교는 목적함수 진행과 stationarity 자격을 분리합니다. 낮은 J 또는 양의 Hessian만으로 최소점을 인증하지 않습니다. 섭동 응답은 원래 native gradient·full-curvature gate 뒤에만 계산합니다. C는 별도로 고정한 한 제어점의 native forecast이며 점수 선택에 사용하지 않았습니다.</p>
          <p><strong>핵심 판정:</strong> 원래 full native J가 더 낮아져도 적격 최소점이 성립한 것은 아닙니다. 두 Powell start 모두 예산·평가 상한에서 멈췄고, fb0a의 실제 gradient는 stationarity gate보다 컸습니다. 그 점의 positive-definite Hessian도 최소점 인증으로 승격하지 않으며, 뒤의 다른 face-chart 점에서는 음의 곡률이 확인됐습니다.</p>
          {native_plot}
          <h3>예시가 이론적으로 풀 수 있는 문제인가</h3>
          <p><strong>원래 정규화 최소화 문제는, 고정된 유한 입력에서 목적함수가 전 제어공간에 연속·유한하게 정의된다는 전제 아래 최소해가 존재합니다.</strong> identity prior를 쓰는 이 예시의 J(c,p) = Σρ(rᵢ(c,p)) + ½‖c‖² + Jₛ(c)는 J ≥ ½‖c‖²를 만족합니다. 따라서 비용의 하위수준집합은 유계이고 닫혀 있으며, 유한차원에서 최소값을 달성합니다. 이는 정확한 수학적 연산에 대한 조건부 존재 논증으로, FP64 overflow나 코드가 거부하는 제어영역까지 보장하지 않습니다.</p>
          <p><strong>최소해 존재와 참상태의 유일 복원은 다릅니다.</strong> 관측 Jacobian은 12×26이므로 매끄러운 분기에서 관측만의 미소 제어 nullspace는 적어도 14차원입니다. prior가 그 방향을 정규화하지만 참상태를 선택한다는 보장은 없습니다. 비선형 수송·변환과 residual 곡률 때문에 identity prior가 있어도 전체 J의 전역 볼록성·유일성이 자동 성립하지 않습니다. 일반적으로 참상태에서 관측잔차가 0이어도 prior 기울기 때문에 참상태가 J의 최소점이 아닐 수 있습니다.</p>
          <p><strong>현재 Q=0 면의 추적은 원래 문제의 제한된 가설입니다.</strong> Q(c)=wᵀtanh(c₂₀:₂₅)=0인 전체 면은 0 유동을 포함하는 닫힌 비공집합이므로 같은 전제 아래 면 제한 최소해도 존재합니다. 하지만 그 해가 현재 차트·limiter 분기 자격을 만족하거나 원래 문제의 최소해라는 보장은 없습니다. 면 위의 접선 정상성, 양측 gradient의 유효한 혼합, 정확 축약 곡률·활성구조를 따로 확인해야 합니다. 국소 최소점이 존재해도 비퇴화 곡률과 고정 활성구조가 없으면 매끄럽고 유일한 파라미터 응답이나 정확 implicit 수반을 얻는다는 보장은 없습니다.</p>
          <h3>현재 기법으로 추적이 어려운 이유</h3>
          <p>저장 근거는 두 병목을 구분합니다. PR300은 selector 변화와 큰 유한 잔차모형 오차가 함께 나타났습니다. PR303 후반은 대체로 모형이 잘 맞지만 이미 방향별 최소배율에서 χ가 약 0.44%로 작습니다. PR303의 8번째는 벡터 오차≈31.046%로 별도 예외입니다. 전자는 국소 미분을 유한 이동으로 연결하는 문제이고, 후자는 방향 효율의 문제입니다. J/R 동시 감소는 탐색 정책이며 J 최소화의 모든 중간 단계에 필요한 조건은 아닙니다. 이 정책이 유효한 경로를 제한할 가능성은 있으나 전체 정체의 원인으로 입증되지는 않았습니다.</p>
          <p>또한 identity prior와 현재 J≈0.06099는 비용이 더 낮은 해에 ‖c‖≤√(2J)≈0.3493을 줍니다. 이 범위에서 유동·성장 tanh 변환의 도함수는 약 0.887 이상입니다. 따라서 이 경로의 정체를 강한 tanh 포화 탓으로 단정할 근거는 없습니다. 위 관측 nullspace도 prior가 포함된 전체 Hessian의 특이성을 뜻하지 않습니다.</p>
          <p>PR304의 비음수 결합은 반경의 약 63.79%만 사용했습니다. 저장 모형의 KKT 검사에서 이미 그 원뿔의 최적해이므로 반경만 늘리는 비교는 우선순위가 낮습니다. 부호를 허용한 결합은 다른 탐색공간입니다. 음의 추가 계수 자체를 하강방향으로 승인하지 않고, 결합 전체의 반경·양측 비용 하강과 실제 J/R 자격을 검사해야 합니다.</p>
          <p><strong>판정: 풀 수 없다는 증거는 없습니다. 현재 방식의 빠른 종단 수렴·유일한 참상태 복원·정확 응답을 보장할 근거도 없습니다.</strong> 기존 합성 한 사례의 서로 다른 계산점·실험이므로 어느 기상 유형 전체가 성공/실패한다고 일반화하지 않습니다.</p>
          <p><strong>참제어 복원 목표에는 더 강한 한계가 확인됐습니다.</strong> 참제어의 prior만 0.94215인데 현재 전체 J는 0.06099입니다. J≥prior이므로 참제어는 이 고정 J의 최소점이 될 수 없습니다. 또한 참제어는 현재 Q=0 탐색면 밖입니다. 최소화 문제의 해가 존재하는 것과 원하는 기상 상태를 복원하는 문제는 다릅니다. solver 개선만으로 이 목표 불일치를 해소할 수는 없습니다.</p>
          <p><a href="../../graphify-out/fv-root-cause-20260919/html_case_review_20261012/THEORETICAL_REVIEW.md">이론 검토와 코드 근거</a> · <a href="../../graphify-out/fv-root-cause-20260919/html_case_review_20261012/CASE_REVIEW.json">사례 목록 JSON</a></p>
          {plot}
          <h3>실행 기록 · {len(records)}개</h3>
          {''.join(cards)}
        </div>
      </section>
      {END}'''


def update_page(path: Path, section: str) -> None:
    text = path.read_text()
    if START in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        updated = before + section + after
    else:
        anchor = '      <div class="lab"'
        if text.count(anchor) != 1:
            raise ValueError(f"expected exactly one replay panel in {path}")
        updated = text.replace(anchor, section + "\n\n" + anchor)
    path.write_text(updated)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    page = HERE / "fv_response.html"
    if page.exists():
        page_text_before = page.read_text()
        match = re.search(r"const DATA = (.+?);\s*\n", page_text_before)
        if match is None:
            raise ValueError("existing replay payload not found")
        data = json.loads(match.group(1))
        original_data_literal = match.group(1)
        (OUT / "TOY_REPLAY_METRICS.json").write_text(json.dumps(
            {key: data[key] for key in ("scope", "metrics", "checks", "source")},
            ensure_ascii=False, indent=2) + "\n")
    records = case_records()
    sources = sorted({source for record in records for source in evidence_paths(record)})
    pins = {source: hashlib.sha256((EVIDENCE / source).read_bytes()).hexdigest() for source in sources}
    signed_span_chart = "signed_span_20261012/forecast_mae.svg"
    pins[signed_span_chart] = hashlib.sha256((EVIDENCE / signed_span_chart).read_bytes()).hexdigest()
    curve_svg = OUT / "native_forecast_8eff6b1f_mae.svg"
    curve_png = OUT / "native_forecast_8eff6b1f_mae.png"
    curve_data = OUT / "native_forecast_8eff6b1f_mae.json"
    current_curve_svg = OUT / "native_forecast_6ca56498_mae.svg"
    current_curve_png = OUT / "native_forecast_6ca56498_mae.png"
    current_curve_data = OUT / "native_forecast_6ca56498_mae.json"
    visualizations: dict[str, dict[str, str]] = {}
    if curve_svg.exists() and curve_png.exists() and curve_data.exists():
        data = json.loads(curve_data.read_text())
        if (data.get("selection_use") is not False
                or len(data.get("lead_minutes_after_analysis", [])) != 18
                or data.get("first_lead_candidate_worse_than_persistence_minutes") != 110
                or not data.get("candidate_control_sha256", "").startswith("8eff")
                or hashlib.sha256(curve_svg.read_bytes()).hexdigest() != data.get("svg_sha256")
                or hashlib.sha256(curve_png.read_bytes()).hexdigest() != data.get("png_sha256")):
            raise ValueError("native 8eff forecast plot failed its saved-array and lead-alignment checks")
        visualizations["native_forecast_8eff6b1f_mae"] = {
            "data_sha256": hashlib.sha256(curve_data.read_bytes()).hexdigest(),
            "svg_sha256": data["svg_sha256"],
            "png_sha256": data["png_sha256"],
            "candidate_control_sha256": data["candidate_control_sha256"],
            "first_worse_lead_minutes": str(data["first_lead_candidate_worse_than_persistence_minutes"]),
        }
    elif any(r["title"].startswith("C · Current native forecast") for r in records):
        raise ValueError("verified C forecast evidence exists, but its 18-lead MAE plot is missing")
    if current_curve_svg.exists() and current_curve_png.exists() and current_curve_data.exists():
        current_data = json.loads(current_curve_data.read_text())
        current_endpoint = next(r["start"] for r in records if r["title"].startswith("C · Current native forecast"))
        if (current_data.get("selection_use") is not False
                or len(current_data.get("lead_minutes_after_analysis", [])) != 18
                or current_data.get("first_lead_candidate_worse_than_persistence_minutes") != 110
                or current_data.get("candidate_control_sha256") != current_endpoint
                or hashlib.sha256(current_curve_svg.read_bytes()).hexdigest() != current_data.get("svg_sha256")
                or hashlib.sha256(current_curve_png.read_bytes()).hexdigest() != current_data.get("png_sha256")):
            raise ValueError("current 6ca forecast plot failed its saved-array and lead-alignment checks")
        visualizations["native_forecast_6ca56498_mae"] = {
            "data_sha256": hashlib.sha256(current_curve_data.read_bytes()).hexdigest(),
            "svg_sha256": current_data["svg_sha256"],
            "png_sha256": current_data["png_sha256"],
            "candidate_control_sha256": current_data["candidate_control_sha256"],
            "first_worse_lead_minutes": str(current_data["first_lead_candidate_worse_than_persistence_minutes"]),
        }
    (OUT / "CASE_REVIEW.json").write_text(json.dumps(
        dict(scope="stored execution/observer receipts and model-only diagnostics; renderer makes no new physical calls", cases=records, evidence_sha256=pins, visualizations=visualizations),
        ensure_ascii=False, indent=2) + "\n")
    section = render_section(records)
    update_page(HERE / "fv_response_template.html", section)
    if (HERE / "fv_response.html").exists():
        update_page(HERE / "fv_response.html", section)
        after = (HERE / "fv_response.html").read_text()
        match_after = re.search(r"const DATA = (.+?);\s*\n", after)
        if match_after is None or match_after.group(1) != original_data_literal:
            raise ValueError("existing replay DATA literal changed during case-review update")
    print(f"Updated {len(records)} stored cases; original replay payload preserved")


if __name__ == "__main__":
    main()
