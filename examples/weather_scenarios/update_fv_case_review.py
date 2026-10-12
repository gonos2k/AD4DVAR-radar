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
         "s 계수 0. 두 경로의 제어점·J·R가 같아 가속 미입증. 새 HVP 4회, GN row/풀이 0회. 새 확정점 fb0a의 미래점수는 미평가."),
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
    return records


def number_pair(before: float | None, after: float | None, unit: str = "") -> str:
    if before is None or after is None:
        return "미평가"
    decrease = 100 * (before - after) / before if before else None
    change = f" · {decrease:.6f}% 감소" if decrease is not None else ""
    return f"{before:.12g} → {after:.12g}{unit}{change}"


def render_section(records: list[dict]) -> str:
    cards = []
    for record in records:
        source = "../../graphify-out/fv-root-cause-20260919/" + record["source"]
        if "demo" in record:
            data = record["demo"]
            metrics = [("gradient max", f"{data['gradient_max']:.6g}"),
                       ("GN 선형 수반 잔차", f"{data['adjoint_relative_residual']:.6g}"),
                       ("FV MAE +1/+2분", " / ".join(f"{x:.6g}" for x in data["forecast_mae"]) + " dBZ"),
                       ("persistence MAE", " / ".join(f"{x:.6g}" for x in data["persistence_mae"]) + " dBZ")]
        elif "E" in record:
            metrics = [("E · dBZ MSE", number_pair(record["E0"], record["E"], " dBZ²"))]
        else:
            metrics = [(key, number_pair(record[key + "0"], record[key])) for key in ("J", "R")]
        rows = "".join(f"<dt>{html.escape(key)}</dt><dd>{html.escape(value)}</dd>" for key, value in metrics)
        end = record["end"] or "없음 · 비선택 또는 후보 미평가"
        cards.append(f'''<details class="case-card"><summary>{html.escape(record['title'])}<span>{html.escape(record['status'])}</span></summary>
          <div class="case-detail"><dl>{rows}<dt>시작점</dt><dd>{html.escape(record['start'])}</dd><dt>확정/관찰 끝점</dt><dd>{html.escape(end)}</dd></dl>
          <p>{html.escape(record['counts'])}</p><p>{html.escape(record['note'])}</p><a href="{html.escape(source)}">저장 수치 확인</a></div></details>''')
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
        @media(max-width:520px) {{ .case-detail dl {{ grid-template-columns:1fr; }} }}
      </style>
      <section class="panel" id="session-runs" aria-labelledby="sessionRunsTitle">
        <h2 class="panel-title" id="sessionRunsTitle"><span>수행 사례 · 무엇이 되고, 무엇이 어려운가</span><small>2026-10-12 · PR300–304 저장 근거</small></h2>
        <div class="case-review-body">
          <p>아래 26제어·12점관측·3시간 연구와 기존 4×5·2분 응답 데모는 서로 다른 예시입니다. 비용 J, 정상성 잔차 R, 미래 오차 E를 각각 판정합니다. 사례 제목을 열면 수치와 계산점·근거를 확인할 수 있습니다.</p>
          <div class="case-verdicts">
            <div class="case-verdict"><strong>잘된 범위 · 실제 J/R 진행</strong>작은 gradient 이동은 반복 승인됐고 9단계에서 R 4.215677% 감소. 한 점의 계산·국소모형 검증은 작동합니다. 최종 최소점의 수렴은 아직 확인하지 못했습니다.</div>
            <div class="case-verdict limited"><strong>제한된 범위 · 방향 가속</strong>PR300의 앞선 9개 큰 GN 후보는 J를 낮춰도 시작점보다 R가 증가했습니다. 최종 GN 후보는 R도 감소했지만 gradient 후보보다 높았습니다. gradient+GN 및 gradient+s는 같은 gradient 후보를 반환해 추가 이득이 없었습니다. 기존 수치 solver 결함은 보강됐지만 수렴 효율은 남아 있습니다.</div>
            <div class="case-verdict limited"><strong>미검증 · 물리 복원과 응답</strong>미래 MSE는 작은 고정 합성 사례의 관찰 감소입니다. 참상태 유일 복원·독립 기상예측·정확 수반·섭동 재최적화는 미완료입니다. 준비만 끝난 실험은 실패 사례로 세지 않습니다.</div>
          </div>
          <h3>예시가 이론적으로 풀 수 있는 문제인가</h3>
          <p><strong>원래 정규화 최소화 문제는, 고정된 유한 입력에서 목적함수가 전 제어공간에 연속·유한하게 정의된다는 전제 아래 최소해가 존재합니다.</strong> identity prior를 쓰는 이 예시의 J(c,p) = Σρ(rᵢ(c,p)) + ½‖c‖² + Jₛ(c)는 J ≥ ½‖c‖²를 만족합니다. 따라서 비용의 하위수준집합은 유계이고 닫혀 있으며, 유한차원에서 최소값을 달성합니다. 이는 정확한 수학적 연산에 대한 조건부 존재 논증으로, FP64 overflow나 코드가 거부하는 제어영역까지 보장하지 않습니다.</p>
          <p><strong>최소해 존재와 참상태의 유일 복원은 다릅니다.</strong> 관측 Jacobian은 12×26이므로 매끄러운 분기에서 관측만의 미소 제어 nullspace는 적어도 14차원입니다. prior가 그 방향을 정규화하지만 참상태를 선택한다는 보장은 없습니다. 비선형 수송·변환과 residual 곡률 때문에 identity prior가 있어도 전체 J의 전역 볼록성·유일성이 자동 성립하지 않습니다. 일반적으로 참상태에서 관측잔차가 0이어도 prior 기울기 때문에 참상태가 J의 최소점이 아닐 수 있습니다.</p>
          <p><strong>현재 Q=0 면의 추적은 원래 문제의 제한된 가설입니다.</strong> Q(c)=wᵀtanh(c₂₀:₂₅)=0인 전체 면은 0 유동을 포함하는 닫힌 비공집합이므로 같은 전제 아래 면 제한 최소해도 존재합니다. 하지만 그 해가 현재 차트·limiter 분기 자격을 만족하거나 원래 문제의 최소해라는 보장은 없습니다. 면 위의 접선 정상성, 양측 gradient의 유효한 혼합, 정확 축약 곡률·활성구조를 따로 확인해야 합니다. 국소 최소점이 존재해도 비퇴화 곡률과 고정 활성구조가 없으면 매끄럽고 유일한 파라미터 응답이나 정확 implicit 수반을 얻는다는 보장은 없습니다.</p>
          <h3>현재 기법으로 추적이 어려운 이유</h3>
          <p>저장 근거는 두 병목을 구분합니다. PR300은 selector 변화와 큰 유한 잔차모형 오차가 함께 나타났습니다. PR303 후반은 대체로 모형이 잘 맞지만 이미 방향별 최소배율에서 χ가 약 0.44%로 작습니다. PR303의 8번째는 벡터 오차≈31.046%로 별도 예외입니다. 전자는 국소 미분을 유한 이동으로 연결하는 문제이고, 후자는 방향 효율의 문제입니다. J/R 동시 감소는 탐색 정책이며 J 최소화의 모든 중간 단계에 필요한 조건은 아닙니다. 이 정책이 유효한 경로를 제한할 가능성은 있으나 전체 정체의 원인으로 입증되지는 않았습니다.</p>
          <p>또한 identity prior와 현재 J≈0.06099는 비용이 더 낮은 해에 ‖c‖≤√(2J)≈0.3493을 줍니다. 이 범위에서 유동·성장 tanh 변환의 도함수는 약 0.887 이상입니다. 따라서 이 경로의 정체를 강한 tanh 포화 탓으로 단정할 근거는 없습니다. 위 관측 nullspace도 prior가 포함된 전체 Hessian의 특이성을 뜻하지 않습니다.</p>
          <p><strong>판정: 풀 수 없다는 증거는 없습니다. 현재 방식의 빠른 종단 수렴·유일한 참상태 복원·정확 응답을 보장할 근거도 없습니다.</strong> 기존 합성 한 사례의 서로 다른 계산점·실험이므로 어느 기상 유형 전체가 성공/실패한다고 일반화하지 않습니다.</p>
          <p><a href="../../graphify-out/fv-root-cause-20260919/html_case_review_20261012/THEORETICAL_REVIEW.md">이론 검토와 코드 근거</a> · <a href="../../graphify-out/fv-root-cause-20260919/html_case_review_20261012/CASE_REVIEW.json">사례 목록 JSON</a></p>
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
        match = re.search(r"const DATA = (.+?);\s*\n", page.read_text())
        if match is None:
            raise ValueError("existing replay payload not found")
        data = json.loads(match.group(1))
        (OUT / "TOY_REPLAY_METRICS.json").write_text(json.dumps(
            {key: data[key] for key in ("scope", "metrics", "checks", "source")},
            ensure_ascii=False, indent=2) + "\n")
    records = case_records()
    sources = sorted({record["source"] for record in records})
    pins = {source: hashlib.sha256((EVIDENCE / source).read_bytes()).hexdigest() for source in sources}
    (OUT / "CASE_REVIEW.json").write_text(json.dumps(
        dict(scope="cached PR300–304; no new FV/HVP/forecast calls", cases=records, evidence_sha256=pins),
        ensure_ascii=False, indent=2) + "\n")
    section = render_section(records)
    update_page(HERE / "fv_response_template.html", section)
    if (HERE / "fv_response.html").exists():
        update_page(HERE / "fv_response.html", section)
    print(f"Updated {len(records)} stored cases; original replay payload preserved")


if __name__ == "__main__":
    main()
