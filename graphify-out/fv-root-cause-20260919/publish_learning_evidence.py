"""Insert saved numerical evidence into the original demo without replacing frames."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
learning = json.loads((HERE / 'fv_learning.json').read_text())
stationary = json.loads((HERE / 'refine_failed_endpoint.json').read_text())
before = learning['holdout_score_before']
after = learning['holdout_score_after']
passed = learning['holdout_gain_above_error_budget'] and learning['reload_holdout_score_error'] == 0
status = '통과' if passed else '미해소'
block = f'''  <details class="learn" id="fvLearningEvidence">
    <summary>정상점·민감도·학습 검증 — 별도 4×5 합성 실험</summary>
    <div class="lesson-body">
      <p><strong>240×240 회전 사례의 인증 결과가 아닙니다.</strong> 같은 FV 방정식의 작은 CPU FP64 실험에서 정확한 Hessian으로 정상점을 보정하고 관측오차 크기를 학습했습니다. 일반 FV 배경장 prior·운영 학습은 미검증입니다.</p>
      <p>기존 실패 사례의 기울기 노름: {stationary['before']['selected_gradient_norm']:.3e} → {stationary['after_gradient_norm']:.3e}. Hessian 최소 고유값 {stationary['hessian_min_eigenvalue']:.4f}, 조건수 {stationary['hessian_condition_number']:.1f}.</p>
      <p>학습 매개변수 θ = log(관측오차 배율). 정확한 매개변수 민감도 {learning['parameter_gradient']:.8g}. 관측 민감도와 이 매개변수 미분을 구분합니다.</p>
      <p>고정된 별도 검증 자료의 가중 dBZ MSE: {before:.12g} → {after:.12g}. 개선율 {(before-after)/before*100:.6f}%. 재적재 점수 차이 {learning['reload_holdout_score_error']:.3e}. 국소 수치 오차 추정치 {learning['holdout_error_budget']:.3e} (엄밀한 오차 상한 아님). 판정: {status}.</p>
      <p>작은 유한 섭동의 관측 영향은 재동화 Taylor 시험으로 검증했습니다. 현재 학습은 양의 오차 크기 한 개의 갱신이며, 전역 최적성이나 실자료 일반화를 증명하지 않습니다.</p>
      <p><a href="../../graphify-out/fv-root-cause-20260919/fv_learning.json">학습 원시 결과</a> · <a href="../../graphify-out/fv-root-cause-20260919/refine_failed_endpoint.json">정상점 원시 결과</a> · <a href="../../graphify-out/fv-root-cause-20260919/STATIONARITY_PROGRESS.md">검증 범위</a></p>
    </div>
  </details>
'''
for name in ('template.html', 'index.html'):
    path = ROOT / 'examples/weather_scenarios' / name
    text = path.read_text()
    original_data = re.search(r'<script id="demo-data" type="application/json">(.*?)</script>', text, re.S).group(1)
    text = re.sub(r'  <details class="learn" id="fvLearningEvidence">.*?</details>\n', '', text, flags=re.S)
    text = text.replace('  <p class="foot">', block + '  <p class="foot">', 1)
    new_data = re.search(r'<script id="demo-data" type="application/json">(.*?)</script>', text, re.S).group(1)
    assert new_data == original_data
    path.write_text(text)
    print(name, 'frame data unchanged', hashlib.sha256(new_data.encode()).hexdigest())
