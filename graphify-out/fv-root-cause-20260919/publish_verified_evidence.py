"""Publish local verification and NN evidence without recomputing forecasts."""
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
read = lambda name: json.loads((HERE / name).read_text())
data = read('rotation240_checkpoint.json')
old = read('fv_rotation240_strict.json')
checks = {}
for key in ('observations', 'truth', 'persistence'):
    checks[key] = data['cases'][0][key] == old['cases'][0][key]
for method in ('p0', 'fv'):
    for key in ('forecast', 'metrics'):
        checks[f'{method}_{key}'] = data['cases'][0]['methods'][method][key] == old['cases'][0]['methods'][method][key]
assert all(checks.values())
verification = read('rotation240_stationarity.json')
assert verification['stationarity_verified']
data['meta']['provenance']['src/advar/variational.py'] = read('rotation240_loaded_source.json')['variational_sha256_loaded_for_solve']
solver = data['cases'][0]['methods']['fv']['solver']
solver.update(stationarity_verified=True, reason=verification['reason'], local_verification=verification)
data['cases'][0]['methods']['fv']['state'].update(
    stationarity_verified=True, reason=verification['reason'], status='completed')
(HERE / 'rotation240_verified.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
(HERE / 'rotation240_preservation.json').write_text(json.dumps(checks, indent=2)+'\n')

nn = read('fv_neural_learning.json')
obs = read('fv_neural_observation.json')
isolated = read('fv_neural_observation_path.json')
block = f'''  <details class="learn" id="fvNeuralEvidence">
    <summary>실제 신경망 학습·FSOI 수치 검증 — 별도 4×5 합성 실험</summary>
    <div class="lesson-body">
      <p>현재 240×240 회전 분석은 국소 1차 정상성 검사를 통과했습니다. 아래 학습은 별도의 작은 FV 실험이며, 240×240 학습이나 배경장 neural prior 검증 결과가 아닙니다.</p>
      <p>3개 매개변수의 실제 신경망이 관측오차 표준편차를 출력합니다. 정확한 Hessian과 혼합 미분으로 한 번 갱신한 뒤, 다른 초기장·잡음의 자료를 다시 동화했습니다.</p>
      <p>별도 자료의 dBZ MSE: {nn['holdout_score_before']:.14g} → {nn['holdout_score_after']:.14g}. 감소 {nn['holdout_gain']:.5e} ({nn['holdout_gain']/nn['holdout_score_before']*100:.6f}%). 국소 오차 추정 척도 {nn['holdout_error_budget']:.5e} (엄밀한 상한 아님). 모델 저장·재적재 후 제어값·예측장·점수 차이 모두 0.</p>
      <p>고정된 학습 모델에서 관측→특징 정규화→오차 가중치와 초기 배경장 B=y[0] 경로를 함께 미분했습니다. 전체 관측 민감도 방향값 {obs['directional_total']:.9e}; 재동화 중앙차분 오차 {obs['finite_difference_errors'][0]:.3e} → {obs['finite_difference_errors'][1]:.3e}.</p>
      <p>신경망 경로만의 추가 영향 {abs(obs['frozen_stats_directional_difference']):.3e}는 가장 작은 차분 오차보다 작아 독립 판별되지 않았습니다. 작은 합성 실험의 학습·재적재 증거이며, 지속 학습·실자료 성능·일반 FV 운영 적격성은 미검증입니다.</p>
      <p>후속 분리 시험: 원관측·배경장·점수를 고정하고 특징→신경망 오차 경로만 변화시킨 재동화에서 민감도 {isolated['directional_isolated']:.8e}, 중앙차분 오차 {max(isolated['finite_difference_errors']):.3e}로 이 국소 경로를 확인했습니다. 전체 관측을 변화시키는 실험과는 구분합니다. <a href="../../graphify-out/fv-root-cause-20260919/fv_neural_observation_path.json">분리 시험 원시 결과</a></p>
      <p><a href="../../graphify-out/fv-root-cause-20260919/fv_neural_learning.json">신경망 학습 원시 결과</a> · <a href="../../graphify-out/fv-root-cause-20260919/fv_neural_observation.json">전체 관측 민감도</a> · <a href="../../graphify-out/fv-root-cause-20260919/rotation240_stationarity.json">240×240 국소 정상성</a> · <a href="../../graphify-out/fv-root-cause-20260919/LOCAL_VERIFICATION_RESULTS.md">검증 범위와 미완료 항목</a></p>
    </div>
  </details>
'''
path = ROOT / 'examples/weather_scenarios/template.html'
text = re.sub(r'  <details class="learn" id="fvNeuralEvidence">.*?</details>\n', '', path.read_text(), flags=re.S)
text = text.replace('  <p class="foot">', block+'  <p class="foot">', 1)
path.write_text(text)
print('Preserved all forecasts and metrics; prepared verified payload and template.')
