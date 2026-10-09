# PR264 정체점의 같은 점 양측 분기와 27변수 결합 보정

이번에는 원래 `95a55578…6df8c842` 점에서 단순 −g 보정을 더 실행하지 않고, 외부 Qy[4,3]=0 면의 비매끄러운 최적조건을 별도 연구 타입으로 평가했다. 원래26개 제어·모든prior·관측·경계·시간/J를 유지했다. 기존 smooth gradient1e−10 기준과 정상점·수반 적격성은 바꾸지 않았다.

## 구현과 사전 계약

기본 transport는 기존 max/min 분할을 그대로 사용한다. 새 opt-in `selected_face_extension`은 선택한 한 면에서만 양의 분기에 (Q,0), 음의 분기에 (0,Q)를 사용한다. **Q 배열을 0으로 덮어쓰지 않으며**, 같은 signed Q의 국소 upwind 분기를 연장한다. CFL/echo/support/경계 budget에서 같은 선택을 적용한다. Checkpoint 재계산은 선택한 설정과 native None을 모두 캡처·복원해, 바깥 context가 바뀌어도 같은 연산을 미분한다. 이는 선택한 분기의 미분을 평가하는 연구 경로이며, 해당 부호 밖에서 다른 물리 모델을 제안하는 것이 아니다.

현재95a의 원래 J·gradient·전체분기를 먼저 재확인했다. 이후 다른25개 제어좌표를 유지한 채 pivot을 atanh로 복원해 Q=0 차트점으로 옮겼다. 영점에서 native J만 계산하며 native tie AD를 양측 미분으로 사용하지 않는다. 같은 차트점의 두 선택 분기에서 g−,g+를 새로 계산한다. 분석360 Euler단계의 다른 면유량과 limiter slope/active-gap을 해당 성분 scale의128eps 기준으로 검사하고, 선택 면만 진단 copy에서 제외했다. 미래 분기나 예측 점수는 이 분석 목적함수의 원인으로 사용하지 않았다.

η=±1e−8의 같은 차트점에서 생산용 Q와 부호, native/extension J·gradient·한 HVP씩의 parity와 분석 branch 일치를 먼저 검사했다. 통과 후에만 각 분기26개 기저 HVP로 두 ambient Hessian을 계산했다. 총 cap은 parity4+basis52=56이다.

결합식은 F=[(1−θ)g−+θg+, Q/qscale]이며 qscale=max|기저·한계 유량가중치|=.84, gradient scale=1(J/표준화제어)을 사전 고정했다. Jacobian은 mixed Hessian, g+−g− column, nᵀ/qscale row를 사용하는27×27 행렬이다. 한 번의 dense pivoted LU 풀이 뒤 실제 상대 선형잔차1e−10을 확인한다. 정상점이나 최소점 판정과는 별개의 결합 보정 후보 생성이다.

원래 제어 반경.05, 최대8 dyadic 후보, θ∈[0,1]을 유지한다. 후보는 Q0 차트로 재구성하고 실제 이동량을 FV 이전에 검사한다. **실제 원래 J와 scaled F² Armijo 감소, 후보의 strict 조건 및 기준점과 같은 다른 분석 selector**를 모두 요구한다. 새점의 두 분기/native J·F·분기·source/input/runtime/deadline을 재확인한 뒤에만 확정한다. 최소점·smoothroot·응답 claim은 항상false다.

새 연구 계획은 내부600초/외부660초/표본RSS1GiB/guard1회/56HVP/선형계1회/최대1보정으로 사전 고정했다. 이는 기존 보정 실행을 더 오래 돌린 것이 아니라, 현재 양측 전체 Hessian과 다른 최적조건을 평가하는 한정된 실험이다.

## 실제 결과: 선형계는 풀렸지만 후보는 미채택

- 기준95a 원래 J=.06130758760875752, native 최대gradient≈.105137은 재현됐다.
- Q0 차트의 native J=.061307587424344834이며 이는 **optimizer 확정점이 아니다**.
- 같은 점의 θ=.34623726456466775, scaled ||F||=.08096580117955689다. 이 값을 기존 raw gradient norm 감소율로 바꾸어 진도라고 표시하지 않는다.
- 보관 F와 Jacobian의 jump column에서 재구성한 양측 unit-normal gradient는 −.0996689644 / +.1881942284다. 법선방향은 면을 향하지만 큰 접선 잔차가 남는다.
- 두26×26 분기 Hessian과27×27 Jacobian을 실제 계산했다. Dense solve의 기록된 상대잔차는3.1469141e−18이다. 두 Hessian의 상대 비대칭은 약1.1e−15로 낮았다.
- 기준점 mixed Lagrangian의 접선 곡률 진단 최소 고윳값은 .22063566이다. L=Jmix−μQ, μ=nᵀgmix/(nᵀn) convention으로 계산했다. **접선 정상성 미달이며 candidate/최소점의 곡률 인증이 아니다.**

단일 guard는553.125924초, 표본 최대 자식RSS535,576,576bytes, exit0으로 완료했다. HVP 시작/완료56/56, PCG0, 결합 선형계1회였다. 시간/RSS 한도로 중단한 결과가 아니다.

8개 후보 중 첫 후보는 곡선 차트의 실제 변위가 반경을 넘어 FV 이전에 거부됐다. 나머지7개 모두 J와 Q roundoff 조건은 통과했으며, 마지막2개는 F² 조건도 통과했다. 그러나 **7개 모두 기준점의 다른 limiter 선택과 달라 고정 branch support gate에서 거부**됐다. 후보 자체에는360단계가 있고 `nonfinite_or_tie=False`였으며, 선택 면을 제외한 다른 면유량의 끝점 부호는 바뀌지 않았다.

| 가장 작은 두 후보 α | 원래 J | ||F||² | 현재점 strict | 기준점 다른 selector 유지 | 채택 |
|---:|---:|---:|---|---|---|
| .01314167293 | .061263705810 | .006314384484 | 통과 | 실패 | 아니오 |
| .006570836463 | .061285188664 | .006404082613 | 통과 | 실패 | 아니오 |

기준 Q0 차트 ||F||²=.006555460961 대비3.67749%,2.30919% 감소를 실제 평가했으나 **미채택 후보의 수치**다. 가장 작은 후보에는 x choose_left26개 변경과 y slope/right-sign의2단계 변화가 남았다. 이는 끝점 사이 selector 변화이며, 여러 경계가 동시에 활성이라는 증명이나 해당 후보의 물리적 부적격성은 아니다. 고정한 현재 분기 근방/후보 격자라는 이번 정책의 한계로 해석해야 한다.

최종 상태는 `coupled_step_refused`, optimizer_steps0이다. **원래95a 확정점·비용·gradient는 보존됐다.** 이번 결과는 최소점 부재,27변수 최적조건의 불가능성, 일반 비매끄러운 Newton 방법의 실패를 뜻하지 않는다. 다음 연구의 쟁점은 원래 문제의 다른 limiter 전환과 현재 branch 갱신을 어떻게 구분해 다룰지이며, 실패를 숨기기 위한 목적함수/기준 변경은 하지 않았다.

## 증거와 한계

계획 SHA83ea077a6d7ba615a84729a3588a92b156687dd02a8fc7320cf9b90ecf5ce299에122source/78archive를 고정했다. 바뀐 core의 이전604709… bytes를 kink_source에 보존했다. 라이브 probe와 실행 당시 보존본·원시 source_before/source_after는 모두 SHA d05c2c…로 같다. 실제 prototype에 strict flag/stage-count 검사와 initial chart-direction J 기울기 검사가 포함돼 있었으며, 중간 초안의 우려를 실행 코드의 결함으로 잘못 승계하지 않았다. 후처리 소스 변경이나 실제 FV 재시도는 없다.

집중시험38개/기존경고18개/2.58초 통과, error-level 타입0. 같은 부호에서의 native parity, tie donor 미분, y/minmod·support·budget, 바깥 context 변경 후 native/selected replay gradient/HVP,27식·기하·실제 PR264 계보·미확정/오류/시간 종료를 시험했다. Graphify는 변경4파일의 격리 AST118nodes/466edges를 추출하고 공유 graph를 보존했다. GREEN/RED 최종검토와 별도 보관값 산술이 일치했다.

**성공한 parity 비교의 개별 수치/벡터는 원시 JSON에 보존되지 않았다.** 검사 통과는 네 parity HVP 완료 후 조건부 Hessian 단계로 진행한 실행 경로와 고정 소스에서 확인하며, raw 비교값을 독립 재산술한 것으로 주장하지 않는다. 기준 g−/g+도 원시 vector 필드는 없지만 저장된 F·jump column·θ에서 대수적으로 복원해 norm과법선값을 대조했다. 후보 g−/g+는 원시 기록에 있다. 이는 증거의 범위 제한이며 새 AD 검증을 수행했다는 뜻이 아니다.

큰 raw/output launcher는 gzip으로 손실 없이 보관했다. KINK_ARCHIVE_20261009.json은 압축/해제 바이트 SHA를 기록하며 raw JSON의 SHA는 부모 child_sha와 같다. `KINK_ANALYZE_20261009.py`는 raw가 없으면 gzip을 읽어 보관 행렬·trace·해시만 재산술한다. 이 후처리는 FV/gradient/HVP를 생성하지 않는다.

원래3시간 정상점/비매끄러운 최소 후보의 최종 자격·수반/재분석·독립 합성 미래점수는 미완료다. 현재 연구의 역사적1차8/8와 외부 FV 가중70/100은 그대로 유지한다. 모델 에코/유량/gradient를 강수량·관측 바람·물리 예측오차로 확대하지 않는다.
