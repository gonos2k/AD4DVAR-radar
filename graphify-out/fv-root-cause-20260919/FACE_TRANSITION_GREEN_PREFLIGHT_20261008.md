# PR #260 face-transition diagnostic — GREEN preflight

**결론: 이 고정 계획의 단일 제한 실행은 진행 가능하다.** 목적함수·gradient·FV/HVP 계산은 이 preflight에서 실행하지 않았다.

계획 파일은 `FACE_TRANSITION_DIAGNOSTIC_PLAN_20261008.json`, SHA-256 `6b79d07ce609ebe35879dcb3f1cec95649a786155c392e7ea041063b4e19283e`다. 독립적으로 해시와 manifest를 확인했고, 113 source pin과 54 archive pin이 모두 일치했다. 계획의 입력은 PR260의 `2cdccade…` endpoint, face는 `Qy[4,3]`, probe는 η=±1e−6 및 ±2e−6, η=0은 원래 전체 J만 계산한다. root에서 real loader 및 producer receipt closure 통과, focused 22 tests 통과, type check 오류 0을 보고받았다.

검토한 수치 경로는 적절하다. 임의 pivot chart는 남은 25개 제어를 고정한 채 `atanh`로 face flux를 역산하고, 각 점에서 생산 flux 연산자로 요청 η와 실제 유량을 대조한다. 비영점 네 점에만 원래 전체 26-control 목적함수와 gradient를 평가하며, η=0에서는 AD 및 branch 판정을 하지 않는다. inner/outer gradient 쌍은 공통 η=0 기하 법선으로 투영하고, 각 쌍의 정확한 ΔPhi 항등식과 유한 선분 위 clamped 최소 norm θ를 기록한다. 이는 표본 진단이며 극한이나 최소점 증명으로 해석하지 않는다.

Observer stage mapping은 profile의 360 analysis + 3240 future stage 수와 일치한다. donor는 실제 observer의 내부 q 및 qx/qy flux로 재구성하고, positive-growth stage-0 boundary에는 생산 `_scale_by_growth`를 적용한다. 모든 실제 경계 edge에 대해 scaling overflow guard를 먼저 확인해 transport의 추가 fallback Euler 호출로 인덱스가 밀리는 경우를 거부한다. 상세 donor/raw/effective boundary trace는 **analysis 360 stage만** 저장한다. future 구간은 전체 branch signature를 analysis/future로 나누어 저장하고 차이 index를 보고하므로, donor 세부 trace의 범위는 계획대로 제한돼 있다.

남은 해석 한계는 분명하다. 유한 η 표본은 한 면의 양측 미분 극한, J 연속성, Phi 연속성 또는 원인의 단일-face 귀속을 증명하지 않는다. 다른 face/limiter 변화는 전체 branch-signature 차이로 함께 보고해야 한다. 이 진단은 물리적 무풍/강수 경계나 기상학적 정확도 평가가 아니다.
