# PR292 교차점 진단 RED 최종 검토

`active_intersection_20261011_attempt1/diagnostic.json`과 metrics/resource/run 기록의 저장 배열만 읽고 재산술했다. FV/AD 재실행은 하지 않았다.

- Raw SHA `25f8f8dc…0081f`가 metrics 및 child SHA와 일치하고, 계획 SHA는 `eb3d2a9b…bf814a`다. Guard 1회, exit 0, 37.665초, 표본 peak RSS 760,446,976 bytes, 자원 종료 없음. Canonical `68a6e821…11f9ed8` 유지; optimizer 0, commit false.
- 같은 교차 제어점에서 `Q=0`, production face residual `−0.0`; `ζ=L−R=1.27898e−13`, state-scale bound `2.25192e−12`. root의 L/R은 약 `−0.3316038064535`로 음수 동부호다. 정규화한 두 사건 normal의 singular values는 `1.04005, 0.95828`로 이 유한 계산에서는 독립이다.
- 반경 `1e−9, 5e−10, 2.5e−10`의 12개 표본 모두 네 Q/ζ 부호 조합에 도달했다. 다른 selector/face-sign 변화는 없고 trace 179의 선언한 x limiter selector만 ζ 양쪽에서 달라졌다. 모든 trace의 비유한/동률 표시는 false다.
- 저장된 유한 표본 hull norm은 각각 `0.0559126598, 0.0559126956, 0.0559127135`; common-tangent projected gradient norm은 약 `0.05588223`으로 0이 아니다. hull 가중치는 두 finite sector에 놓이며, 이는 정확한 event-limit Clarke hull이나 정지성 인증이 아니다. 외부-face gradient jump의 n-직교 상대잔차는 `3.04e−6 → 7.60e−7`; limiter jump의 w-직교 상대잔차는 `1.84e−3 → 4.61e−4`다. 유한 반경에서 법선 정렬 경향을 보일 뿐 극한 등식은 입증하지 않는다.
- Native J는 교차점에서 `0.061024868376558995`였다. Q=0 chart의 common-tangent 두 비용 표본은 실제 이동 약 `1e−7, 5e−8`에서 ΔJ `−5.5869e−9, −2.7938e−9`(방향몫 `−0.05586925, −0.05587578`)이고 face check를 통과했다. 두 ζ 값은 `−1.14e−13, 7.82e−14`다. 관측된 하강 방향과 반경 경향은 진단 범위의 유한 증거이며, 정확한 교차면 전체의 최소성 결론은 아니다.
- 이 한 점의 진단은 canonical 변경, 정확한 Clarke 정상성/최소점, 미래 예측 점수 또는 solver 일반 성능을 뒷받침하지 않는다. 현재 FV 완료 표시는 70% 유지, 실제 adjoint 및 current-future score는 미완료다.

최종 findings/task/verification/KG/archive 묶음도 raw 진단과 대조했다. 186/259 plan pin 모두 현재 파일과 일치했고, 추가 독립 oracle를 포함한 5개 저장 source/test snapshot의 해시도 맞았다. ROOT 38개 시험·type 0·Graphify 59 node/146 edge 기록과 생산 import가 없는 독립 oracle 범위가 일치한다. 70%는 검증 JSON의 명시적 마일스톤 정의와 동일하며 실행 성공률로 표현되지 않았다. hull 수치도 교차점의 finite-gradient 표본으로 한정되어 canonical/base와 혼동하지 않는다.
