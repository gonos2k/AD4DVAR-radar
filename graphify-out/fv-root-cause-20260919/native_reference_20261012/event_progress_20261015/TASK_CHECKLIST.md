# Native limiter 전환 → 진행 → 끝점 자격

PR305는 main d13d9a0c로 병합됨을 GitHub에서 확인했다. 기존 전체공간 Powell·동일 ray·기존 미래검증을 반복하지 않는다. 사용자 검토의 완료/미완료 구분을 유지한다.

- [x] 기존 구조 그래프, native 6ca endpoint, limiter bracket 및 원래 J를 확인.
- [x] 기본 detached observer 유지, 명시적 connected capture와 replay 거부 구현.
- [x] 사건 gap은 실제 stage47/48 x(1,1)의 right-left. 두 root를 별도로 찾고 각 점의 실제 active set·법선을 비교.
- [x] 새로운 phase budget: 추가900초/direct-J256/gradient128/Hessian-row432/profile-forward48/RSS2GiB. 이전 누계512/134와 추정3437.7707초에서 기록하며 이전3600초/512회 cap 준수로 표현하지 않음.
- [x] 끝점용120초/direct-J2/gradient12/Hessian-row144/profile12를 예약. 시작/완료와 partial row를 분리.
- [x] source/input/runtime PLAN 고정 및 source snapshot 확인.
- [x] native 시작점 재자격·실제 두 gap root·선택된 profile 법선 독립성.
- [x] 실제 원래 J 후보 진행과 각 새점 미분/사건 갱신:3회 native 감소 채택.
- [x] 끝점 native/gradient/활성구조/선택된 검색 곡률을 같은점에서 확인. Original replay gradient 네 sector 대조 통과.
- [x] 적격 최소점 또는 정확 미해결 사유를 기록: 정상성 미달·활성 limiter P2 부족·y123/124 전환 거부. 활성 limiter의 접선 정상성/검색 곡률을 완결된 P2로 승격하지 않음.
- [x] GREEN/RED 최종 검토, 영향시험48개·타입error0·Graphify증분 기록.
- [ ] 신규PR 게시 및 자동 검사 확인.

현재 활성인 단일 positive-gap 사건만 검색 등식으로 쓴다. 나머지 과거 전환을 영구 제약으로 추가하지 않는다. 사건별 정상성·곡률은 원래 비매끄러운 최소점의 증명이 아니다. 적격 기준점 이전 응답 실행 gate와 같은 합성 기준의 예측 실패 결과를 유지한다.
