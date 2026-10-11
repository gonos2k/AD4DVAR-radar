# FV 역문제 정의 RED 최종 검토

문서·README·static-check/verification/KG/task/Graphify 기록을 읽기 전용으로 대조했다. 새 FV/AD/최적화/응답 계산이나 소스 수정은 없었다.

- 기준 목표는 고정 p의 원래 20분 robust-normalized J에 대한 적격 국소최소다. README도 `Q_y[4,3]=0`을 영구 물리 제약이 아닌 활성면 가설로 연결한다. 문서는 `D_phys`의 전역 membership/전체 허용근방이 코드로 인증되지 않았음을 분명히 한다.
- 변수 계약이 source와 일치한다: c 26개, p 13개, p[12]=β는 첫 관측과 고정 pattern으로 initial background를 정한다. analysis/future boundary schedules는 p와 독립적으로 고정한다. mixture ϑ는 보조변수로 따로 쓴다. 현재 β=0. J와 최종 +180분 E(dBZ²)는 구별되어 있다.
- Q 단위와 정규화도 일치한다. static source-bound 산술의 기저 계수 `0.077, -0.048, 0.035, 0.016, -0.009`와 face weights `[0, -0.08, -0.28, -0.14, -0.84]`의 내적은 `Q_y[4,3]=0.104`; 코드의 `q_s=0.84`와 같은 유량 convention이다. 정적 계산은 FV 상태/root/minimum 위치를 찾지 않았고 Q=0은 생성 참값 조건이 아니다.
- P2, per-step Armijo, `R_end<R_start(1−1e−8)`, `‖G‖≤1e−8` stop-candidate 및 `‖∇J‖∞<1e−10` strict-branch stationarity gate는 각각 같은-point closure, step acceptance, 경로 회복, final-qualification 필요 후보, scalar stationarity 후보로 분리됐다. 어느 것도 단독 최소점 완료로 취급되지 않는다.
- Strict-interior mixture/jump gate를 해당 경로의 지원조건으로 한정하고 endpoint weight refusal이 수학적 최소 불가능을 뜻하지 않는다고 명시했다. Smooth Hessian positive-curvature gate도 strict nondegenerate local minimum의 인증 범위로 좁혔으며, 퇴화 최소점과 multi-event qualification을 unresolved로 남긴다. Q=0의 F=0도 충분조건이 아니며 면의 제한해를 원래 문제 해와 동일시하지 않는다.
- Static-check 기록은 생성 Q 재산술 및 8개 source hash와 일치한다. Graphify가 고정한 문서/README/static-check 해시도 현재 파일과 맞고, 새 수치 후보·runtime 수정은 0이다. Canonical `68a6e821…11f9ed8`은 P2-closed endpoint이며 적격 최소점·정확 FV adjoint/response·현재 미래점수는 아직 미완료; 70%는 역사적 연구 마일스톤이지 해결률이 아니다.

추가 RED blocker는 없다. 남은 경계는 원래 물리 admissible domain의 전역 판정과 general multi-event/degenerate minimum certificate가 아직 구현되지 않았다는 문서의 명시적 unresolved 항목이다.
