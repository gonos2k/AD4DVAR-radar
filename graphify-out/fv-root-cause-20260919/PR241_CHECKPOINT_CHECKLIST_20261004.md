# PR241 후속 — HVP 저장·재개 체크리스트

기준 PR241 headeeb741f0 / main d251b8c4. 종료한 정규화P2는 유지한다.

| ID | 작업 | 상태/종료 조건 |
|---|---|---|
| C1 | HVP 열 저장·재개 수집기 | 구현·합성/실제 저장 검증 완료: 같은 operator receipt·연속26열·열별 hash·atomic 저장·배타 writer·고정 attempt budget |
| C2 | 기존 guarded 진단의 선택 연결 | 대역 및 실제 guarded 연결 확인: 별도 체크포인트·새 출력폴더·부분 counts·audit_pending·실행/수치 구분 |
| D1 | J와 E 의존성 구분 | 3개 대역 연결 시험 통과; 실제 FV 시험이나 E 변경 후 cache reuse 선언은 아님 |
| R1 | 현재3시간 점의 새 H/eigenpair/Schur | 현재점 전체26열·독립27번째 HVP·Schur 확보. 계획된2회 실행으로 완료; 스텝 없음 |
| R2 | 정상점·전체 응답·재분석 | 미완료 유지; R1의 곡률만으로 정상점 또는 예측 skill을 인증하지 않음 |

검증 중 발견한 budget-header 변경 허용과 잘못된 query가 기존 완료 cache 상태를 바꾸는 경로는 회귀시험으로 차단한다. 실제 original J/g/branch 단계와 Hessian kernel quota의 비용을 구분하며, 외부 실행 한도는 kernel ledger만으로 보장했다고 주장하지 않는다.

기존 원시3시간 report와 fresh_hessian/source pins는 그대로다. 이전 저장하지 않은 HVP는 복구·재사용하지 않는다. 의미 없는 극소 반례 확대, 새 solver/물리 옵션, 전체CPU/패키지CI는 이번 범위에 넣지 않는다. 공유 .venv/lock/Graphify·사용자 AGENTS/홈페이지를 보존한다.


## 최종 실제 실행 결과

동화 Hessian J_cc는 현재 cd6b 점에서 새로 수집했다. 첫 실행은26개 basis 열과 독립 검사 product를 저장한 뒤 예산 거부로 끝났고, 두 번째 실행은 이27개 product를 모두 재사용해 검증·Schur를 완료했다. 총 HVP 저장 수27이며 중간 저장이 없는 과거 실패의 제품은 쓰지 않았다. 첫 kernel 시작 열0, 재개 시작 열26이다.

원래 J=.08389718093130327, 전체 gradient 최대=1.0588168237438937, 전체H 고윳값 범위[-1.5290711560,4197.6863178852]. Hff 범위[.8612141104,1821.7462457531], Schur 최소=-27.5013204655이며 음의 방향1개다. 독립 최소 고유쌍 실제 상대잔차2.1938305256e-16이 기존1e-8 기준을 통과했다. 전체 symmetry와 Hff solve gates도 원래 기준대로 검사했다. 최소 고유쌍 대조는 모든 행렬 성분의 별도 검증이 아니다.

첫 guard246.895초/표본 RSS368,033,792bytes, 둘째12.571초/346,636,288bytes, 실험 전체259.579초. 두 실행 모두exit0·completed, 첫 수치checkpoint_budget_refusal / 둘째endpoint_diagnostic_completed. kernel240초 cap을2회 예약했고 외부300초 cap2회로600초를 예약했다. 최대3회/900초 계획에서 완료 즉시 중단해 세 번째 실행은 하지 않았다.

제어값·관측/배경p·J·prior·원래64개 수치 소스·시간/경계·검증장과 source/runtime은 전후 고정됐다.3,600단계 끝점 분기·J/g/Phi 재현도 각 회차에서 통과했다. 이것은 동화 곡률의 새 근거이며 미래 점수 E 또는 기상 예측 skill의 진전이 아니다. 최종 gradient는 원래1e-10 기준 미달이라 정상점·응답은 계속 미완료다.

## 다음 탐색 판단 — 적용하지 않은 제안

현재 H는 부정부호이므로 무수정 Newton–PCG의 SPD 전제를 통과했다고 볼 수 없다. Hff가 SPD인 이 점에서는 동역학 블록에만 mu=max(0,1-lambda_min(S))=28.5013204655를 주면 수정 Schur의 곡률 floor1을 만들 수 있다. 이는 다음 원래J Armijo 탐색의 제안이며 이번에 스텝·shift를 적용하지 않았다. 비영 gf를 포함한 Schur RHS와 원래H 잔차를 유지하고, 채택 이후에는 새점의 곡률/분기를 다시 확인해야 한다. 최종 수반에 수정M_mu를 사용하지 않는다.

authoring 집중 시험60 passed, 기존 경고18개, 오류 수준 타입검사0. J/E 연결3건은 수송을 대역으로 바꾼 의존성 시험이고, 위2회는 실제 원래FV J/g/branch/HVP 진단이다. 과거81/30개 시험과 합산하지 않는다. 독립 기상예측·학습·유한 영향·legacy 가드 전면이관·최신 전체Linux통과는 이 결과로 종료하지 않는다.
