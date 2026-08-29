# PR #148 — 실행 경로 우선 단순화 체크리스트

## 권위

- 기준 main/merge: `f44e7ef036e3a0eb5526944f2b32fbb79f3b856d`
- 기준 tree: `60ce671508d5dda6560af307e847702fa1799344`
- PR #147 exact-head CI: `33226080064` SUCCESS
- PR #147 merge/main CI: `33252793248` SUCCESS
- 작업 브랜치: `agent/pr148-audit-mps-coverage`
- 병합 권한: 없음. PR 작성은 가능하나 merge는 HOLD

## 이번 사이클의 설계 규칙

- [x] 실제 decoder가 소비하지 않는 41개 metadata envelope 삭제
- [x] caller metadata를 권한으로 바꾸던 generic action wrapper 삭제
- [x] test 이름을 다시 실행하는 subprocess lifecycle probe 삭제
- [x] 별도의 generic cold-replay API 추가 시도 철회
- [x] source coverage 100% 래칫 삭제
- [x] 실제 계산 경로와 그 직접 회귀시험만 최종 diff에 남았는지 확인
- [x] 삭제된 검증 주장과 남은 과학적 HOLD를 README와 체크리스트에 명시
- [x] runtime registry를 current+predecessor 범위로 축소

## 지적별 판정

| ID | 원 지적 | 처리 | 상태 |
|---|---|---|---|
| R148-001 | 41/41 fixture가 실제 historical cold replay가 아님 | envelope와 “41/41 cold replay” 주장을 삭제. 원본 역사 bytes가 없는 세대는 cold replay 지원을 주장하지 않음 | CLAIM WITHDRAWN; archive 확보 전 HOLD |
| R148-002 | caller-controlled audit action authority | `FrozenAuditGeneration`/action wrapper 전체 삭제 | CLOSED BY DELETION |
| R148-003 | current-v6 MPS가 device float64를 생성 | MPS nominal affine은 FP32 device path, hard physical gate는 CPU binary64로 분리. typed verification/source selection은 CPU-only 오류로 조기 거부 | PARTIAL: primitive local PASS, full MPS P1은 backend 제약으로 NO-GO |
| R148-004 | `[0,1]` 시간 구간이 source coverage를 편향할 수 있음 | 100% gate로 실행을 막지 않음. 보수적 unassigned 정책 유지, real-radar cohort coverage 측정 전 confirmatory 승격 주장 보류 | OPEN SCIENTIFIC |
| R148-005 | decoder 지원 집합과 registry 역방향 분류 | 범용 fixture/ratchet으로 해결하지 않음. 실제 decoder의 명시적 지원 집합과 family별 직접 시험 유지 | NOT GENERALIZED |

## 남기는 실행 변경

- [x] current-v6 MPS nominal forward/inverse affine가 device float64를 만들지 않음
- [x] current-v6 hard speed authority는 detached CPU binary64 directed interval 사용
- [x] verification geometry와 source-selection certification은 CPU-only 경계를 조기 선언
- [x] MPS에서 실행되는 physics coordinate 생성의 unsupported `new_tensor` 경로 제거
- [x] current-v6 MPS primitive를 실제 호출하는 회귀시험 추가
- [x] source score의 strict lower-over-all-upper 수학과 top-2 구현 유지

## 직접 검증

- [x] registry 소형 권위 시험
- [x] nowcast current-v6 CPU/MPS 직접 시험
- [x] sensitivity source-selection CPU 경계 및 MPS metric 직접 시험
- [x] basedpyright: 0 errors
- [x] 전체 CPU pytest: 828 passed, 488 subtests
- [x] wheel build/import 및 설치된 CLI shape·finite smoke
- [ ] exact-head CI

## 과학·운영 Gate

| 범위 | 판정 |
|---|---|
| CPU offline exploratory | GO WITH EXPLICIT ASSUMPTIONS |
| CPU multi-radar confirmatory verification/FSO/FSOI | HOLD — 실제 assignment coverage cohort 필요 |
| MPS current-v6 primitive metrics | 코드 경로 추가, exact-head MPS 증거 전 CONDITIONAL |
| MPS full P1 analysis | NO-GO — 현재 PyTorch deterministic MPS backend 제약 |
| Historical cold replay | 원본 bytes와 실제 family loader가 있는 직접 시험만 인정 |
| External publication | HOLD — polygon/geodesy/beam/real-radar cohort 미종결 |
| Canary/LIVE | NO-GO / OUT OF SCOPE |
| PR merge | HOLD |
