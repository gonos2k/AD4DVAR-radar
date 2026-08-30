# PR #149 작업 체크리스트

> 이 파일은 작성 시점의 작업 메모다. PR 상태, merge commit과 CI의 최종
> 권위는 GitHub PR/commit/Actions metadata다. 이 파일을 상태 ledger나 merge
> authority로 사용하지 않는다.

## 기준

- Review baseline: `main@21c7b64f6bc093757056523e6af5fe1e46d80362`
- Tree: `4f79692f85fec7bc39f6cc74ce5094dcb62bb628`
- Predecessor: PR #148, merge `21c7b64f6bc093757056523e6af5fe1e46d80362`
- Predecessor exact-head CI: `33261435655` SUCCESS
- Predecessor main CI: `33277169537` SUCCESS
- Branch start: `origin/main@21c7b64f6bc093757056523e6af5fe1e46d80362`
- Worktree: tracked files clean at branch creation; pre-existing untracked user files preserved

## 검토 항목

| ID | 우선순위 | 의미 경계 | 확인 | 최소 처리 | 검증 |
|---|---:|---|---|---|---|
| R149-001 | P2-ARCHITECTURE | `advar-nowcast`가 package 초기화 때문에 비핵심 ledger/promotion/intervention을 eager-import | REPRODUCED / CLOSED FOR RUNTIME IMPORT | `advar.__init__`을 current core API로 축소; legacy 기능은 직접 module import만 허용 | installed wheel의 CLI import closure 13 modules; ledger/promotion/intervention/ensemble sensitivity 미적재 |
| R149-002 | P2-MPS-EVIDENCE | current-v6 MPS 시험이 primitive만 호출하고 3-frame P0 합성을 실행하지 않음 | REPRODUCED / CLOSED FOR P0 | 기존 primitive test를 current-v6 end-to-end `nowcast()` test로 교체 | native MPS 실행 PASS; 합성 중 발견한 CPU-f64/MPS 승격 결함 직접 수정 |
| R149-003 | P2-GOVERNANCE | PR #148 checklist가 병합 후에도 pre-merge 상태를 보임 | REPRODUCED / CLOSED AS HISTORICAL NOTE | 최종 권위 한 줄만 추가 | `PR148_CHECKLIST.md` 직접 확인 |
| R149-004 | P3-SEMANTICS | registry 표의 `Scientific`이 confirmatory/publication 권한으로 오인될 수 있음 | REPRODUCED / CLOSED BY WORDING | 열 이름과 설명만 좁힘; contract/digest 변경 없음 | registry/README 일치 시험 PASS |

## 과다 보증체계 점검

- `src/advar`는 약 102,290줄이며 `promotion.py`와 `ledger.py`가 약 49%를
  차지한다. 이는 현재 offline nowcast 실행 범위보다 큰 연구·감사 표면이다.
- 이번 변경은 이 두 모듈을 삭제했다고 주장하지 않는다. 대신 기본 CLI와
  top-level API의 import graph에서 격리했다.
- package import 후 `advar.cli`까지 적재되는 `advar` 모듈은 24개에서 13개로
  줄었고, 비핵심 ledger/promotion/intervention 계열은 적재되지 않는다.
- 이 수치를 지키는 새 registry, allowlist 또는 CI ratchet은 만들지 않는다.
  후속 감량은 실제 호출자와 archive 요구를 확인한 뒤 삭제 또는 `tools/`
  격리로 처리한다.

## 구현 검증

- `uv run python -m pytest -q`: 828 passed, 488 subtests passed
- focused adjacent regression: 481 passed, 247 subtests passed
- native MPS current-v6 end-to-end regression: PASS
- CI-equivalent basedpyright: 0 errors
- source/wheel build: PASS (`0.114.0`)
- isolated installed wheel and `advar-nowcast` CLI smoke: PASS
- `git diff --check` and `compileall`: PASS
- new contract generation, evidence family, registry, fixture matrix, lifecycle
  subprocess 또는 checklist validator: 없음

## 보존할 개선

- PR #148의 audit fixture/action/lifecycle 삭제를 되돌리지 않는다.
- current-v6 hard physical gate의 CPU binary64 directed authority를 유지한다.
- multi-radar coverage는 runtime permission token으로 만들지 않고 외부 cohort HOLD로 유지한다.
- operational deployment는 NO-GO / out of scope로 유지한다.

## 금지 범위

- import graph registry/allowlist/manifest/ratchet 추가 금지
- top-level lazy compatibility `__getattr__` 추가 금지
- MPS evidence contract 또는 fixture registry 추가 금지
- checklist authority validator 추가 금지
- 외부 serialization schema와 contract generation 변경 금지

## 결정

| 범위 | 판정 |
|---|---|
| CPU P0/P1 offline path | 기존 가정 아래 GO |
| MPS current-v6 P0 | native end-to-end CODE/RUN PASS |
| MPS full P1 | NO-GO |
| Multi-radar confirmatory verification/FSO/FSOI | HOLD — 실제 cohort 필요 |
| External publication | HOLD |
| Canary/LIVE | NO-GO / OUT OF SCOPE |
| PR merge | HOLD — 명시적 사용자 승인 전 병합하지 않음 |
