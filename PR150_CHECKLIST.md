# PR150 follow-up checklist

기준: AD4DVAR-radar 추가 검토 결과의 최소 권장 변경 범위.

## 상태

- [ ] R150-001 P2-PROVENANCE — exact-SHA native MPS workflow 실행과 run ID 기록
  - run `33444541302`, source `758b7f5008f1776738520ea47a83a5d80bc658ce`
  - ephemeral `advar-mps` runner로 실행했으나 `actions/setup-python`이
    요구하는 `/Users/runner/hostedtoolcache` 권한 부재로 pytest 전 실패
  - 완료 증거: current-v6 pytest step 성공과 artifact digest까지 확인
- [x] R150-002 P2-NUMERICAL — 기존 MPS E2E 시험의 수치 assertion 보강
  - 완료 증거: 알려진 displacement, CPU parity, valid domain, growth path,
    directed cast 경계가 기존 테스트에서 검증됨
- [x] R150-003 P3-DISTRIBUTION — `0.114.0` import migration README 안내
  - 완료 증거: package root API와 owning module import 예시가 README에 명시됨

## 판정 기록

| 항목 | 상태 | 증거 |
| --- | --- | --- |
| R150-001 | IN PROGRESS | project-local `.venv` workflow 수정 완료; exact-SHA 재실행 대기 |
| R150-002 | CODE-DONE | `tests.test_nowcast`, 167개 PASS; native MPS 실행은 R150-001에서 확인 |
| R150-003 | DONE | README `0.114 import migration` 추가 |

## 로컬 검증

- `python -m unittest tests.test_nowcast -q` — 167개 PASS
- `PYTORCH_ENABLE_MPS_FALLBACK=0` workflow regression oracle 5개 — local native MPS PASS
- `git diff --check` — PASS
- package root `__all__` — 핵심 5개 API만 노출
- 전체 `unittest discover` — 기존 CLI 호출의 `--background-age-minutes` 누락으로 별도 실패

## MPS runner portability

macOS `setup-python`의 system tool-cache 설치를 사용하지 않는다. Self-hosted
runner가 제공하는 `python3.12`로 checkout 내부 `.venv`를 만들고 모든 workflow
명령을 해당 가상환경에서 실행한다.

## 범위 제한

다음은 추가하지 않는다: MPS evidence contract, fixture registry, import graph
registry, symbol manifest, checklist validator, 새 lifecycle subprocess,
required MPS branch gate.
