# PR150 follow-up checklist

- [x] R150-001 P2-PROVENANCE — exact-SHA native MPS workflow 실행과 run ID 기록
  - run `33453174306`, source `f095a497fa2b41bdf2f4d43a8c1bb8d3fb106472`
  - `Run native MPS regression oracles`: SUCCESS
  - signed artifact: 없음. 별도 measured P1 certification은 deterministic MPS
    연산 미지원으로 실패했으며 기존 `MPS full P1: NO-GO` 판정을 유지한다.
- [x] R150-002 P2-NUMERICAL — 기존 MPS E2E 시험의 수치 assertion 보강
  - 완료 증거: 알려진 displacement, CPU parity, valid domain, growth path,
    directed cast 경계가 기존 테스트에서 검증됨
- [x] R150-003 P3-DISTRIBUTION — `0.114.0` import migration README 안내
  - 완료 증거: package root API와 owning module import 예시가 README에 명시됨

## 로컬 검증

- `python -m unittest tests.test_nowcast -q` — 167개 PASS
- `PYTORCH_ENABLE_MPS_FALLBACK=0` workflow regression oracle 5개 — local native MPS PASS
- `git diff --check` — PASS
- package root `__all__` — 핵심 5개 API만 노출

## 범위 제한

다음은 추가하지 않는다: MPS evidence contract, fixture registry, import graph
registry, symbol manifest, checklist validator, 새 lifecycle subprocess,
required MPS branch gate.
