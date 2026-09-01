# PR151 follow-up checklist

- [x] R151-001 P2-CI/AUTHORITY — MPS workflow를 P0 regression 전용으로 축소
  - P1 certification, signing secret, artifact upload 단계를 제거했다.
- [x] R151-002 P2-NUMERICAL/TEST — local FP32 outward cast를 정확히 검증
  - multiplier를 1로 고정하고 바로 다음 FP32 값을 기대한다.
- [x] R151-003 P2-TEST — 선택 lead의 valid domain이 비어 있지 않음을 검증

Full MPS P1은 계속 NO-GO이며 이 체크리스트는 P1 인증을 주장하지 않는다.
