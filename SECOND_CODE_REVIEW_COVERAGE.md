# 두 번째 전체 코드 검토 범위

기준 커밋: `94c75d3323c986b6fb2963336b906acc85384e12`. 팀 모델 `gpt-5.6-luna` / `xhigh`.

34개 담당자의 실제 읽기 기록을 배정 구간별로 검증한 뒤 합쳤다. 71개 파일 166,618행을 모두 포함하며, 각 파일 SHA-256을 기준 Git blob과 대조했다. 중복 읽기는 한 번만 센다. EOF를 넘는 read 명령의 끝 번호는 실제 마지막 행에서 잘랐으며, 원보고서와 정규화 내역은 ZIP에 보존했다.

이것은 읽기 범위이며 결함 부재의 증명이 아니다. 각 행을 두 역할이나 네 분야 모두가 독립 검토했다는 뜻도 아니다. 문서와 내장 JSON 자료 전체의 수작업 읽기는 집계에서 제외한다.

[통합 판정·해결 체크리스트](SECOND_CODE_REVIEW_CHECKLIST.md) · [원보고서·재현·해시 묶음](review_artifacts/team_audit_94c75d3.zip)

## 담당 구간

| 담당 | 역할 | 배정 및 읽기 완료 행 수 |
| --- | --- | ---: |
| ledger_1 | GREEN | 5,600 |
| ledger_2 | RED | 5,600 |
| ledger_3 | GREEN | 5,600 |
| ledger_4 | RED | 5,504 |
| promotion_1 | GREEN | 5,700 |
| promotion_2 | RED | 5,700 |
| promotion_3 | GREEN | 5,700 |
| promotion_4 | RED | 5,700 |
| promotion_5 | GREEN | 5,250 |
| sensitivity_1 | RED | 4,600 |
| sensitivity_2 | GREEN | 4,600 |
| sensitivity_3 | RED | 4,600 |
| sensitivity_4 | GREEN | 4,229 |
| variational_1 | RED | 4,900 |
| variational_2 | GREEN | 4,856 |
| nowcast_1 | RED | 5,300 |
| nowcast_2 | GREEN | 5,263 |
| test_promotion_1 | RED | 5,200 |
| test_promotion_2 | GREEN | 5,200 |
| test_promotion_3 | RED | 5,200 |
| test_promotion_4 | GREEN | 5,194 |
| test_ledger_1 | RED | 5,909 |
| test_nowcast_1 | GREEN | 6,352 |
| test_sensitivity_1 | RED | 4,200 |
| test_sensitivity_2 | GREEN | 4,006 |
| test_variational_1 | RED | 6,579 |
| numerical_core | GREEN/RED | 5,089 |
| runtime_contracts | GREEN/RED | 1,780 |
| intervention | RED | 4,032 |
| run_artifact | GREEN/RED | 5,773 |
| cli_acceptance | GREEN/RED | 3,932 |
| build_geometry | RED | 4,360 |
| ci_dependencies | GREEN/RED | 3,024 |
| lab_evaluation | GREEN/RED | 2,086 |

## 파일별 범위

| 파일 | 기준 행 수 | 읽기 | SHA-256 |
| --- | ---: | ---: | --- |
| [.github/scripts/build_deployment_bundle.py](.github/scripts/build_deployment_bundle.py) | 2,089 | 100% | `1cfe4fac46a4b5f8964b185f57e6ea6c85a0ecdbada99e71ea85c0dd1a4b48d2` |
| [.github/scripts/check_basedpyright.py](.github/scripts/check_basedpyright.py) | 92 | 100% | `b6f220e59d0445fc21298ddf53991da61b4f5d68b5468fe93485f1f31598a846` |
| [.github/scripts/check_dependency_locks.py](.github/scripts/check_dependency_locks.py) | 168 | 100% | `78bac052b40e5af44c52e6cf7513eea00632e0898a12d9f5837861f5e458b05d` |
| [.github/scripts/generate_metric_domain_evidence.py](.github/scripts/generate_metric_domain_evidence.py) | 961 | 100% | `b66cbc33b0be20ec9dca8891024e3a30051b174b7a9347555687d972cc045cc7` |
| [.github/scripts/run_real_case_acceptance.py](.github/scripts/run_real_case_acceptance.py) | 38 | 100% | `e10c9889d7e50cb8afb9c444831be9d659869f1d7e08f2844fb0d26bf1efb58d` |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | 282 | 100% | `71f338dfb4b00e6a136d96f42d8ad263d85350861b24ff164e49caacfd90b59e` |
| [.github/workflows/mps-certification.yml](.github/workflows/mps-certification.yml) | 35 | 100% | `303aeef7df454f44c6c7e08b709c730ecbe43253cba69d22837d9badef2aad99` |
| [benchmarks/benchmark_variational_fso.py](benchmarks/benchmark_variational_fso.py) | 204 | 100% | `ea26602c2afbf0abacceb7882990c693342274d84a601541eb74cb5e81ffd952` |
| [examples/initial_field_lab/app.js](examples/initial_field_lab/app.js) | 253 | 100% | `43986dcf95f8e3dd8dd010012c763f1285a4e815494b42c7264dbbab4c7c0799` |
| [examples/initial_field_lab/index.html](examples/initial_field_lab/index.html) | 198 | 100% | `0f3b197ba5c0b14d11505551bda75f3003fbf6695d600f3f25f96a8fe1a8dd3d` |
| [examples/initial_field_lab/server.py](examples/initial_field_lab/server.py) | 528 | 100% | `dbdebeb08b2cdd3bac20051b217c093a36dee2bb939ec82338868454a512185f` |
| [examples/initial_field_lab/styles.css](examples/initial_field_lab/styles.css) | 549 | 100% | `bd78a8d8b507f8f5f7f7a8e1448dd663cfbafe703fc95f8b2c60ea717351de0f` |
| [pyproject.toml](pyproject.toml) | 28 | 100% | `30b247988a593d315cb027ac56a42d46322703ee2e44b6ca28f66612040a2e2a` |
| [requirements/ci-py310-linux.lock](requirements/ci-py310-linux.lock) | 837 | 100% | `81157f3d4283630a8f8e99d01941923e5d9a178bbb942678bd2223113dda2b0c` |
| [requirements/ci-py312-linux.lock](requirements/ci-py312-linux.lock) | 820 | 100% | `03289d1953b81a0f20f57a14770acac23513fc02cf8ce3f7f684defb0cd1b657` |
| [requirements/runtime-py310-linux.lock](requirements/runtime-py310-linux.lock) | 382 | 100% | `5ae9bc63c41b878e9d0d3149f90d96bc3e40760691387bff9b6397f341c09163` |
| [requirements/runtime-py312-linux.lock](requirements/runtime-py312-linux.lock) | 380 | 100% | `60dcf9d1fb2cc2e56cb61a4e4ab66e0be3392048b45a95a4d457cbe08699242d` |
| [src/advar/__init__.py](src/advar/__init__.py) | 17 | 100% | `eff05b97e93ef997a5266f59a5e24f7a56c8d531767903d7a579504a59762147` |
| [src/advar/_contract_registry.py](src/advar/_contract_registry.py) | 191 | 100% | `fa6e685c47b709ab5fa6aee6745a1b4f42f27b6db4c78a4461130e019f84d849` |
| [src/advar/_digest.py](src/advar/_digest.py) | 46 | 100% | `c35f6f01526ca0bd0caa64bc59b46bc5c2309d642c20de638e119495f3230d5c` |
| [src/advar/_input_derivation.py](src/advar/_input_derivation.py) | 188 | 100% | `2cd4d0691313f9e3819e7414048cf1e3aa56a8e0cb59de4ae908fecc49ec7fcd` |
| [src/advar/_learned_input.py](src/advar/_learned_input.py) | 72 | 100% | `21f43447ef69f3a4b1bc138e5eff2c0e9536ee866ba27882bb9e0f14e81855ac` |
| [src/advar/_runtime.py](src/advar/_runtime.py) | 275 | 100% | `0e0ea6423706e23d109bda60bb506b4e8b7b94fc2fa1bf72ed001ea85279a7d0` |
| [src/advar/acceptance.py](src/advar/acceptance.py) | 431 | 100% | `e670127614ed7cc3abb5208ee3d760a3ba54dcb00916b34537a4c821cabc5f2c` |
| [src/advar/action_artifacts.py](src/advar/action_artifacts.py) | 72 | 100% | `2eeee6618fdcdaeb4a29e4ea70babc50278fc9659a067ed71bf5e25e2e119ebc` |
| [src/advar/action_contracts.py](src/advar/action_contracts.py) | 54 | 100% | `6990d0b469f057ca7619693f5ff21f01bd48905b001625ef51faf6f46b06c8eb` |
| [src/advar/calibration.py](src/advar/calibration.py) | 804 | 100% | `ef53617b3dbffd84819e5943e4324666b99d78d9f8fdfcdecd15e453909af073` |
| [src/advar/cli.py](src/advar/cli.py) | 1,531 | 100% | `5dfd6ac7482cdbb4246148af0c823313de9d5fa3870288f778e84c3bcfd6883f` |
| [src/advar/diagnostics.py](src/advar/diagnostics.py) | 135 | 100% | `148b5b652d91651b00a24e5e8a218707ca1e2194c12e36f61c04d33b4522cdfa` |
| [src/advar/ensemble_sensitivity.py](src/advar/ensemble_sensitivity.py) | 956 | 100% | `b2fd77fe42f6cc7c5490b027a2330938d76e982d99218a082e8e05d4aa011212` |
| [src/advar/intervention.py](src/advar/intervention.py) | 2,763 | 100% | `58bf6f3d4204de37da0f0c9c500929fae3146704d320581577f21241eb864e1e` |
| [src/advar/ledger.py](src/advar/ledger.py) | 22,304 | 100% | `006afacd0ec2ce32c8f478ddedb3513205718ef1bbe14c225143185c4010fb1c` |
| [src/advar/linearization_artifact.py](src/advar/linearization_artifact.py) | 817 | 100% | `f55705fa909ddb34b2f93c30a097e51bfa3e8dc645d45c98960f680457deff4c` |
| [src/advar/matrix_free.py](src/advar/matrix_free.py) | 388 | 100% | `736132bccc24adfeec213afb5ab186a4bc898b10a155524d12de9423e866127d` |
| [src/advar/metrics.py](src/advar/metrics.py) | 69 | 100% | `1363d2162f01bed67ddcac7fa46ca2f589d971d045a5f7ffa1e47fbb3d5d8d4c` |
| [src/advar/nowcast.py](src/advar/nowcast.py) | 10,563 | 100% | `f13d0a049d826eadb5d1e24ec58e1db61c73c70dbede1c37dda4bd81e04fe426` |
| [src/advar/physics.py](src/advar/physics.py) | 199 | 100% | `36790ee8debf11b94ab7c54eee79cb795270e754f9ad7fbea228701d16eb29eb` |
| [src/advar/promotion.py](src/advar/promotion.py) | 28,050 | 100% | `397654b1a370b1ec94c2010f26c1a58b9097816ad779f71662a43ff241e1c07a` |
| [src/advar/range_geometry.py](src/advar/range_geometry.py) | 607 | 100% | `24de2356ffee2e8ea20aa84ab0eac7e9492fa1c0a56f7fd9f3051345a1b25318` |
| [src/advar/run_artifact.py](src/advar/run_artifact.py) | 2,557 | 100% | `1e4475c73861f685daa7abf3280ae16a4858aea7cd1deaf7f8d311cb39d56680` |
| [src/advar/runtime_closure.py](src/advar/runtime_closure.py) | 518 | 100% | `b8ba26a433ae70ed122734fdacb00647a1818f361cd614b44ec9b733056c5918` |
| [src/advar/sensitivity.py](src/advar/sensitivity.py) | 18,029 | 100% | `f11a46a790de292367f3140ebdbca67e726fc7f674be81fd4899951f5ed27ef3` |
| [src/advar/variational.py](src/advar/variational.py) | 9,756 | 100% | `f9a052638b2ebe416ae64c19978c3f60f0576e0f11110551f2e652236c655191` |
| [tests/test_acceptance.py](tests/test_acceptance.py) | 207 | 100% | `fb9b3425089049722818c9211df838cefca596524698c7c7a88b5544a275b515` |
| [tests/test_calibration.py](tests/test_calibration.py) | 435 | 100% | `a283199d689e1add41d2b1c6e89dea380571f88b8aba98aabdf67b377f1fcaef` |
| [tests/test_cli.py](tests/test_cli.py) | 1,521 | 100% | `3c103f1b0844f9d72ac7dfd6b167b730f5c7ad99a33b9749df646f2e8817585d` |
| [tests/test_contract_registry.py](tests/test_contract_registry.py) | 102 | 100% | `9139c95f9b46666ae3b1f671be83803dcb66ecf182ddf9e6b623365f41c4e676` |
| [tests/test_deployment_bundle.py](tests/test_deployment_bundle.py) | 999 | 100% | `5873414fb2ce9c92ffe367829954448a56c215e72e6081dcc365fa6d3ef217ba` |
| [tests/test_ensemble_sensitivity.py](tests/test_ensemble_sensitivity.py) | 335 | 100% | `5b069bd911250e8965a50c2914abaa75e4eeb0d890bfae8325ad440ac5f4b0ba` |
| [tests/test_initial_field_lab.py](tests/test_initial_field_lab.py) | 148 | 100% | `2532218cdcd6e5fbc83c37c2cf808dc3363156ab7ee08c68376227294d290195` |
| [tests/test_initial_field_lab_ui.cjs](tests/test_initial_field_lab_ui.cjs) | 124 | 100% | `ab2d3f7c9896db35e5dfaeb6d779892836a28d638bc92f4c96c70f895a4e7e29` |
| [tests/test_ledger.py](tests/test_ledger.py) | 5,909 | 100% | `51edc836ea35167d9da468b62f7f663b6535e364030b6d4cc15c379017732185` |
| [tests/test_matrix_free.py](tests/test_matrix_free.py) | 290 | 100% | `fa71ecc5af3e0496ca38b1b99a2373cda779300941e85bbe71eb21720e9ec013` |
| [tests/test_metrics.py](tests/test_metrics.py) | 35 | 100% | `3ca87ff5a0d379b03d665a4acf3436839d0de92dd02e6776edf7e28f0614c1b2` |
| [tests/test_nowcast.py](tests/test_nowcast.py) | 6,352 | 100% | `f333d8b0e6feac64ea6d21c1a0ecd34d36c909ec79a6d29352fc63bc70aadb1d` |
| [tests/test_numerical_review.py](tests/test_numerical_review.py) | 287 | 100% | `8383043ea21517ca40a9b93da14a237f0f0a98ec774ace8354c907d674e355ab` |
| [tests/test_pcg.py](tests/test_pcg.py) | 308 | 100% | `8e26822a6c120046a6f2d36e58881106c24bb56e9718f0bdcbb0deec0210933b` |
| [tests/test_promotion.py](tests/test_promotion.py) | 20,794 | 100% | `b865a50e7443faa2066e3ab8b42e6eff2d3d4dd5f48bcf8824cf5c1b8d7df2ac` |
| [tests/test_review_intervention.py](tests/test_review_intervention.py) | 245 | 100% | `3c7d5b119609629f8814f0b7ff4de3eaf7f504f7b6c3441a127b7daae3691780` |
| [tests/test_review_ledger.py](tests/test_review_ledger.py) | 207 | 100% | `f9c79bd567cc3e94a3d38347d17364a141548c4f1001a8badb2793fb9d278e6f` |
| [tests/test_review_legacy_loaders.py](tests/test_review_legacy_loaders.py) | 311 | 100% | `294eb2cbdc201ab727a8955f4ef435b73f9039b8bececcb399c4a39d65e8e4ef` |
| [tests/test_review_numerics.py](tests/test_review_numerics.py) | 145 | 100% | `957dafe8094c0cc6668a6cf6352566e6cdbd53dc67b6809d5c9a4d69ab1f7373` |
| [tests/test_review_promotion_contracts.py](tests/test_review_promotion_contracts.py) | 221 | 100% | `9f3b09f2d7682a5b8935a70e591835847fd549f36b2483b72fd02e26ea723da3` |
| [tests/test_review_promotion_statistics.py](tests/test_review_promotion_statistics.py) | 65 | 100% | `60addb32336c22ba1770531b14c64c25ec1e1842a21125d2f6f2a567026295dc` |
| [tests/test_review_run_identity.py](tests/test_review_run_identity.py) | 191 | 100% | `dcb6fc6c28fc1995d994258d144cac39c4dcdcfd3c9a0a031d8ad88ed64f0da1` |
| [tests/test_run_artifact.py](tests/test_run_artifact.py) | 3,025 | 100% | `fb296a00f8e3b664bb2d13667aebdcb91f53a0fcb45f4ff08e05986b518e4716` |
| [tests/test_runtime.py](tests/test_runtime.py) | 42 | 100% | `d13aa295f84b93155c3f5d6c068852c4fe5a91a6025d43ba5c4537307bd15a7e` |
| [tests/test_runtime_closure.py](tests/test_runtime_closure.py) | 203 | 100% | `28dc0e05e9950ae5e8a2a57f347c4913a09306317caeab4a958153856af0251b` |
| [tests/test_sensitivity.py](tests/test_sensitivity.py) | 8,206 | 100% | `21e79f6c9d021f17a4ebb5c0cab5e16ae737117461a9c69bb9d1974a94de85b4` |
| [tests/test_shift_zero.py](tests/test_shift_zero.py) | 96 | 100% | `7cc885f2397fcdaf6e27d9e995424d90a0fbecc16939a3eb255b50d8698ca8e5` |
| [tests/test_variational.py](tests/test_variational.py) | 6,579 | 100% | `7fa86117af791c60fe00aaba200ac09423820c92f9e091c5bcc68c790266c9d8` |

운영 소스는 조사 기준과 같다. 조사 종료 시 CI에서 확인된 fixture 누락을 `tests/test_ledger.py`, `tests/test_promotion.py`에서 수정했으므로 위 두 파일의 행 수·해시는 수정 전 기준이다. 수정 후 검증과 해시는 `review_artifacts/second_audit_validation.json`에 따로 기록한다.
