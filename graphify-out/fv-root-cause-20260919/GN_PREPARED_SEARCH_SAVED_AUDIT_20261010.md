# Prepared GN search saved-data audit

Plan SHA-256: `1392032f3f0b859f74f03551a98fa5465216cb7fd981250984b603a495526a1a`; frozen pin checks: **True**.

This audit reads saved JSON only. It does not execute FV, seed preparation, AD, or HVP work.

## Work counts

| Counter | Before | After | Per-plan delta |
|---|---:|---:|---:|
| jacobian_rows_completed | 0 | 24 | 24 |
| jacobian_rows_started | 0 | 24 | 24 |
| dense_solves_completed | 0 | 1 | 1 |
| dense_solves_started | 0 | 1 | 1 |
| hvp_calls_completed | 0 | 2 | 2 |
| hvp_calls_started | 0 | 2 | 2 |
| optimizer_steps_applied | 0 | 1 | 1 |

Archived same-point HVP records verified: 2 / 2; fresh HVP records at new point: 2; HVP counter delta: 2.

Explicit run exit code: 0; child SHA matches saved step: True; current-control byte SHA matches: True.
Frozen source/archive pins: 151/151 sources and 195/195 archives.

## Candidate and progress

Trials recorded: 21; first accepted slot: 21; alpha: 6.968124339867196e-07.
Objective decrease: 2.756148882079391e-06%; residual-merit decrease: 0.00016582794761929953%; control displacement L2: 3.236519866409528e-08.

| # | alpha | J | J slack | R/G² | R/G² slack | paired branch | base trace | accepted | status |
|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|---|
| 1 | 0.7306607947800585 | 0.06061990617317896 | 0.0005723959662689779 | 0.11983927048767742 | -0.11536161939235857 | True | False | False | gate_refused |
| 2 | 0.36533039739002926 | 0.06071720356325307 | 0.0004751870006262787 | 0.025607617374053863 | -0.02112957691826199 | True | False | False | gate_refused |
| 3 | 0.18266519869501463 | 0.060897347343977476 | 0.00029508743211757726 | 0.00793262387864015 | -0.0034543887426117656 | True | False | False | gate_refused |
| 4 | 0.09133259934750732 | 0.06102553715853949 | 0.0001669197236634154 | 0.00522990397125492 | -0.0007515714951082803 | True | False | False | gate_refused |
| 5 | 0.04566629967375366 | 0.061100296215018765 | 9.217172023807257e-05 | 0.004785958080851557 | -0.0003075769346457882 | True | False | False | gate_refused |
| 6 | 0.02283314983687683 | 0.061139868765412955 | 5.26046963708407e-05 | 0.004742604253272913 | -0.0002641987720375808 | True | False | False | gate_refused |
| 7 | 0.011416574918438414 | 0.06116553704217927 | 2.693918286800906e-05 | 0.0056532902810158756 | -0.001174872632265761 | True | False | False | gate_refused |
| 8 | 0.005708287459219207 | 0.06117881403741245 | 1.3663569266568476e-05 | 0.004626837083499212 | -0.0001484133509917062 | True | False | False | gate_refused |
| 9 | 0.0028541437296096036 | 0.06118561041126208 | 6.867886232810039e-06 | 0.004518357384422732 | -3.993061003653049e-05 | True | False | False | gate_refused |
| 10 | 0.0014270718648048018 | 0.06118903627653942 | 3.4423663634042256e-06 | 0.004494022114624371 | -1.5593819298821872e-05 | True | False | False | gate_refused |
| 11 | 0.0007135359324024009 | 0.06119075538428505 | 1.7234313217442954e-06 | 0.004493673029490091 | -1.5243973694868511e-05 | True | False | False | gate_refused |
| 12 | 0.00035676796620120045 | 0.0611916169147978 | 8.619871609827356e-07 | 0.004490134427591656 | -1.1704991561596254e-05 | True | False | False | gate_refused |
| 13 | 0.00017838398310060022 | 0.06119204784071766 | 4.3110441710902947e-07 | 0.004492024508974195 | -1.3594882826716698e-05 | True | False | False | gate_refused |
| 14 | 8.919199155030011e-05 | 0.061192263340122885 | 2.15626599880947e-07 | 0.004492969803810089 | -1.4540082603901536e-05 | True | False | False | gate_refused |
| 15 | 4.4595995775150056e-05 | 0.061192371134896464 | 1.0784262029911451e-07 | 0.004485411438340981 | -6.981669605438459e-06 | True | False | False | gate_refused |
| 16 | 2.2297997887575028e-05 | 0.06119242505963551 | 5.392327825232135e-08 | 0.004485648470352966 | -7.218677852746831e-06 | True | False | False | gate_refused |
| 17 | 1.1148998943787514e-05 | 0.06119245202256289 | 2.696304936961047e-08 | 0.004485766990394765 | -7.337186012206995e-06 | True | False | False | gate_refused |
| 18 | 5.574499471893757e-06 | 0.06119246550418202 | 1.3482779497031583e-08 | 0.004485826251389231 | -7.396441065503356e-06 | True | False | False | gate_refused |
| 19 | 2.7872497359468785e-06 | 0.06119247224502645 | 6.742609685861378e-09 | 0.004485855882150896 | -7.42606885658411e-06 | True | False | False | gate_refused |
| 20 | 1.3936248679734393e-06 | 0.06119247561545381 | 3.372519638555893e-09 | 0.0044858706976188155 | -7.440882839210641e-06 | True | False | False | gate_refused |
| 21 | 6.968124339867196e-07 | 0.061192477301754936 | 1.686387168797765e-09 | 0.004478422389776647 | 7.425745603487077e-09 | True | True | True | accepted |

The archived first-order residual model predicts a G² decrease of 7.426458117736068e-09; the actual decrease is 7.426488249015484e-09 (actual/model 1.0000040572880016). Residual-model error is 8.298536672047055e-12 L2, or 0.00013945027280464476 of the actual residual-vector change.

## Closure checks

- frozen_plan_and_preflight_pins_match: **True**
- all_frozen_source_and_archive_hashes_match: **True**
- run_completed_with_explicit_zero_exit: **True**
- run_child_sha_matches_saved_step: **True**
- base_control_matches_plan: **True**
- current_control_tensor_byte_sha_matches: **True**
- search_used_at_most_24_trials: **True**
- at_most_one_step_applied: **True**
- postcommit_readiness_complete_or_no_commit: **True**
- source_input_runtime_closed: **True**
- two_prior_hvps_reused_and_two_new_hvps_computed: **True**
- all_candidate_G_norms_and_mixed_gradients_close: **True**
- all_candidate_paired_branches_passed: **True**
- search_stopped_at_first_passing_dyadic_slot: **True**
- saved_residual_model_math_closes: **True**
- postcommit_24row_solve_hvp_receipt_closes: **True**
- P2_individual_side_gradients_and_theta_close: **True**

All checks passed: **True**.
