# GREEN final audit: same-point direction comparison

The guarded comparison completed within its frozen scope. I independently
checked the saved plan and source bundle, source and receipt closure, P2/full-G
and Armijo records, arm selection, and resource receipt. This was a saved-data
audit; I ran no FV objective, forecast, gradient, HVP, or candidate search.

The plan SHA-256 is
`36686ccf05c55ab9dff1d6c0c4658e14049a108eba402b76c86fc0b9ac904b35`; the
209-member source bundle hash is
`556fec30ca5b0a06c0b00601c8652b616b02705a09d7ac82fcea3203fd0c46bc`.
All source member hashes match the 209 pinned paths, and the plan contains
283 archive pins. Read-only plan admission passed. The run receipt's child
hash matches `attempt1/step.json`, and its resource object matches the
separate resource receipt. Source hashes, runtime, and immutable archived
input fields close; control SHA changes from the admitted base by design.

The run closed at base `b647…88660` under `minmod_net_flux_fp64_v1` with
source/input/runtime closure, `deadline_passed=true`, no guard termination,
133.640 seconds elapsed, and 1,119,698,944 sampled peak RSS bytes under the
660-second / 2-GiB outer caps. Both arms independently passed current P2,
repeated Armijo, and full-G checks. Work counts were four completed HVPs,
24 GN Jacobian-row VJPs, one dense solve, and 11 candidate evaluations
(one baseline, ten GN). One candidate was committed; no terminal E or
postcommit readiness evaluation occurred.

The baseline tangent arm won the frozen repeated-residual-first selector:
its repeated `R` was `0.002559866170789153`, versus `0.002567751483553327`
for robust GN. GN had lower repeated `J` (`0.06100086445604125` versus
`0.061002341038211755`), but the selector uses `R` first and only uses `J` for
scale-aware `R` ties. The final confirmed control is
`d19e882d31b634ef0be755e6419b57ecf3691734efc736d5f40ea18c3ac4145e`.

The direction-efficiency statistic `chi` was about `0.00485` for the baseline
and `0.87468` for GN. Those values did not predict which arm would pass the
finite candidate gates: the first nine GN trials passed objective Armijo but
failed residual Armijo, and its tenth trial passed both. The reported residual
model-error / actual-delta norms were about `0.418` for the selected baseline
candidate and `1.001` for the selected GN candidate. These are local
linearization comparisons at the tested candidates; they do not establish
whole-model Taylor accuracy, explain the error source, or support a general
claim that GN is inferior. Candidate branch signatures differed from the base
in the recorded trials, and no additional operator or postcommit evaluation
was performed.

Final packaging check: the receipt manifest lists exactly 16 members; all
archived hashes and byte lengths match both the manifest and retained
`attempt1` files. The receipt tar hash matches its manifest and is 483,331
compressed bytes (58,507,151 uncompressed member bytes). Findings, metrics,
verification, and `verify_result.py` stay within stored-run receipts and
saved-vector arithmetic; no FV/operator rerun was performed.
