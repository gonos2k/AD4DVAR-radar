# PR166 follow-up: local stationary-path validation

Baseline: merged main `bc63dc05`. Reuse the saved 26-control nominal root;
no nominal optimizer, original failed run, 180-minute or 240-grid rerun.

For F(c,p)=grad_c J, compute the exact parameter JVP F_p d and solve
H c_dot = -F_p d using the same robust Hessian. The full B=y[0]+theta*pattern
contract participates. Both signed predictors c0±h*c_dot are corrected with the
existing dense Newton oracle and its unchanged max-gradient1e-10 requirement.
The direction is raw sin(k), k=0..59, followed by zero theta component; it is
neither normalized nor sin(2*pi*k/60). h follows the predeclared .001*2^-j order.

Every predictor/accepted endpoint uses all54 actual RK-stage limiter selections
and face signs. This is local branch evidence; equal endpoints and checked
Newton iterates are not certification of every intervening parameter/state path.
The original .001 finite-impact experiment remains separate. Public minmod
response eligibility and finite-path certification stay false.

The conditional score target is the frozen nominal forecast+.1*pattern. No
independent forecast skill, held-out improvement, typed prior learning, or
production GN convergence claim follows from this experiment.

The diagnostic timer now captures monotonic start once; a small AST regression
with mocked timestamps10 and13.25 returns3.25s. No historical JSON was overwritten.

Before examining new results, derivative agreement was set separately to
`abs(central-adjoint) <= 1e-4*abs(adjoint)` (nonzero directional signal).
At h=.0005, both endpoints converge on the nominal branch but relative error
is5.5567230e-4, so this is a structural success and derivative-tolerance failure.
The predeclared halving sequence is continued; neither tolerance nor direction
is changed. Coarse results and their measured producer are retained, and the
saved H/tangents/adjoint are reused for smaller sizes.

## Completed local result

| h, dBZ amplitude | Central reanalysis | Relative difference to adjoint | Max endpoint gradient |
|---|---:|---:|---:|
| .0005 | -7.56212484e-4 | .0555672% | below1e-10 |
| .00025 | -7.55897503e-4 | .0138916% | below1e-10 |
| .000125 | -7.558187595153344e-4 | **.00347295%** | 8.4694327e-11 |
| .0000625 | -7.557990734183978e-4 | **.000868254%** | 7.9231140e-12 |

The adjoint directional response is -7.557925112229098e-4. The last two successive
pairs pass the predeclared .01% relative criterion. Their absolute differences
are2.6248292e-8 and6.5621955e-9. All four final endpoints retain all54 RK-stage
selectors and face signs; minimum normalized slope margins are7.52655e-4 and
7.57106e-4. This closes the tested observation-direction local-response task.
It does not close background-parameter reanalysis: only its tangent was cached.

The coarse run cost261.15s, sampled peak RSS339673088bytes. Resuming at j=3 reused
H/tangents/adjoint and cost158.40s, sampled peak357236736bytes. These are bounded
probe costs, not whole-chain D7. An initial archive-relative-import startup failure
is retained separately; no optimizer ran during that failed startup.

RED requested cheap checks on reused linear solves. Rechecking the exact stored
systems gives tangent relative residuals2.71116e-16 (observation) and2.59621e-16
(theta), adjoint2.65293e-16, and relative Hessian asymmetry2.37772e-16. These checks
are now enforced before using a cache; corrupt tangent/adjoint/asymmetry/nonfinite
regressions are included. The measured producer snapshots precede these validation
additions; unchanged numerical experiments were not rerun for validation-only code.

The resumed diagnostic's structural suffix counter originally reported3 instead
of4 because its initialization stopped at a derivative-tolerance failure. Its two
passing derivative pairs were counted correctly. The counter logic is fixed and
regressed. `minmod_local_path_raw.json` preserves the exact run output; the published
report corrects only this metadata, with the raw SHA256 and reason attached. No
control, score, derivative, timing or branch array was changed.

## Verification and display

47 distinct focused cases passed in6.84s, followed by one new counter regression:
**48 distinct focused cases** total. The final publisher/cache rerun is not added
again. No full CI or unchanged long experiments were run. GREEN and RED reviewed
mathematical scope, resume identity and cache validation.

The original HTML frame-data hashes remain unchanged. The separate local-response
table was checked in Aside desktop DOM and screenshot. Mobile viewport emulation
was unavailable in the installed Aside REPL; no new mobile verification is claimed.
The existing horizontally scrollable table container is retained. The original
.001 failure and all general-response/learning limits remain visible.
