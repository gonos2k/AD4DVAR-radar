# G4b zero-face control-space classification

The currently supported 4×5 and 8×10 inverse profiles each use five active
streamfunction controls, including the vertex-coordinate modes `Y` and `X`.
The shared face fluxes are

```text
Qx = psi[1:, :] - psi[:-1, :]
Qy = -(psi[:, 1:] - psi[:, :-1])
```

At the **zero-flow control**, both face arrays are identically zero in value.
However, the coefficient map is `limit * tanh(control)`, with strictly
positive limits. A unit JVP in the `Y` control changes every `Qx` face;
a unit JVP in the `X` control changes every `Qy` face. The new regression
checks this for both grids. Thus none of these nominal zero faces is
structurally zero with respect to the **full supported flow-control space**.
The strict full-control minmod response tracer's refusal of zero flux remains
appropriate; merely finding a zero numerical value does not make its
upwind derivative safe. The result is local to the tested unsaturated zero
control and the two fixed bases, not a claim about all arbitrary bases or
FP64-saturated controls.

As a counterexample to a blanket zero-face ban in every possible control
space, a separate one-mode `Y` basis gives `Qy=0` and `D_control Qy=0`
exactly, while `Qx` and its JVP are nonzero. This restricted basis is **not**
declared eligible for the current full-control implicit response; no solver
gate was relaxed. It shows why future eligibility must be tied to the
requested control space, rather than flux value alone.

The three small tests passed with 18 existing TorchScript warnings; the
new test file typechecked with 0 issues. Logs and source SHA are recorded in
`fv_zero_face_scope_tests.log`, `fv_zero_face_scope_typecheck.log` and the
manifest. No GN, adjoint, signed reanalysis or independent forecast
validation was executed. Under the **current two full-flow-control profiles**
the zero-face question is closed as a justified refusal; support for a new
restricted-control response profile remains future work, not a validation
claim from this test.
