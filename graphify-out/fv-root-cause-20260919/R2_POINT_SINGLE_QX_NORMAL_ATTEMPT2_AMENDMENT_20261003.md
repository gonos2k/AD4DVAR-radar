# Single-qx normal setup correction

Attempt1 failed with exit1 before returning any interval result: `_mp` accepts int/float/decimal strings, but the single-qx reference passed fixed Fraction basis weights. The weights (1,0,4,-7/2,16) are all exactly binary64-representable; converting these fixed constants to float preserves their exact values and their interval representation. No parameter, basis, prior, branch, normal direction, dps80, tolerance or resource policy changed.

Preserve attempt1 parent/log/preflight/source snapshots. One separate attempt2 under the original60s/256MiB guard uses the corrected source. A complete captured-fixture negative-side interval regression now reaches all36 analysis stages, alongside the10 chart/audit tests (11passed). This is a programming-input correction, not a retry of a scientific normal refusal; attempt1 numerical status is not_reached.
