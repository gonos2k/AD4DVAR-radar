# Forecasts compared with the independent semidiscrete reference

The three saved forecast arrays and RED's independent reference have identical
shape `[18, 4, 5]`. Their absolute times match exactly: 1,800 through 12,000
seconds, corresponding to +10 through +180 minutes after the analysis endpoint
at 1,200 seconds. All three point hashes match the sealed forecast receipt;
the signed 96108 forecast matches RED's stored candidate forecast exactly
(maximum absolute difference 0). The archived 4×5 terminal truth in the
reference archive also matches the pinned forecast case exactly.

| Point | Mean 18-lead MSE (dBZ²) | Mean 18-lead MAE (dBZ) | Terminal MSE (dBZ²) | Terminal MAE (dBZ) |
|---|---:|---:|---:|---:|
| 2a4d | 2.43440523937553 | 1.1628535684080816 | 7.384569591781805 | 2.380372460313088 |
| fb0a | 2.4343526122190773 | 1.1628441219008459 | 7.384327740910282 | 2.3803275353648186 |
| 96108 | 2.4343452362760383 | 1.1628438583994622 | 7.384255623387801 | 2.380312275343212 |

RED's reference archive contains a persistence forecast made from the 96108
analysis-end frame. Its mean 18-lead MAE is 1.14586846934291 dBZ and terminal
MAE is 1.4791990812913958 dBZ. Thus 96108's forecast is worse against this
reference by 0.016975389056552 dBZ mean MAE and 0.901113194051816 dBZ terminal
MAE. The forecast first becomes worse than persistence at +110 minutes: its
MAE is lower at +100 minutes, then exceeds persistence at +110 minutes. There
is no comparable analysis-frame persistence array for 2a4d or fb0a.

These numbers measure discrepancy from the independent semidiscrete
reference, not observed forecast skill. The reference is not continuous-PDE
truth or held-out weather. The run's original 4×5 terminal truth supports the
three terminal E scores, but no 18-lead truth sequence is available for
lead-wise truth-error or persistence-skill claims. Physical-run accounting is
per producer: GREEN's score observer made three forwards; RED ran four
exploratory candidate forwards and one final reference-producer candidate
forward (five total); this comparison made zero new forwards. The 12×12
adjacent analytic cases are not counted as current forecasts.

The separate cached-algebra proof in
[`PHYSICAL_RECOVERY.json`](../../PHYSICAL_RECOVERY.json) gives the known
generator a prior-cost lower bound of 0.9421520472 for this J, versus the
accepted 96108 J of 0.0609882086 (15.45×). It performs no FV or score calls
and does not turn this reference comparison into a forecast-skill claim.
