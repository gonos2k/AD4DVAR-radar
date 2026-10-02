# Normal derivative attempt3: fix inactive-slot arithmetic type

Attempt2 exited1 in whitening because inactive standardized slots used raw interval0 as a left operand against a derivative Jet. It produced no completed normal result. Preserve exact source/test/harness/preflight/log/resource. This is a reference implementation error, not a failed normal optimality inequality.

Use a zero Jet for inactive slots, preserving zero derivative and the fixed missing mask. The first-order mathematical expression, captured input, two upwind extensions, 80 digits, source branch audits and30s/1GiB guard are unchanged. Eight focused tests now include inactive zero whitening. Authorize one separately recorded attempt3 after code/tests/type freeze; no optimization, response or tolerance change. Any numerical ambiguity remains a refusal, not a retry trigger.
