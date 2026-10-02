# Normal derivative attempt2: fix setup double wrapping

Attempt1 exited1 before Euler evaluation because `_JetMath.mpf` already returns an IntervalJet, while setup wrapped it with a helper expecting a raw interval. The original source/test/harness bytes and log/resource/preflight are preserved. This is a programming/setup failure, not a failed mathematical normal condition.

One separately recorded attempt2 is authorized after the helper accepts existing same-side jets without losing derivative/event metadata. Seven focused tests verify scalar calculus, side slopes, ambiguity/untagged refusal, event-tag isolation and idempotent wrapping. Keep the same pinned input5694b56e...174e, original plan, 80 digits, two side extensions, all other branch audits and30s/1GiB guard. No optimization, response, changed input or numerical tolerance relaxation. No automatic retry after a numerical uncertainty.
