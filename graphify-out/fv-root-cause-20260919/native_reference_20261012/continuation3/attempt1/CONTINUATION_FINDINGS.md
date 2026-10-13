# Native Powell continuation

The transport support-only fix was applied under a new sealed plan `67fa44ac8c6c08f79df99a9fe2c5bee10b385a7769bb65ad6caafafeb9c52a01`. The original search plan and logs remain unchanged. Source transition is restricted to `src/advar/transport.py`; the echo update and minmod reconstruction are unchanged.

Cached replay of zero-start SciPy Powell consumes the previous 699 successful native-J samples and exactly reproduces callback 700. Under the fixed support source, that point now evaluates to J=15.70867402864837. The run then reaches a cumulative-call budget refusal with zero-start best-so-far J=8.05918166711018 at callback 997; it is not a convergence or stationarity result.

The fb0a start completed its original 1,000 Powell calls at J=0.06070843108138. Three same-point native-J checks reproduce the old fb0a J, zero-start J, and prior zero checkpoint J exactly. Across old completed evaluations, those checks, and continued zero-start calls, the successful native-J total is 2,000.

The recovered callback-700 control, zero checkpoint, per-evaluation logs, and source/input pins are preserved in this attempt. The continuation made no Q/R/E/HVP calls; HVP status from the earlier aborted partial attempt remains unknown. Search used 65.686s in the guarded continuation at sampled peak RSS 369983488 bytes.
