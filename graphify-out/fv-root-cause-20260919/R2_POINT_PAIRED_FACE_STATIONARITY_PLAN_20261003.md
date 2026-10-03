# Original R2 two-face tangent correction plan (2026-10-03)

Preserve the original26-control zero-centered point problem,13parameters, all4 detected points at3times, symmetricfixed correlation, externalB(theta), full model/boundary support, growth/time/score. Start only from the last accepted control of the completed one-face numerical refusal (control SHA11c1ddcb77e65974d13de583d3ff48acea496af17f285c2f46f4575a267d4797). This is a new projection onto qx[3,4]=qy[3,0]=0, not acceptance of an archived refused candidate.

The coefficient rows [1,0,4,-3.5,16] and [0,-1,-3,-.5,-3] have invertible originalpivot0/1 blockdiag(1,-1). Compose the existing face charts, retaining originalopen domains/finite-precision representability checks. Use3 structurally projected streamfunction modes with rank3; exactvertex equality must give both selectedzeros. The24-control objective retains all26 prior terms including both recovered original pivot latents. All pointlikelihood/geometry/whitening and verification fields remain unchanged.

One fixed-other-branch Newton–PCG correction: max4Newton/max16backtracks/max104PCG, originalgradient-merit Armijo, actuallinear relative residual1e-10, fresh tangent max<1e-10. Require exacttwozero faces in every54RK stage; all47 otherfaces and allstrict interiorlimiter margins keep128eps resolvability and1e-4 margins. Check36analysis/54forecastreplay equality, backgroundtransform/detected observation validity, unchanged p and fullsignature. Changedotherbranches refuse; no automaticsector expansion or margin/tolerance relaxation.

Audit24HVP columns for initialand qualified-final numerical symmetry/SPD; dense matrix isdiagnostic only. Existing exactHVP Newton and PCG remain the solver. Save everycandidate,lastacceptedvector,actualPCG RHS/solution/residual; independently recheck rowbranch/budget and freshHs+g. Known post-chart branch/domain/numerical failures stay terminalrefusals; archive/source/input/runtime/chart-conversion preflight failures or invariant/program errors propagate and parentclassifies failed/not-reached. No worker self-declared execution success.

Outcome is tangentstationarycandidate only ifallfresh gates pass. No full26 smoothroot, normalminimum, adjoint/VJP, nonlinear parameterreanalysis, uniqueness/globalminimum, uniformparameterball or physicalskill claim. Four inwardone-sided normal conditions and selected tangent-response validation require separately declared subsequent work.

One serialCPUFP64 child:300s wall, sampled1GiB RSS/.25s. Pin both source/test/plan and originaloneface report/control/input/runtime beforeafter; parent verifies childexit/phase/source/input/resource and numericalstatus separately. Preserve numericalrefusal; no automatic rerun or extra starts. No wholeCPU/package/deploymentCI request.

Prior report SHA256: b26f361becce01db9782138b0832c470dc9ec026322e0b7a816cc50a9867f194
