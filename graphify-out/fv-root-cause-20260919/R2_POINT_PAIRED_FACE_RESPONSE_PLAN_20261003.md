# Original R2 selected paired-face response plan (2026-10-03)

Prerequisites: source/input/runtime-bound24D tangent root with freshmaxgradient<1e-10, numericalSPD exactHVP Hessian and strict54-stage remainingbranch; both selectedfluxes identicallyzero; independentone-sided IV signs strictlyleftnegative/rightpositive forBOTH normals and consistent acrossfourorthants. Ifany prerequisitefails, publishno response. Original26prior,13p,pointgeometry/fixedquality/std/correlation/background/boundaries/time and frozenverification remain unchanged. This is a selectedpaired-face conditionalresponse, not the old classical26Droot API.

Use existing compute_local_response on the24D bound scalarobjective/score. Return13-vector direct/indirect/total and trueH^Tlambda−E_t residual<=1e-10; no denseHessian for solves. Direction d=(middle4 ones;otherp zero), directterm0 becauseexternalBgdepends onlytheta. Fresh mixedJ_tp*d and matrix-free predictor solveHtt*dt=−J_tp*d, actualresidual1e-10/max104CG.

Actualsignedreanalysis at h=.00025,.000125dBZ, bothsigns, startt0±h*dt. Pass actualchangedp toobjective/refiner/branch/score/independentnormalfixture. Max2Newton/max16backtracks/max104CG withoriginalgradient-merit Armijo. Fresh endpointgradient<1e-10, same54-stage nonselectedbranch, originalcoefficient/chartdomains andremaining1e-4 margins. At EACHendpoint auditfresh24HVP numericalSPD and itsOWN eightIV normalevaluations/allfourstrict signs. Unsupportedendpoint remainsunsupported, no cachednominalnormalcertificate.

Centralfinite-difference score mustmatchconditionaladjoint atbothsizes withrelativeerror<=1e-4 andsmallerabsoluteerror atsmallerh. Preserve allcomponents, absoluteandrelativeerrors, actualPCG/residual/counters, endpoints/control/p/J/E/branch/domain/normal evidence andsource identities. ParameterlinearVJP consistency alone is not nonlinearresponse validation. No freshGN or extra directions/automatichshrink.

One serialCPUFP64 child600s wall/sampled1GiB/.25s; pinbaseline/root/normal/input/plan/producer/reference/test/kernel sources beforeafter; parentpublishes only afterchildexit/finishedphase/source/input/resource audit andGREEN/RED review. No fullCPU/package/deploymentCI. Exactroot/uniformpball/globalminimum/uniqueoptimizer/physicalskill/finiteimpact validity remain unproved.

Paired root report SHA256: d38d1cec4550a6347ceb14db23395679a59a7cad89f09096ec7185f984f6a8cf
