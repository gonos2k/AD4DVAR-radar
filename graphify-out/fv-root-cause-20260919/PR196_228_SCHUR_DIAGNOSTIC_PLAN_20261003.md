# PR196–228 S1 fresh Schur block diagnostic at PR227 endpoint

Pin raw PR227 field-corrected endpoint/controla78d1b8e..., parameterse3a45fac..., input838fcf77..., all21 originalsourcefiles andcurrentnewproducer/test/runtime. ReevaluateJ/fullgradient/54stagebranch andarchiveidentity beforeafter. Initialfieldgmax7.03e-12 isnotzero and fullgmax0.006128 remainsnonstationary. LaterR4-A selectedactiveface result isadifferentpoint/contract; it doesnotreplace thisrequestedblockdiagnostic.

Exactlyone fresh26column truegradient-JVP Hessian assembly. No optimizercandidate, parameterperturbation, adjoint or response. Checkfinite/numericallysymmetricH; positivewell-conditionedHff20; solve Hff*X=Hfd and Hff*y=gf withindependentbackward residualgates. S=Hdd−Hdf*X (6variables), rhs=−gd+Hdf*y; useitsownscale for symmetry/condition/eigen diagnostics. Dense6solve onlyifnonsingular/conditioneligible; indefiniteS may bealgebraicallydiagnosed butmustnot be calledminimization/SPD/CGeligible. No CG isused here. Reconstructdf=−Hff^-1(gf+Hfd*dd); comparecomplete26step withdirect26solve andverifyHstep+g/fullandblockresiduals1e-10. No unrelatedunitdenominatorfloors.

RecordH/Hff/S spectra, conditions, eliminatedRHS, fieldresponsecoupling, full-step parity,residuals, unchangedsource/input/branch/runtime andallscientificrefusals. Hffnonpositive/unsafeS givesstructuredrefusal; programming/invariantfailures propagate. Any step islinearization only, notappliedoraccepted; neitherSchur norlocalpositivecurvatureprovesoriginalsmoothroot/normal/physicalskill.

One CPUFP64 child under120s wall andsampled1GiB/.25s; no automatedretry after scientific/resource refusal. Preserveworkerlog/raw/resource/preflight; parentassignsexecutionstatusafterexit. Source/test/planhashesbeforeafter. No wholeCPU/package/deploymentCI.

Raw SHA256: 8ad5b5e3d56a074ad79a28fa0f23c0cbdae745c26efed5101cd826fe2d9442cf
