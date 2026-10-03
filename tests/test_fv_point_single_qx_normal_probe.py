"""Pure single-qx normal audit/wiring tests; no interval evaluation or FV model call."""
from typing import Any

import pytest

from examples.weather_scenarios import fv_point_single_qx_normal_probe as probe


def _native() -> dict[str, Any]:
    qx=[[1]*6 for _ in range(4)]; qy=[[1]*5 for _ in range(5)]
    qx[3][4]=0
    return {"step":0,"stage":0,
            "x":{"slope_sign":[[1,0,-1],[1,1,1]],"choose_left":[[True,False,False],[True,False,True]]},
            "y":{"slope_sign":[[1,1,1],[-1,0,1]],"choose_left":[[True]*3,[False,False,True]]},
            "qx_sign":qx,"qy_sign":qy}


def _branch(side: int) -> dict[str, Any]:
    native=_native(); native["qx_sign"][3][4]=side
    return {"step":0,"stage":0,"x":[1,0,2,1,2,1],"y":[1,1,1,2,0,1],
            "qx_sign":[v for row in native["qx_sign"] for v in row],
            "qy_sign":[v for row in native["qy_sign"] for v in row]}


def test_native_36_row_audit_checks_all_48_other_faces_and_both_limiter_bins():
    native:list[dict[str,Any]]=[_native() for _ in range(36)]
    branch:list[dict[str,Any]]=[_branch(-1) for _ in range(36)]
    for i,row in enumerate(native+branch): row["step"],row["stage"]=i%36//2,i%2
    probe.audit_trace(native,branch,-1)
    branch[9]["qy_sign"][0]=-1
    with pytest.raises(probe.Refusal,match="48 other"):
        probe.audit_trace(native,branch,-1)


def test_bad_trace_shape_is_a_malformed_capture_not_an_unsupported_normal():
    with pytest.raises(ValueError,match="36 native"):
        probe.audit_trace([_native()],[],1)


def test_capture_control_and_parameter_hash_are_pinned_to_release():
    release:dict[str,Any]={"last_accepted_tangent":[0]*25,"final_objective":1.0,"final_branch":{"signature_sha256":"s"},
             "input_before":{"tensor_sha256":{"parameters":"p"}},"source_after":{},"final_trace":[1]*36}
    paired:dict[str,Any]={"input_before":release["input_before"],"input_after":release["input_before"]}
    data:dict[str,Any]={"release_report_sha256":probe.RELEASE_SHA,"paired_report_sha256":probe.PAIRED_SHA,
          "tangent":[0]*25,"native_objective":1.0,"native_signature_sha256":"s",
          "input_identity":release["input_before"],"parameters_sha256":"p","native_analysis_trace":[1]*36,
          "release_source_sha256":{}}
    probe.verify_capture(data,release,paired)
    data["tangent"][20]=1
    with pytest.raises(ValueError,match="released native point"):
        probe.verify_capture(data,release,paired)


def test_binary_flux_contract_requires_exact_qx_zero_and_point11_derivative():
    zero={"lower":[0,0,0,0],"upper":[0,0,0,0]}
    point11={"lower":[0,7926335344172073,-56,53],"upper":[0,7926335344172073,-56,53]}
    point0={"lower":[0,0,0,0],"upper":[0,0,0,0]}
    probe.audit_flux({"selected_flux_values_binary":[zero,point0],
                      "selected_flux_derivatives_binary":[point11,point0]})
    with pytest.raises(ValueError,match="face contract"):
        probe.audit_flux({"selected_flux_values_binary":[point11,point0],
                          "selected_flux_derivatives_binary":[point11,point0]})


def test_unknown_reference_exception_propagates_unchanged(monkeypatch):
    def broken(*_args,**_kwargs): raise RuntimeError("callback bug")
    monkeypatch.setattr(probe.reference,"evaluate",broken)
    with pytest.raises(RuntimeError,match="callback bug"):
        probe.evaluate_side({},[],1)
