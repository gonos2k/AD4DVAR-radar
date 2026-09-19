"""Small regression checks for the local FV observability probe."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "weather_scenarios"))
from fv_observability_probe import run_probe


def test_clear_and_constant_cases_have_unobservable_flow(tmp_path):
    report = run_probe(tmp_path / "observability.json")
    cases = {case["case"]: case for case in report["cases"]}
    for name in ("knownclear", "knownconstant_inflow"):
        case = cases[name]
        assert case["conditional_flow_jacobian"]["rank"] == 0
        assert case["projected_flow_jacobian"]["rank"] == 0
    assert cases["knownclear"]["conditional_dynamics_jacobian"]["rank"] == 0
    assert (tmp_path / "observability.json").exists()


def test_asymmetric_case_reports_local_flow_observability_without_full_rank_claim():
    report = run_probe()
    case = next(
        case
        for case in report["cases"]
        if case["case"] == "asymmetricpositive_knownzero_exterior"
    )
    assert case["conditional_flow_jacobian"]["shape"] == [126, 8]
    assert case["projected_dynamics_jacobian"]["shape"] == [126, 9]
    assert report["claims"]["flow_recovered_from_data"] is False
    assert report["claims"]["full_d7_128x128"] is False
