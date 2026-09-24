from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.matrix_free import observe_pcg_calls, pcg  # noqa: E402
from advar.local_refinement import pcg as refinement_pcg  # noqa: E402
from examples.weather_scenarios import fv_partial_reanalysis_probe as partial_probe  # noqa: E402


def _system():
    diagonal = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
    rhs = torch.tensor([1.0, 2.0, 7.0], dtype=torch.float64)
    expected = rhs / diagonal
    return diagonal, rhs, expected


def _solve(solve=pcg):
    diagonal, rhs, _ = _system()
    return solve(lambda value: diagonal * value, rhs, rtol=1.0e-13)


def _record(calls, label=None, events=None):
    def factory(original):
        def wrapped(*args, **kwargs):
            calls.append((args[1].clone(), kwargs.copy()))
            if events is not None:
                events.append(label)
            return original(*args, **kwargs)

        return wrapped

    return factory


def test_observed_and_unobserved_spd_solve_have_exact_same_result():
    diagonal, rhs, expected = _system()
    unobserved = _solve()
    calls = []

    with observe_pcg_calls(_record(calls)):
        observed = _solve()

    assert len(calls) == 1
    for result in (unobserved, observed):
        assert result.converged
        torch.testing.assert_close(result.solution, expected, rtol=1.0e-14, atol=1.0e-15)
        residual = rhs - diagonal * result.solution
        true_relative_residual = float(
            torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(rhs)
        )
        assert true_relative_residual > 0.0
        assert result.relative_residual == pytest.approx(
            true_relative_residual, rel=1.0e-14, abs=0.0
        )
    torch.testing.assert_close(observed.solution, unobserved.solution, rtol=0.0, atol=0.0)
    assert observed.iterations == unobserved.iterations
    assert observed.relative_residual == unobserved.relative_residual


def test_overlapping_thread_observers_are_independent_and_skip_ordinary_caller():
    started = Barrier(3)
    release = Event()
    calls = [[], []]

    def worker(index):
        with observe_pcg_calls(_record(calls[index])):
            started.wait(timeout=15.0)
            assert release.wait(timeout=15.0)
            return _solve()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, index) for index in range(2)]
        started.wait(timeout=15.0)
        try:
            ordinary = _solve()
            assert calls == [[], []]
        finally:
            release.set()
        results = [future.result(timeout=15.0) for future in futures]

    _, _, expected = _system()
    assert [len(thread_calls) for thread_calls in calls] == [1, 1]
    for result in [ordinary, *results]:
        assert result.converged
        torch.testing.assert_close(result.solution, expected, rtol=1.0e-14, atol=1.0e-15)


def test_nested_observers_run_inner_before_outer():
    events = []
    outer_calls = []
    inner_calls = []

    with observe_pcg_calls(_record(outer_calls, "outer", events)):
        _solve()
        with observe_pcg_calls(_record(inner_calls, "inner", events)):
            _solve()
        _solve()

    assert len(outer_calls) == 3
    assert len(inner_calls) == 1
    assert events == ["outer", "inner", "outer", "outer"]


def test_observer_context_is_cleaned_up_after_exception():
    calls = []

    with pytest.raises(RuntimeError, match="sentinel"):
        with observe_pcg_calls(_record(calls)):
            _solve()
            raise RuntimeError("sentinel")

    _solve()
    assert len(calls) == 1


def test_preimported_local_refinement_pcg_alias_uses_active_observer():
    calls = []

    with observe_pcg_calls(_record(calls)):
        result = _solve(refinement_pcg)

    _, _, expected = _system()
    assert len(calls) == 1
    assert result.converged
    torch.testing.assert_close(result.solution, expected, rtol=1.0e-14, atol=1.0e-15)


def test_partial_reanalysis_monitor_receives_pcg_operator_calls():
    report = {"linear_solves": []}
    with observe_pcg_calls(
        lambda original: partial_probe._monitor_pcg(original, report, "analytic_phase")
    ):
        result = _solve(refinement_pcg)
    assert result.converged
    assert len(report["linear_solves"]) == 1
    entry = report["linear_solves"][0]
    assert entry["phase"] == "analytic_phase"
    assert entry["hvp_calls"] >= result.iterations
    assert entry["iterations"] == result.iterations
    assert entry["pcg_relative_residual"] == result.relative_residual
