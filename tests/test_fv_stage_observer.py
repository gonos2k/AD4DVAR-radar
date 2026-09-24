"""Context isolation checks for the optional minmod-stage observer."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest
import torch

from advar import transport
from examples.weather_scenarios import fv_minmod_inverse_probe as probe


def _stage_inputs(variant: int = 0):
    dtype = torch.float64
    height, width = 4, 5
    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing="ij",
    )
    q = (
        2.0 + 0.1 * x + 0.2 * y + 0.01 * x * y
        + 0.005 * x.square() + 0.009 * y.square()
        + variant * (0.013 * x + 0.007 * y.square())
    )
    qx = torch.full((height, width + 1), 0.17 + 0.03 * variant, dtype=dtype)
    qy = torch.full((height + 1, width), -0.11 - 0.02 * variant, dtype=dtype)
    edges: transport.BoundaryEdges = (
        torch.full((height,), 1.5 + 0.1 * variant, dtype=dtype),
        torch.full((height,), 1.5 + 0.1 * variant, dtype=dtype),
        torch.full((width,), 1.5 + 0.1 * variant, dtype=dtype),
        torch.full((width,), 1.5 + 0.1 * variant, dtype=dtype),
    )
    return q, qx, qy, edges, q.new_tensor(0.07), q.new_tensor(1.3)


def _run_stages(variant: int):
    q, qx, qy, edges, dt, area = _stage_inputs(variant)
    captured = []

    def save(stage_q, stage_qx, stage_qy):
        captured.append(tuple(
            tuple(float(value) for value in tensor.reshape(-1))
            for tensor in (stage_q, stage_qx, stage_qy)
        ))

    with transport.observe_minmod_stages(save):
        for _ in range(3):
            q = transport._euler_minmod(q, qx, qy, edges, dt, area)
    return q, captured


def test_nested_observers_are_called_inner_then_outer_and_preserve_stage_output():
    q, qx, qy, edges, dt, area = _stage_inputs()
    expected = transport._euler_minmod(q, qx, qy, edges, dt, area)
    calls = []

    def outer(stage_q, stage_qx, stage_qy):
        calls.append(("outer", stage_q, stage_qx, stage_qy))

    def inner(stage_q, stage_qx, stage_qy):
        calls.append(("inner", stage_q, stage_qx, stage_qy))

    with transport.observe_minmod_stages(outer):
        with transport.observe_minmod_stages(inner):
            actual = transport._euler_minmod(q, qx, qy, edges, dt, area)

    assert [call[0] for call in calls] == ["inner", "outer"]
    for _, seen_q, seen_qx, seen_qy in calls:
        assert seen_q is not q and torch.equal(seen_q, q)
        assert seen_qx is not qx and torch.equal(seen_qx, qx)
        assert seen_qy is not qy and torch.equal(seen_qy, qy)
    assert torch.equal(actual, expected)


def test_mutating_a_diagnostic_snapshot_does_not_change_transport():
    q, qx, qy, edges, dt, area = _stage_inputs()
    expected = transport._euler_minmod(q, qx, qy, edges, dt, area)
    original_q, original_qx, original_qy = q.clone(), qx.clone(), qy.clone()

    def mutate(stage_q, stage_qx, stage_qy):
        stage_q.zero_()
        stage_qx.zero_()
        stage_qy.zero_()

    with transport.observe_minmod_stages(mutate):
        actual = transport._euler_minmod(q, qx, qy, edges, dt, area)
    assert torch.equal(actual, expected)
    assert torch.equal(q, original_q)
    assert torch.equal(qx, original_qx)
    assert torch.equal(qy, original_qy)


def test_nested_observers_receive_independent_snapshots():
    q, qx, qy, edges, dt, area = _stage_inputs()
    outer_values = []

    def outer(stage_q, stage_qx, stage_qy):
        outer_values.append((stage_q, stage_qx, stage_qy))

    def inner(stage_q, stage_qx, stage_qy):
        stage_q.zero_()
        stage_qx.zero_()
        stage_qy.zero_()

    with transport.observe_minmod_stages(outer):
        with transport.observe_minmod_stages(inner):
            transport._euler_minmod(q, qx, qy, edges, dt, area)
    assert len(outer_values) == 1
    assert torch.equal(outer_values[0][0], q)
    assert torch.equal(outer_values[0][1], qx)
    assert torch.equal(outer_values[0][2], qy)


def test_observer_context_resets_after_exception():
    q, qx, qy, edges, dt, area = _stage_inputs()
    calls = []

    def fail(*_):
        calls.append("called")
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        with transport.observe_minmod_stages(fail):
            transport._euler_minmod(q, qx, qy, edges, dt, area)

    expected = transport._euler_minmod(q, qx, qy, edges, dt, area)
    with transport.observe_minmod_stages(lambda *_: calls.append("unexpected")):
        actual = transport._euler_minmod(q, qx, qy, edges, dt, area)
    assert calls == ["called", "unexpected"]
    assert torch.equal(actual, expected)


def test_concurrent_observers_match_serial_signatures_without_cross_talk():
    serial = {variant: _run_stages(variant) for variant in (0, 1)}
    overlap = Barrier(2)

    def run(variant):
        q, qx, qy, edges, dt, area = _stage_inputs(variant)
        captured = []

        def record(stage_q, stage_qx, stage_qy):
            captured.append(tuple(
                tuple(float(value) for value in tensor.reshape(-1))
                for tensor in (stage_q, stage_qx, stage_qy)
            ))
            if len(captured) == 1:
                overlap.wait(timeout=10)

        with transport.observe_minmod_stages(record):
            for _ in range(3):
                q = transport._euler_minmod(q, qx, qy, edges, dt, area)
        return q, captured

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, variant) for variant in (0, 1)]
        concurrent = [future.result(timeout=20) for future in futures]

    for variant, (actual, signature) in enumerate(concurrent):
        expected_output, expected_signature = serial[variant]
        assert signature == expected_signature
        assert len(signature) == 3
        assert torch.equal(actual, expected_output)


def test_concurrent_strict_branch_traces_keep_each_exact_signature():
    def trace(variant, rendezvous=None):
        q, qx, qy, edges, dt, area = _stage_inputs(variant)
        if variant:
            qx, qy = -qx, -qy

        def call():
            if rendezvous is not None:
                rendezvous.wait(timeout=10)
            return transport._euler_minmod(q, qx, qy, edges, dt, area)

        return probe.inspect_branches(call)

    serial = [trace(variant) for variant in (0, 1)]
    rendezvous = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(trace, variant, rendezvous) for variant in (0, 1)]
        concurrent = [future.result(timeout=20) for future in futures]
    for actual, expected in zip(concurrent, serial):
        assert actual["euler_stages"] == expected["euler_stages"] == 1
        assert actual["choices"] == expected["choices"]
        assert actual["face_signs"] == expected["face_signs"]
        assert actual["minimum_scaled_slope_margin"] == expected["minimum_scaled_slope_margin"]
    assert serial[0]["face_signs"] != serial[1]["face_signs"]


def test_zero_face_forward_does_not_enter_another_threads_observer():
    q, qx, qy, edges, dt, area = _stage_inputs()
    zero_qx = torch.zeros_like(qx)
    zero_qy = torch.zeros_like(qy)
    zero_edges: transport.BoundaryEdges = (
        torch.zeros_like(edges[0]), torch.zeros_like(edges[1]),
        torch.zeros_like(edges[2]), torch.zeros_like(edges[3]),
    )
    baseline = transport._euler_minmod(q, zero_qx, zero_qy, zero_edges, dt, area)
    observer_started = Event()
    release_observer = Event()
    strict_calls = []

    def strict_trace():
        def record(stage_q, stage_qx, stage_qy):
            strict_calls.append((stage_q.clone(), stage_qx.clone(), stage_qy.clone()))
            observer_started.set()
            assert release_observer.wait(timeout=10)

        with transport.observe_minmod_stages(record):
            return transport._euler_minmod(q, qx, qy, edges, dt, area)

    def ordinary_forward():
        assert observer_started.wait(timeout=10)
        actual = transport._euler_minmod(q, zero_qx, zero_qy, zero_edges, dt, area)
        release_observer.set()
        return actual

    with ThreadPoolExecutor(max_workers=2) as pool:
        strict_future = pool.submit(strict_trace)
        ordinary_future = pool.submit(ordinary_forward)
        strict_result = strict_future.result(timeout=20)
        ordinary_result = ordinary_future.result(timeout=20)

    assert len(strict_calls) == 1
    assert torch.equal(strict_calls[0][0], q)
    assert torch.equal(strict_calls[0][1], qx)
    assert torch.equal(strict_calls[0][2], qy)
    assert torch.equal(strict_result, transport._euler_minmod(q, qx, qy, edges, dt, area))
    assert torch.equal(ordinary_result, baseline)
    assert torch.equal(ordinary_result, q)
