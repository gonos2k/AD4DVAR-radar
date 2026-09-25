"""Async task ownership for optional FV and PCG diagnostics."""

import asyncio
from threading import Event

import torch

from advar import transport
from advar.matrix_free import observe_pcg_calls, pcg


def _stage(marker: float) -> torch.Tensor:
    q = torch.full((4, 5), marker, dtype=torch.float64)
    qx = torch.full((4, 6), 0.1, dtype=torch.float64)
    qy = torch.full((5, 5), -0.1, dtype=torch.float64)
    edges: transport.BoundaryEdges = (
        torch.full((4,), marker, dtype=torch.float64),
        torch.full((4,), marker, dtype=torch.float64),
        torch.full((5,), marker, dtype=torch.float64),
        torch.full((5,), marker, dtype=torch.float64),
    )
    return transport._euler_minmod(q, qx, qy, edges, q.new_tensor(0.05), q.new_tensor(1.0))


def _solve(marker: float):
    rhs = torch.tensor([marker, 2.0 * marker], dtype=torch.float64)
    diagonal = torch.tensor([2.0, 3.0], dtype=torch.float64)
    return pcg(lambda value: diagonal * value, rhs, rtol=1e-13)


def _pcg_record(markers: list[float]):
    def factory(original):
        def recorded(operator, rhs, **kwargs):
            markers.append(float(rhs[0]))
            return original(operator, rhs, **kwargs)
        return recorded
    return factory


def test_sibling_async_tasks_keep_their_own_stage_and_pcg_records():
    async def scenario():
        entered = (asyncio.Event(), asyncio.Event())
        release = asyncio.Event()
        stages: list[list[float]] = [[], []]
        solves: list[list[float]] = [[], []]

        async def worker(index: int):
            marker = float(index + 1)
            with transport.observe_minmod_stages(
                lambda q, _qx, _qy: stages[index].append(float(q[0, 0]))
            ), observe_pcg_calls(_pcg_record(solves[index])):
                entered[index].set()
                await release.wait()
                result = _stage(marker)
                await asyncio.sleep(0)
                solved = _solve(marker)
                return result, solved

        tasks = [asyncio.create_task(worker(index)) for index in (0, 1)]
        await asyncio.gather(*(event.wait() for event in entered))
        ordinary_stage = _stage(3.0)
        ordinary_solve = _solve(3.0)
        assert stages == [[], []] and solves == [[], []]
        release.set()
        results = await asyncio.gather(*tasks)
        _stage(4.0)
        _solve(4.0)
        return stages, solves, ordinary_stage, ordinary_solve, results

    stages, solves, ordinary_stage, ordinary_solve, results = asyncio.run(scenario())
    assert stages == [[1.0], [2.0]]
    assert solves == [[1.0], [2.0]]
    for marker, (stage, solved) in enumerate(results, start=1):
        assert torch.equal(stage, _stage(float(marker)))
        assert solved.converged
        torch.testing.assert_close(solved.solution, torch.tensor(
            [marker / 2.0, 2.0 * marker / 3.0], dtype=torch.float64
        ))
    assert torch.equal(ordinary_stage, _stage(3.0))
    assert ordinary_solve.converged


def test_inherited_child_does_not_record_after_parent_observer_scope_exits():
    async def scenario():
        release = asyncio.Event()
        stages: list[float] = []
        solves: list[float] = []

        async def delayed():
            await release.wait()
            stage = _stage(2.0)
            solved = _solve(2.0)
            return stage, solved

        with transport.observe_minmod_stages(
            lambda q, _qx, _qy: stages.append(float(q[0, 0]))
        ), observe_pcg_calls(_pcg_record(solves)):
            _stage(1.0)
            _solve(1.0)
            child = asyncio.create_task(delayed())
        release.set()
        stage, solved = await child
        return stages, solves, stage, solved

    stages, solves, stage, solved = asyncio.run(scenario())
    assert stages == [1.0]
    assert solves == [1.0]
    assert torch.equal(stage, _stage(2.0))
    assert solved.converged


def test_inherited_thread_context_does_not_record_after_scope_exits():
    async def scenario():
        started = Event()
        release = Event()
        stages: list[float] = []
        solves: list[float] = []

        def delayed():
            started.set()
            assert release.wait(timeout=5)
            return _stage(2.0), _solve(2.0)

        with transport.observe_minmod_stages(
            lambda q, _qx, _qy: stages.append(float(q[0, 0]))
        ), observe_pcg_calls(_pcg_record(solves)):
            child = asyncio.create_task(asyncio.to_thread(delayed))
            assert await asyncio.to_thread(started.wait, 5)
        release.set()
        stage, solved = await child
        return stages, solves, stage, solved

    stages, solves, stage, solved = asyncio.run(scenario())
    assert stages == [] and solves == []
    assert torch.equal(stage, _stage(2.0))
    assert solved.converged


def test_expired_inner_scope_keeps_live_outer_observer():
    async def scenario():
        release = asyncio.Event()
        stage_outer: list[float] = []
        stage_inner: list[float] = []
        pcg_outer: list[float] = []
        pcg_inner: list[float] = []

        async def delayed():
            await release.wait()
            _stage(2.0)
            _solve(2.0)

        with transport.observe_minmod_stages(
            lambda q, _qx, _qy: stage_outer.append(float(q[0, 0]))
        ), observe_pcg_calls(_pcg_record(pcg_outer)):
            with transport.observe_minmod_stages(
                lambda q, _qx, _qy: stage_inner.append(float(q[0, 0]))
            ), observe_pcg_calls(_pcg_record(pcg_inner)):
                child = asyncio.create_task(delayed())
            release.set()
            await child
        return stage_outer, stage_inner, pcg_outer, pcg_inner

    assert asyncio.run(scenario()) == ([2.0], [], [2.0], [])
