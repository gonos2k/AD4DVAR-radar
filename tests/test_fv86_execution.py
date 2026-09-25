"""Cheap guard and accounting tests; no scaled inverse solve."""
import importlib.util
import os
from pathlib import Path
import sys
from threading import Event, Timer

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'examples/weather_scenarios'/f'{name}.py')
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(('wall','rss','expected'),[(.05,2*1024**3,'wall_time_limit'),(5,1,'rss_limit')])
def test_resource_guard_terminates_child(tmp_path,wall,rss,expected):
    module=load('fv86_resource_runner')
    result=module.run_guarded([sys.executable,'-c','import time; time.sleep(10)'],
        wall_seconds=wall,rss_bytes=rss,report_path=tmp_path/'guard.json',log_path=tmp_path/'guard.log')
    assert result['resource_termination']==expected
    assert result['exit_code']!=0
    assert result['elapsed_seconds']<4


def test_resource_guard_records_normal_exit(tmp_path):
    module=load('fv86_resource_runner')
    result=module.run_guarded([sys.executable,'-c','print("done")'],wall_seconds=5,
        rss_bytes=2*1024**3,report_path=tmp_path/'guard.json',log_path=tmp_path/'guard.log')
    assert result['resource_termination'] is None
    assert result['exit_code']==0
    assert type(result['child_pid']) is int and result['child_pid']!=os.getpid()
    assert (tmp_path/'guard.log').read_text().strip()=='done'


def test_resource_guard_cancels_only_the_requested_child(tmp_path):
    module=load('fv86_resource_runner')
    cancelled=Event()
    timer=Timer(.1,cancelled.set)
    timer.start()
    try:
        result=module.run_guarded(
            [sys.executable,'-c','import time; time.sleep(10)'],
            wall_seconds=5,rss_bytes=2*1024**3,
            report_path=tmp_path/'cancel.resource.json',
            log_path=tmp_path/'cancel.log',
            cancel_requested=cancelled.is_set,
        )
    finally:
        timer.cancel()
    assert result['resource_termination']=='cancelled'
    assert result['exit_code']!=0
    assert result['elapsed_seconds']<4
    assert result['monitor_error'] is None


@pytest.mark.parametrize(('message','expected'),[
    ('operator must be symmetric positive definite','nonpositive_curvature'),
    ('matrix-free adjoint PCG did not converge','linear_budget_exhausted'),
    ('matrix-free Newton PCG true residual exceeds tolerance','linear_residual_failed'),
    ('stationarity refinement iteration budget exhausted','stationarity_not_met'),
    ('scaled minmod branch identity mismatch','branch_uncertified'),
    ('unexpected callback defect','unclassified_refusal'),
])
def test_numerical_refusal_categories_preserve_distinctions(message,expected):
    module=load('fv86_execution_probe')
    assert module.failure_category(message,'refinement')==expected
