"""Source-bound static arithmetic for the FV problem definition; no model run."""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=Path(__file__).resolve().parent


def function(path: str, name: str) -> ast.FunctionDef:
    tree=ast.parse((ROOT/path).read_text())
    return next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==name)


def tensor_literal(path: str, function_name: str, assigned_name: str) -> Any:
    node=function(path,function_name)
    assignment=next(item for item in ast.walk(node) if isinstance(item,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id==assigned_name for target in item.targets))
    if not isinstance(assignment.value,ast.Call):
        raise ValueError('expected a literal tensor construction')
    return ast.literal_eval(assignment.value.args[0])


def module_constant(path: str, name: str) -> Any:
    tree=ast.parse((ROOT/path).read_text())
    assignment=next(node for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id==name for target in node.targets))
    return ast.literal_eval(assignment.value)


def main() -> None:
    basis_source='examples/weather_scenarios/fv_minmod_inverse_probe.py'
    generator='examples/weather_scenarios/fv_long_horizon_case.py'
    profile=function(basis_source,'make_spatial_case')
    basis=next(node.value.args[0] for node in ast.walk(profile) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='basis' for target in node.targets)
        and isinstance(node.value,ast.Call))
    if ast.unparse(basis)!='(y, x, x * y, 0.5 * (x.square() - y.square()), x.square() * y)':
        raise ValueError('independent polynomial arithmetic no longer matches the source basis')
    limits=tensor_literal(basis_source,'make_spatial_case','limits')
    fractions=tensor_literal(generator,'make_flow_case','fractions')
    coefficients=[a*b for a,b in zip(limits,fractions,strict=True)]
    def psi(y: float,x: float) -> float:
        terms=(y,x,x*y,.5*(x*x-y*y),x*x*y)
        return math.fsum(a*b for a,b in zip(coefficients,terms,strict=True))
    flux=-(psi(4,4)-psi(4,3))
    face_weights=[0.,-limits[1],-4*limits[2],-3.5*limits[3],-28*limits[4]]
    dot=math.fsum(a*b for a,b in zip(face_weights,fractions,strict=True))
    assert math.isclose(flux,dot,rel_tol=64*math.ulp(1.0),abs_tol=0.)
    assert math.isclose(dot,.104,rel_tol=64*math.ulp(1.0),abs_tol=0.)
    interval=60*module_constant(generator,'INTERVAL_MINUTES')
    leads=module_constant(generator,'FORECAST_LEADS')
    paths=[basis_source,generator,'src/advar/fv_point_research_problem.py',
        'src/advar/variational.py','src/advar/transport.py',
        'examples/weather_scenarios/fv_point_3h_forward_case.py',
        'examples/weather_scenarios/fv_point_3h_seed_linear_probe.py',
        'examples/weather_scenarios/fv_point_response_preflight.py']
    seed_source='examples/weather_scenarios/fv_point_3h_seed_linear_probe.py'
    counts={'controls':module_constant(seed_source,'EXPECTED_CONTROLS'),
        'parameters':module_constant(seed_source,'EXPECTED_PARAMETERS'),
        'analysis_euler_stages':module_constant(seed_source,'EXPECTED_ANALYSIS_STAGES'),
        'complete_euler_stages':module_constant(seed_source,'EXPECTED_STAGES')}
    assert counts=={'controls':26,'parameters':13,'analysis_euler_stages':360,'complete_euler_stages':3600}
    observations=[0,interval,2*interval]
    terminal=(2+leads)*interval
    assert observations==[0,600,1200] and terminal==12000 and terminal-observations[-1]==10800
    result={'scope':'AST-extracted constants and independent polynomial/interval arithmetic; no FV, seed, AD, optimization, or response execution',
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},
        'control_parameter_counts':counts,'observation_times_seconds':observations,
        'forecast_leads':leads,'terminal_absolute_seconds':terminal,
        'forecast_after_last_observation_seconds':terminal-observations[-1],
        'generator_limits':limits,'generator_fractions':fractions,
        'generator_coefficients':coefficients,'target_face_weights_on_fractions':face_weights,
        'generator_Qy_4_3_by_polynomial_difference':flux,'generator_Qy_4_3_by_dot_product':dot,
        'Q0_is_not_a_given_truth_constraint':True,'root_or_minimum_location_proved':False,
        'runtime_code_modified':False,'new_numerical_candidates':0}
    output=EVIDENCE/'PROBLEM_DEFINITION_STATIC_CHECKS_20261011.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('terminal_absolute_seconds','generator_Qy_4_3_by_polynomial_difference','generator_Qy_4_3_by_dot_product')}))


if __name__=='__main__':
    main()
