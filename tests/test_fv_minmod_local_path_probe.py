"""Protect cached tangent/adjoint reuse with the measured 26-control system."""
import hashlib
import importlib.util
import json
from pathlib import Path
from dataclasses import replace
from typing import Any

import pytest
import torch
from advar import variational as v

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'local_path_probe', ROOT / 'examples/weather_scenarios/fv_minmod_local_path_probe.py')
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


@pytest.fixture
def system():
    data = json.loads((ROOT / 'graphify-out/fv-root-cause-20260919/minmod_local_path_coarse.json').read_text())
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)
    return (tensor(data['hessian']['matrix']),
            {k: tensor(v['control_tangent']) for k, v in data['tangents'].items()},
            {k: tensor(v['cross_gradient']) for k, v in data['tangents'].items()},
            tensor(data['adjoint']['solution']), tensor(data['adjoint']['rhs']))


def _synthetic_resume_checkpoint(path, *, selected_direction, pairs):
    """Build a current-schema controller fixture without editing archived science."""
    spatial_probe = PROBE._load_module(PROBE.PROBE, "fv_minmod_inverse_for_resume_test")
    obs, frozen, boundary, support = spatial_probe.make_spatial_case()
    saved = json.loads(PROBE.SAVED.read_text())
    control = torch.tensor(saved["control"], dtype=torch.float64)
    parameters = torch.cat((obs.dbz.flatten(), obs.dbz.new_tensor([0.02])))
    directions = {
        "observation_sine60": torch.cat((
            torch.sin(torch.arange(obs.dbz.numel(), dtype=parameters.dtype)),
            parameters.new_zeros(1),
        )),
        "theta": torch.cat((parameters.new_zeros(obs.dbz.numel()), parameters.new_ones(1))),
    }
    saved_sources = saved["source_sha256"]
    current_identity = saved_sources
    input_identity = {
        "shape": list(frozen.initial_background_dbz.shape),
        "controls": int(control.numel()),
        "substeps_per_interval": 9,
        "parameters_sha256": hashlib.sha256(parameters.numpy().tobytes()).hexdigest(),
        "control_sha256": hashlib.sha256(control.numpy().tobytes()).hexdigest(),
    }

    # This synthetic positive-definite linear system exercises resume wiring only;
    # score derivatives still come from the current fixed-input forecast.
    size = control.numel()
    hessian = torch.eye(size, dtype=torch.float64)
    basis = torch.zeros(size, dtype=torch.float64)
    basis[0] = 1.0
    crosses = {name: basis * 1e-6 for name in directions}
    tangents = {
        name: {"parameter_direction": direction.tolist(),
              "cross_gradient": crosses[name].tolist(),
              "control_tangent": (-crosses[name]).tolist()}
        for name, direction in directions.items()
    }

    pattern = torch.linspace(-0.2, 0.3, frozen.initial_background_dbz.numel(),
                             dtype=obs.dbz.dtype).reshape_as(frozen.initial_background_dbz)

    def contract(p):
        y, theta = p[:-1].reshape_as(obs.dbz), p[-1]
        return replace(frozen, initial_background_dbz=y[0] + theta * pattern)

    def forecast(c, p):
        return spatial_probe.echo_to_dbz(
            v.forecast_fv_analysis(
                c, contract(p), leads=1, boundary_start_interval=2,
                boundary_echo=boundary, boundary_support=support,
            ).frames_linear[-1], min_dbz=-10.0,
        )

    verification = forecast(control, parameters).detach() + 0.1 * pattern

    def score(c, p):
        return (forecast(c, p) - verification).square().mean()

    rhs, direct = torch.func.grad(score, argnums=(0, 1))(control, parameters)
    data = {
        "nominal_control": control.tolist(),
        "input_identity": input_identity,
        "source_identity": {"saved": saved_sources, "core_exact": True,
                            "fixture_probe_exact": True},
        "saved_report_sha256": PROBE._hash(PROBE.SAVED),
        "selected_direction": selected_direction,
        "pairs": pairs,
        "hessian": {"matrix": hessian.tolist()},
        "tangents": tangents,
        "adjoint": {"rhs": rhs.tolist(), "direct": direct.tolist(),
                    "solution": rhs.tolist()},
    }
    data["cache_contract"] = PROBE._NUMERICAL_CACHE_CONTRACT
    data["cache_payload_identity"] = PROBE._cache_payload_identity(
        {"current": current_identity}, input_identity, directions
    )
    data["cache_payload_identity"]["saved_report_sha256"] = data["saved_report_sha256"]
    data["linearization_sha256"] = PROBE._linearization_digest(data)
    path.write_text(json.dumps(data))
    return path


_CONTROLLER_SOURCE_PATHS = (
    Path("examples/weather_scenarios/fv_minmod_inverse_probe.py"),
    Path("src/advar/transport.py"),
    Path("src/advar/variational.py"),
    Path("examples/weather_scenarios/fv_sensitivity_probe.py"),
)


def _localized_source_identity(source_sha256, root: Path) -> dict[str, str]:
    """Rebase certificate source keys for a test checkout without changing digests."""
    normalized = {str(Path(name)).replace("\\", "/"): digest
                  for name, digest in source_sha256.items()}
    localized = {}
    for relative in _CONTROLLER_SOURCE_PATHS:
        suffix = relative.as_posix()
        matches = [digest for name, digest in normalized.items()
                   if name.endswith("/" + suffix)]
        if len(matches) != 1:
            raise AssertionError(f"expected one archived source hash for {suffix}")
        localized[str(root / relative)] = matches[0]
    if len(localized) != len(source_sha256):
        raise AssertionError("archived source identity has missing or unexpected paths")
    return localized


def _localized_controller_saved_report(tmp_path: Path, root: Path) -> Path:
    """Copy the archived report for controller tests, rewriting path keys only."""
    archived = json.loads(PROBE.SAVED.read_text())
    localized = dict(archived)
    localized["source_sha256"] = _localized_source_identity(
        archived["source_sha256"], root,
    )
    report_path = tmp_path / "controller" / "minmod_spatial_inverse.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(localized))
    return report_path


def _install_archived_source_identity_for_controller_test(monkeypatch, tmp_path):
    """Isolate resume orchestration from archive gates using a rooted copy."""
    localized_saved = _localized_controller_saved_report(tmp_path, PROBE.ROOT)
    saved = json.loads(localized_saved.read_text())
    trusted_identity = saved["source_sha256"]
    original_hash = PROBE._hash
    source_paths = {str(PROBE.ROOT / relative) for relative in _CONTROLLER_SOURCE_PATHS}
    monkeypatch.setattr(PROBE, "SAVED", localized_saved)
    monkeypatch.setattr(PROBE, "MEASURED_PROBE", PROBE.PROBE)

    def identity_hash(path):
        if str(path) in trusted_identity:
            return trusted_identity[str(path)]
        if path == PROBE.MEASURED_PROBE:
            return trusted_identity[str(PROBE.ROOT / _CONTROLLER_SOURCE_PATHS[0])]
        if str(path) in source_paths:
            raise AssertionError(f"missing localized source identity for {path}")
        return original_hash(path)

    monkeypatch.setattr(PROBE, "_hash", identity_hash)
    original_load = PROBE._load_module

    def load(path, name):
        module = original_load(path, name)
        if path == PROBE.PROBE:
            module._hash = identity_hash
            branch = json.loads(PROBE.SAVED.read_text())["stationary_branch"]
            module.inspect_branches = lambda *_args, **_kwargs: branch
        return module

    monkeypatch.setattr(PROBE, "_load_module", load)


def test_controller_source_identity_follows_an_alternate_checkout_root(tmp_path):
    archived = json.loads(PROBE.SAVED.read_text())
    original_source_sha256 = dict(archived["source_sha256"])
    alternate_root = tmp_path / "different-checkout"

    localized = _localized_source_identity(original_source_sha256, alternate_root)

    assert set(localized) == {
        str(alternate_root / relative) for relative in _CONTROLLER_SOURCE_PATHS
    }
    assert sorted(localized.values()) == sorted(original_source_sha256.values())
    assert json.loads(PROBE.SAVED.read_text())["source_sha256"] == original_source_sha256


def test_measured_cached_linearization_is_resolved(system):
    assert max(PROBE._check_linearization(*system).values()) < 1e-12


@pytest.mark.parametrize('changed', ['tangent', 'adjoint', 'asymmetry', 'nan'])
def test_corrupted_cached_linearization_is_rejected(system, changed):
    H, tangents, crosses, adjoint, rhs = system
    if changed == 'tangent':
        tangents['observation_sine60'][0] += .01
    elif changed == 'adjoint':
        adjoint[0] += .01
    elif changed == 'asymmetry':
        H[0, 1] += 1.
    else:
        H[0, 0] = float('nan')
    with pytest.raises(ValueError, match='cached'):
        PROBE._check_linearization(H, tangents, crosses, adjoint, rhs)


def test_resume_counts_structural_and_derivative_pairs_separately():
    data = json.loads((ROOT / 'graphify-out/fv-root-cause-20260919/minmod_local_path_coarse.json').read_text())
    assert PROBE._consecutive_pairs(data['pairs'], require_derivative=False) == 2
    assert PROBE._consecutive_pairs(data['pairs'], require_derivative=True) == 0


def test_cached_score_derivatives_require_scaled_finite_agreement():
    fresh = torch.tensor([2.0, -3.0, 1.0], dtype=torch.float64)
    assert PROBE._check_cached_score_derivative("rhs", fresh.clone(), fresh) == 0.0
    changed = fresh.clone()
    changed[0] += 1.0e-4
    with pytest.raises(ValueError, match="rhs"):
        PROBE._check_cached_score_derivative("rhs", changed, fresh)
    with pytest.raises(ValueError, match="shape/dtype"):
        PROBE._check_cached_score_derivative("rhs", fresh.to(torch.float32), fresh)
    with pytest.raises(ValueError, match="nonfinite"):
        PROBE._check_cached_score_derivative("rhs", torch.full_like(fresh, float("nan")), fresh)


def test_cached_parameter_direction_is_literal_and_typed():
    direction = torch.tensor([0.0, 1.0, -0.5], dtype=torch.float64)
    cached = {"parameter_direction": direction.tolist()}
    PROBE._check_cached_direction("test", cached, direction)
    changed = {"parameter_direction": [0.0, 1.0, 0.5]}
    with pytest.raises(ValueError, match="direction mismatch"):
        PROBE._check_cached_direction("test", changed, direction)
    with pytest.raises(ValueError, match="shape/dtype"):
        PROBE._check_cached_direction("test", {"parameter_direction": [0.0]}, direction)


def test_cache_payload_binds_numerical_contract_and_rejects_mutation():
    expected = {
        "contract": "fv-minmod-local-path-v1",
        "core_source_sha256": {"transport": "a"},
        "fixture_numerical_ast_sha256": "b",
        "shape": [4, 5],
        "controls": 26,
        "substeps_per_interval": 9,
        "parameters_sha256": "p",
        "control_sha256": "c",
        "direction_names": ["observation_sine60", "theta"],
        "saved_report_sha256": "s",
    }
    cache: dict[str, Any] = {"cache_contract": expected["contract"], "cache_payload_identity": dict(expected)}
    cache.update(hessian={"matrix": [[1.]]}, tangents={}, adjoint={"rhs": [1.], "direct": [2.], "solution": [1.]})
    cache["linearization_sha256"] = PROBE._linearization_digest(cache)
    assert PROBE._check_cache_payload(cache, expected, "untrusted") == "versioned"
    cache["adjoint"]["direct"][0] += .001
    with pytest.raises(ValueError, match="payload digest"):
        PROBE._check_cache_payload(cache, expected, "untrusted")
    changed = {"cache_contract": expected["contract"], "cache_payload_identity": {**expected, "control_sha256": "changed"}}
    with pytest.raises(ValueError, match="payload identity"):
        PROBE._check_cache_payload(changed, expected, "untrusted")


def test_legacy_cache_requires_trusted_archive_and_producer():
    path = ROOT / "graphify-out/fv-root-cause-20260919/minmod_local_path_coarse.json"
    data = json.loads(path.read_text())
    expected = {"saved_report_sha256": data["saved_report_sha256"]}
    trusted = PROBE._hash(path)
    assert PROBE._check_cache_payload(data, expected, trusted) == "legacy-anchored"
    with pytest.raises(ValueError, match="legacy cache"):
        PROBE._check_cache_payload(data, expected, "0" * 64)
    mutated = {**data, "producer_sha256": "0" * 64}
    with pytest.raises(ValueError, match="legacy cache"):
        PROBE._check_cache_payload(mutated, expected, trusted)


@pytest.mark.parametrize('mutation', ['direct', 'shape', 'tiny_component', 'verification', 'score'])
def test_direct_cache_rejects_changes_to_current_score(mutation):
    c = torch.tensor([.2, .3], dtype=torch.float64)
    p = torch.tensor([.4, .5], dtype=torch.float64)
    target = torch.tensor([.1, .2], dtype=torch.float64)
    def score(c, p):
        return ((c + p - target)**2).mean()
    rhs, direct = torch.func.grad(score, argnums=(0, 1))(c, p)
    cached = direct.clone()
    if mutation == 'direct':
        cached[0] += .001
    elif mutation == 'shape':
        cached = cached[:1]
    elif mutation == 'tiny_component':
        cached[0], direct[0] = 2e-20, 1e-20
    else:
        if mutation == 'verification':
            target = target + .01
        else:
            original = score
            score = lambda c, p: 2*original(c, p)
        rhs, direct = torch.func.grad(score, argnums=(0, 1))(c, p)
    with pytest.raises(ValueError, match='cached direct'):
        PROBE._check_cached_score_derivative('direct', cached, direct)


def test_calculation_identity_ignores_logging_but_detects_changed_equations(tmp_path):
    assert PROBE.__file__ is not None
    source_path = Path(PROBE.__file__)
    source = source_path.read_text()
    path = tmp_path / 'probe.py'
    expected = PROBE._calculation_hash(source_path)
    path.write_text(source.replace('report["status"] = "running"', 'report["status"] = "starting"'))
    assert PROBE._calculation_hash(path) == expected
    path.write_text(source.replace('0.1 * pattern', '0.2 * pattern'))
    assert PROBE._calculation_hash(path) != expected


def test_new_direction_clears_old_pairs_and_computes_its_cross(tmp_path, monkeypatch):
    # Stop before optimization: exercise actual preparation/cache/JVP wiring.
    class ReachedCorrector(BaseException):
        pass
    original_load = PROBE._load_module
    observed = []
    def load(path, name):
        module = original_load(path, name)
        if path == PROBE.ORACLE:
            def stop(objective, predictor, changed, **kwargs):
                observed.append(changed)
                raise ReachedCorrector
            module.polish = stop
        return module
    original_jvp = torch.func.jvp
    directions = []
    def jvp(func, primals, tangents, **kwargs):
        if primals[0].shape == (61,):
            directions.append(tangents[0].clone())
        return original_jvp(func, primals, tangents, **kwargs)
    monkeypatch.setattr(PROBE, '_load_module', load)
    monkeypatch.setattr(torch.func, 'jvp', jvp)
    _install_archived_source_identity_for_controller_test(monkeypatch, tmp_path)
    resume = _synthetic_resume_checkpoint(
        tmp_path / "resume.json", selected_direction="observation_sine60",
        pairs=[{"j": 2, "predictors_ok": True, "endpoints": [
            {"same_local_branch": True}, {"same_local_branch": True},
        ], "derivative_pass": True}],
    )
    output = tmp_path / 'new.json'
    with pytest.raises(ReachedCorrector):
        PROBE.run(output, resume=resume,
                  direction_name='middle_time_bias')
    data = json.loads(output.read_text())
    assert data['pairs'] == [] and data['resume_start_j'] == 0
    assert data['selected_direction'] == 'middle_time_bias'
    expected = torch.zeros(61, dtype=torch.float64)
    expected[20:40] = 1
    assert len(directions) == 1 and torch.equal(directions[0], expected)
    assert len(observed) == 1


def test_completed_direction_resume_validates_without_repeating_reanalysis(tmp_path, monkeypatch):
    original_load = PROBE._load_module
    def load(path, name):
        module = original_load(path, name)
        if path == PROBE.ORACLE:
            def unexpected(*args, **kwargs):
                pytest.fail('completed direction must not repeat reanalysis')
            module.polish = unexpected
        return module
    monkeypatch.setattr(PROBE, '_load_module', load)
    _install_archived_source_identity_for_controller_test(monkeypatch, tmp_path)
    source = _synthetic_resume_checkpoint(
        tmp_path / "theta.json", selected_direction="theta",
        pairs=[{"j": j, "predictors_ok": True, "endpoints": [
            {"same_local_branch": True}, {"same_local_branch": True},
        ], "derivative_pass": True} for j in (0, 1)],
    )
    result = PROBE.run(tmp_path / 'validated.json', resume=source, direction_name='theta')
    assert result['status'] == 'complete'
    assert result['pairs'] == json.loads(source.read_text())['pairs']
    assert 'metadata_correction' not in result
    assert result['cache_validation']['payload'] == 'versioned'


def test_run_rejects_changed_core_source_identity(tmp_path, monkeypatch):
    original_hash = PROBE._hash

    def changed_hash(source):
        if source == ROOT / "src/advar/transport.py":
            return "0" * 64
        return original_hash(source)

    monkeypatch.setattr(PROBE, "_hash", changed_hash)
    with pytest.raises(ValueError, match="core or measured fixture source identity mismatch"):
        PROBE.run(tmp_path / "rejected.json")


def test_run_rejects_changed_nominal_branch_signature(tmp_path, monkeypatch):
    _install_archived_source_identity_for_controller_test(monkeypatch, tmp_path)
    original_load = PROBE._load_module

    def load(path, name):
        module = original_load(path, name)
        if path == PROBE.PROBE:
            branch = json.loads(PROBE.SAVED.read_text())["stationary_branch"]
            branch = {**branch, "choices": list(branch["choices"])}
            branch["choices"][0] = "changed"
            module.inspect_branches = lambda *_args, **_kwargs: branch
        return module

    monkeypatch.setattr(PROBE, "_load_module", load)
    with pytest.raises(ValueError, match="saved/current nominal RK branch signature mismatch"):
        PROBE.run(tmp_path / "rejected.json")


def test_run_rejects_changed_measured_fixture_identity(tmp_path, monkeypatch):
    _install_archived_source_identity_for_controller_test(monkeypatch, tmp_path)
    measured_fixture = tmp_path / "measured_fixture.py"
    measured_fixture.write_text(PROBE.PROBE.read_text())
    monkeypatch.setattr(PROBE, "MEASURED_PROBE", measured_fixture)
    original_hash = PROBE._hash

    def changed_fixture_hash(source):
        if source == measured_fixture:
            return "0" * 64
        return original_hash(source)

    monkeypatch.setattr(PROBE, "_hash", changed_fixture_hash)
    with pytest.raises(ValueError, match="core or measured fixture source identity mismatch"):
        PROBE.run(tmp_path / "rejected.json")
