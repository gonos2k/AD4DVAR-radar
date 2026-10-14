"""Independent SciPy time integration for the pinned 4x5 signed-span case.

This integrates the minmod-MUSCL semidiscrete echo ODE with DOP853. It is a
separate time integrator and NumPy RHS, not an independent spatial model or
continuous-PDE truth. The fixed case and candidate are loaded from receipts;
no observations or original problem inputs are regenerated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar.physics import dbz_to_echo, echo_to_dbz
from advar.variational import forecast_fv_analysis
from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as fixed_case_source

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "attempt1/two_direction_subspace/candidate_01.json"
OUTPUT = HERE / "independent_semidiscrete_reference.json"
ARRAYS = HERE / "independent_semidiscrete_reference.npz"
SIGNED_PLAN = HERE / "PLAN.json"
SIGNED_PLAN_SHA256 = "707bf06bf0432a3f46b6fcc8901908186d70336e075ea5dc72b3cf4658bcfe9b"
SIGNED_SOURCE_COUNT = 237
SIGNED_ARCHIVE_COUNT = 306
FRACTIONS = (0.7, -0.6, 0.5, 0.4, -0.3)
GROWTH_PER_INTERVAL = 0.08


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed_plan_identity() -> dict[str, object]:
    if SIGNED_PLAN.is_symlink() or _sha(SIGNED_PLAN) != SIGNED_PLAN_SHA256:
        raise ValueError("current signed-span plan digest mismatch")
    plan = json.loads(SIGNED_PLAN.read_text())
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if (not isinstance(sources, dict) or not isinstance(archives, dict)
            or len(sources) != SIGNED_SOURCE_COUNT
            or len(archives) != SIGNED_ARCHIVE_COUNT):
        raise ValueError("current signed-span plan membership count mismatch")
    for name, pinned in {**sources, **archives}.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or _sha(path) != pinned:
            raise ValueError(f"current signed-span pin mismatch: {name}")
    return {
        "plan_sha256": SIGNED_PLAN_SHA256,
        "source_count": len(sources),
        "archive_count": len(archives),
        "source_hashes": sources,
        "archive_hashes": archives,
    }


def _schedule_facts(echo_schedule: tuple, support_schedule: tuple) -> dict[str, object]:
    if not echo_schedule or len(echo_schedule) != len(support_schedule):
        raise ValueError("boundary echo/support schedules are empty or misaligned")
    first = echo_schedule[0][0]
    constant_edges = all(
        np.array_equal(np.asarray(edge), np.asarray(first[i]))
        for step in echo_schedule for stage in step
        for i, edge in enumerate(stage)
    )
    equal_stages = all(
        all(np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(step[0], step[1]))
        for step in echo_schedule
    )
    support_all_one = all(
        bool(np.all(np.asarray(edge) == 1))
        for step in support_schedule for stage in step for edge in stage
    )
    return {"steps": len(echo_schedule), "constant_edges": constant_edges,
            "equal_stages": equal_stages, "support_all_one": support_all_one}


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same_positive = (a > 0) & (b > 0)
    same_negative = (a < 0) & (b < 0)
    return np.where(same_positive, np.minimum(a, b),
                    np.where(same_negative, np.maximum(a, b), 0.0))


def _slopes(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = q.shape
    sx = np.zeros_like(q)
    sy = np.zeros_like(q)
    if h > 2 and w > 2:
        sx[1:-1, 1:-1] = _minmod(
            q[1:-1, 1:-1] - q[1:-1, :-2],
            q[1:-1, 2:] - q[1:-1, 1:-1],
        )
        sy[1:-1, 1:-1] = _minmod(
            q[1:-1, 1:-1] - q[:-2, 1:-1],
            q[2:, 1:-1] - q[1:-1, 1:-1],
        )
    return sx, sy


def _rhs(_time: float, q_flat: np.ndarray, qx: np.ndarray, qy: np.ndarray,
         edges: tuple[np.ndarray, ...], area: float, growth_rate: float) -> np.ndarray:
    q = q_flat.reshape(qx.shape[0], qy.shape[1])
    sx, sy = _slopes(q)
    left_face, right_face = q - 0.5 * sx, q + 0.5 * sx
    bottom_face, top_face = q - 0.5 * sy, q + 0.5 * sy
    left, right, bottom, top = edges
    qxp, qxm = np.maximum(qx, 0.0), np.minimum(qx, 0.0)
    qyp, qym = np.maximum(qy, 0.0), np.minimum(qy, 0.0)
    left_source = np.concatenate((left[:, None], right_face[:, :-1]), axis=1)
    right_source = np.concatenate((left_face[:, 1:], right[:, None]), axis=1)
    bottom_source = np.concatenate((bottom[None, :], top_face[:-1, :]), axis=0)
    top_source = np.concatenate((bottom_face[1:, :], top[None, :]), axis=0)
    incoming = (
        qxp[:, :-1] * left_source - qxm[:, 1:] * right_source
        + qyp[:-1, :] * bottom_source - qym[1:, :] * top_source
    )
    outgoing = (
        qxp[:, 1:] * right_face - qxm[:, :-1] * left_face
        + qyp[1:, :] * top_face - qym[:-1, :] * bottom_face
    )
    return ((incoming - outgoing) / area + growth_rate * q).reshape(-1)


def _run(q0: np.ndarray, qx: np.ndarray, qy: np.ndarray,
         edges: tuple[np.ndarray, ...], area: float, growth_rate: float,
         times: np.ndarray, *, rtol: float, atol: float, max_step: float) -> tuple[np.ndarray, int]:
    solution = solve_ivp(
        _rhs, (0.0, float(times[-1])), q0.reshape(-1), method="DOP853",
        t_eval=times, rtol=rtol, atol=atol, max_step=max_step,
        args=(qx, qy, edges, area, growth_rate),
    )
    if not solution.success or solution.y.shape[1] != times.size:
        raise RuntimeError(f"DOP853 reference failed: {solution.message}")
    return solution.y.T.reshape(times.size, *q0.shape), int(solution.nfev)


def main() -> None:
    start = time.monotonic()
    signed_plan_before = _signed_plan_identity()
    reference_script_sha = _sha(Path(__file__).resolve())
    candidate_receipt_sha = _sha(CANDIDATE)
    fixed_case_sha = _sha(fixed_case_source.CASE)
    plan_sha = _sha(fixed_case_source.PLAN)
    plan = fixed_case_source.load_plan(fixed_case_source.PLAN, plan_sha)
    case = fixed_case_source.load_fixed_case()
    problem = case["problem"]
    original = case["original"]
    spec = problem.frozen.fv_transport
    if spec is None or tuple(problem.background_dbz.shape) != (4, 5):
        raise ValueError("unexpected fixed-case shape or transport contract")
    if not torch.equal(problem.background_dbz, problem.frozen.initial_background_dbz):
        raise ValueError("pinned full-field initial background differs")
    boundaries = problem.future_boundary_echo
    initial_schedule = _schedule_facts(spec.boundary_echo, spec.boundary_support)
    future_schedule = _schedule_facts(
        problem.future_boundary_echo, problem.future_boundary_support,
    )
    if not (initial_schedule["constant_edges"] and initial_schedule["equal_stages"]
            and initial_schedule["support_all_one"]
            and future_schedule["constant_edges"] and future_schedule["equal_stages"]
            and future_schedule["support_all_one"]):
        raise ValueError("this reference only handles the pinned constant edge schedule")
    first_edges = tuple(np.asarray(edge, dtype=np.float64)
                        for edge in boundaries[0][0])

    # `original` is the pinned known generating control, not a forecast result.
    # Its first 20 field controls are zero; its final five are atanh fractions,
    # and the last control is the bounded interval log-growth coordinate.
    if not torch.equal(original[:20], torch.zeros_like(original[:20])):
        raise ValueError("pinned generator's field controls are not the expected zero vector")
    fractions = torch.tanh(original[20:25]).numpy()
    growth_fraction = torch.tanh(original[25]).item()
    if not np.allclose(fractions, FRACTIONS, rtol=0.0, atol=2e-15):
        raise ValueError("pinned generator flow fractions differ from the archived case")
    max_growth = problem.frozen.nowcast_config.max_log_growth_per_step
    growth_interval = float(max_growth * growth_fraction)
    if abs(growth_interval - GROWTH_PER_INTERVAL) > 2e-15:
        raise ValueError("pinned generator growth differs from archived case")
    psi_coefficients = spec.coefficient_limits * torch.as_tensor(
        fractions, dtype=original.dtype,
    )
    # Rebuild the streamfunction and staggered fluxes in NumPy, independent
    # of the production `face_volume_fluxes` and trajectory routines.
    psi = np.einsum("k,kij->ij", psi_coefficients.numpy(), spec.psi_basis.numpy())
    qx, qy = psi[1:, :] - psi[:-1, :], -(psi[:, 1:] - psi[:, :-1])
    # make_flow_case initializes its truth from the original full t=0 frame;
    # this pinned background is the same complete field (not point-sampled).
    q0_t = dbz_to_echo(
        problem.background_dbz,
        min_dbz=problem.frozen.nowcast_config.min_dbz,
    )
    q0 = q0_t.numpy()
    dy, dx = spec.spacing_yx
    interval_seconds = problem.frozen.nowcast_config.interval_minutes * 60.0
    times = np.arange(3, 21, dtype=np.float64) * interval_seconds
    growth_rate = growth_interval / interval_seconds
    area = float(dy * dx)
    min_dbz = problem.frozen.nowcast_config.min_dbz

    candidate_record = json.loads(CANDIDATE.read_text())
    candidate = torch.as_tensor(candidate_record["control"], dtype=torch.float64)
    actual_candidate_sha = fixed_case_source.tangent._tensor_sha(candidate)
    expected_candidate_sha = "96108bdcf2cd6abc05878bf2cfc9e2531f37cf82e152ad1a9eaf0a67de746411"
    if (candidate_record.get("control_sha256") != expected_candidate_sha
            or actual_candidate_sha != expected_candidate_sha):
        raise ValueError("candidate receipt/control identity mismatch")
    runtime_before = fixed_case_source.shared._runtime()
    input_before = fixed_case_source.shared._input_identity(
        problem, original, candidate, case["parameters"], case["truth"],
    )

    # Independent adaptive time integration at two tolerances. The state at
    # each 10-minute sampling interval is archived for the GREEN comparison.
    ref_loose, nfev_loose = _run(
        q0, qx, qy, first_edges, area, growth_rate, times,
        rtol=1e-8, atol=1e-10, max_step=60.0,
    )
    ref_tight, nfev_tight = _run(
        q0, qx, qy, first_edges, area, growth_rate, times,
        rtol=1e-10, atol=1e-12, max_step=30.0,
    )
    tight_t = torch.from_numpy(ref_tight)
    truth_dbz = echo_to_dbz(tight_t, min_dbz=min_dbz).numpy()

    if candidate_receipt_sha != _sha(CANDIDATE):
        raise ValueError("candidate receipt changed after reference preflight")
    candidate_frames = forecast_fv_analysis(
        candidate, problem.contract(case["parameters"]),
        leads=problem.frozen.nowcast_config.forecast_steps,
        boundary_start_interval=2,
        boundary_echo=problem.future_boundary_echo,
        boundary_support=problem.future_boundary_support,
    ).frames_linear
    candidate_dbz = echo_to_dbz(candidate_frames[1:], min_dbz=min_dbz).numpy()
    candidate_persistence_dbz = echo_to_dbz(
        candidate_frames[0], min_dbz=min_dbz,
    ).numpy()
    persistence_error = np.abs(candidate_persistence_dbz[None, :, :] - truth_dbz)
    if candidate_dbz.shape != truth_dbz.shape:
        raise ValueError("candidate and reference lead shapes differ")

    archived_terminal = case["truth"].numpy()
    archived_vs_reference = np.abs(truth_dbz[-1] - archived_terminal)
    abs_error = np.abs(candidate_dbz - truth_dbz)
    per_lead_mae = abs_error.mean(axis=(1, 2))
    signed_plan_after = _signed_plan_identity()
    runtime_after = fixed_case_source.shared._runtime()
    input_after = fixed_case_source.shared._input_identity(
        problem, original, candidate, case["parameters"], case["truth"],
    )
    provenance_after = {
        "reference_script_sha256": _sha(Path(__file__).resolve()),
        "candidate_receipt_sha256": _sha(CANDIDATE),
        "fixed_case_sha256": _sha(fixed_case_source.CASE),
        "current_cycle_plan_sha256": _sha(fixed_case_source.PLAN),
    }
    provenance_before = {
        "reference_script_sha256": reference_script_sha,
        "candidate_receipt_sha256": candidate_receipt_sha,
        "fixed_case_sha256": fixed_case_sha,
        "current_cycle_plan_sha256": plan_sha,
    }
    closure = {
        "signed_plan_sha_unchanged": signed_plan_before["plan_sha256"] == signed_plan_after["plan_sha256"],
        "all_signed_source_hashes_unchanged": signed_plan_before["source_hashes"] == signed_plan_after["source_hashes"],
        "all_signed_archive_hashes_unchanged": signed_plan_before["archive_hashes"] == signed_plan_after["archive_hashes"],
        "input_identity_unchanged": input_before == input_after,
        "runtime_identity_unchanged": runtime_before == runtime_after,
        "artifact_digests_unchanged": provenance_before == provenance_after,
    }
    if not all(closure.values()):
        raise RuntimeError(f"reference provenance closure failed: {closure}")
    result = {
        "status": "completed",
        "scope": "independent NumPy minmod RHS + SciPy DOP853 time integration of the pinned semidiscrete 4x5 model; not continuous-PDE truth or held-out weather",
        "pinned_case_sha256": fixed_case_source.CASE_SHA,
        "candidate_control_sha256": candidate_record["control_sha256"],
        "candidate_receipt": str(CANDIDATE.relative_to(ROOT)),
        "provenance_digests": provenance_before,
        "closure": closure,
        "signed_plan_sha256": SIGNED_PLAN_SHA256,
        "signed_source_count": SIGNED_SOURCE_COUNT,
        "signed_archive_count": SIGNED_ARCHIVE_COUNT,
        "initial_field_source": "pinned problem.background_dbz full 4x5 input, checked equal to frozen.initial_background_dbz; forward_case.make_case copies this zero-flow case initial field before truth generation; converted to q by archived min_dbz",
        "initial_field_dbz_range": [float(problem.background_dbz.min()), float(problem.background_dbz.max())],
        "flow_fractions_from_pinned_original_control": fractions.tolist(),
        "growth_per_600_seconds": growth_interval,
        "boundary_schedules": {"initial": initial_schedule, "future": future_schedule},
        "times_seconds": times.tolist(),
        "integrator": "SciPy solve_ivp DOP853, separate NumPy minmod-MUSCL semidiscrete RHS with additive growth source",
        "runtime_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "scipy": scipy.__version__, "torch": torch.__version__},
        "tolerance_runs": {
            "rtol_1e-8_atol_1e-10_max_step_60s_nfev": nfev_loose,
            "rtol_1e-10_atol_1e-12_max_step_30s_nfev": nfev_tight,
            "max_abs_echo_difference": float(np.max(np.abs(ref_loose - ref_tight))),
            "max_abs_dbz_difference": float(np.max(np.abs(
                echo_to_dbz(torch.from_numpy(ref_loose), min_dbz=min_dbz).numpy() - truth_dbz
            ))),
        },
        "archived_same_kernel_terminal_truth_comparison": {
            "max_abs_dbz_difference": float(archived_vs_reference.max()),
            "mean_abs_dbz_difference": float(archived_vs_reference.mean()),
            "purpose": "cross-check initial state/flow/boundary/time conventions; archived truth is not the independent reference",
        },
        "candidate_forecast_vs_independent_reference": {
            "per_lead_mae_dbz": per_lead_mae.tolist(),
            "mean_mae_dbz_all_18_leads": float(per_lead_mae.mean()),
            "max_abs_error_dbz_all_leads": float(abs_error.max()),
            "terminal_mae_dbz": float(per_lead_mae[-1]),
            "terminal_max_abs_error_dbz": float(abs_error[-1].max()),
            "analyzed_frame_persistence_mae_dbz_all_leads": float(persistence_error.mean()),
            "per_lead_analyzed_frame_persistence_mae_dbz": persistence_error.mean(axis=(1, 2)).tolist(),
            "candidate_minus_persistence_mean_mae_dbz": float(per_lead_mae.mean() - persistence_error.mean()),
            "candidate_minus_persistence_terminal_mae_dbz": float(per_lead_mae[-1] - persistence_error[-1].mean()),
        },
        "elapsed_seconds": time.monotonic() - start,
        "artifacts": {"arrays": str(ARRAYS.relative_to(ROOT))},
    }
    np.savez_compressed(
        ARRAYS,
        times_seconds=times,
        independent_echo=ref_tight,
        independent_dbz=truth_dbz,
        candidate_forecast_dbz=candidate_dbz,
        analyzed_frame_persistence_dbz=np.broadcast_to(candidate_persistence_dbz, candidate_dbz.shape),
        archived_terminal_dbz=archived_terminal,
    )
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
