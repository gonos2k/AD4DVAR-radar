"""Audit stored signed-candidate receipts without regenerating FV derivatives."""
from pathlib import Path
import hashlib
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK))
import signed_trial as config

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
digest = (WORK / "PLAN_SHA256.txt").read_text().strip()
plan = config.load_plan(config.PLAN, digest)
raw = json.loads((WORK / "attempt1/step.json").read_text())
run = json.loads((WORK / "attempt1/step.run.json").read_text())
cache = config._ensure_cache()
checks = {
    "source_archive_plan": True,
    "raw_run_link": run["child_sha256"] == sha(WORK / "attempt1/step.json"),
    "both_completed": raw["execution_status"] == run["execution_status"] == "completed",
    "same_2a4d_base": raw["base_control_sha256"] == config.BASE_SHA,
    "no_new_HVP": raw["hvp_started"] == raw["hvp_completed"] == 0,
    "four_reused_HVPs": raw["hvp_reused"] == 4 and
        all(event["stage"] == "qualified_reuse" for event in raw["hvp_history"]),
    "no_GN_work": raw["GN_operator"]["jacobian_rows_started"] ==
        raw["GN_operator"]["dense_solves_started"] == 0,
    "both_arms_required": raw["comparison_complete"] and len(raw["arms"]) == 2,
    "two_candidates_one_commit": raw["candidate_evaluations_completed"] == 2 and
        raw["optimizer_steps_applied"] == 1,
    "identity": all(raw[key] for key in ("source_unchanged", "fixed_input_unchanged", "runtime_unchanged")),
    "budgets": raw["deadline_passed"] and run["total_guarded_seconds"] < 660 and
        raw["elapsed_seconds"] + config.PRIOR_GUARDED_SECONDS < 600 and
        run["resource"]["sampled_peak_rss_bytes"] <= config.POLICY["rss_bytes"],
    "no_terminal_claim": not any(raw[key] for key in
        ("root_claim", "minimum_claim", "response_claim", "score_claim", "original_problem_solved", "terminal_E_recomputed")),
}
arms = []
for arm in raw["arms"]:
    name, candidate, repeat = arm["name"], arm["candidate"], arm["repeat"]
    control = torch.tensor(repeat["control"], dtype=torch.float64)
    G = torch.tensor(candidate["G"], dtype=torch.float64)
    full = config.trial.previous.fresh_full_G(repeat, G, config.endpoint.SIDE_WEIGHTS)
    a = torch.tensor(candidate["coefficients"], dtype=torch.float64)
    delta = control - cache["control"]
    expected = cache["V"] @ a
    recorded = candidate["chart_alpha"] * torch.tensor(candidate["DGd"], dtype=torch.float64)
    checks[name + "_repeat"] = config.trial.tangent.fresh_final_closure(arm["proposal"], repeat) and full["passed"]
    checks[name + "_receipt"] = candidate["candidate_receipt_sha256"] == sha(ROOT / candidate["candidate_receipt"])
    checks[name + "_original_JR"] = (repeat["objective"] <= arm["proposal"]["J_armijo_bound"] and
        repeat["F_squared"] <= arm["proposal"]["F_squared_armijo_bound"] and
        repeat["objective"] < raw["base_J"] and repeat["F_squared"] < raw["base_R"])
    checks[name + "_model_linearity"] = config.trial.finite._full_G_match(recorded, expected)["passed"]
    checks[name + "_actual_radius"] = float(delta.norm()) <= config.RADIUS
    # Check reported fidelity against the exact archived model action used by
    # the candidate. The separately checked V@a differs by association roundoff.
    fidelity = config.trial.model_fidelity(cache["G"], recorded, G)
    checks[name + "_fidelity"] = all(abs(fidelity[key] - candidate["fidelity"][key]) <=
        256 * torch.finfo(G.dtype).eps * max(abs(fidelity[key]), abs(candidate["fidelity"][key]), torch.finfo(G.dtype).tiny)
        for key in fidelity)
    mixed = torch.tensor(full["G"], dtype=torch.float64)[:26]
    arms.append(dict(name=name, J=repeat["objective"], R=repeat["F_squared"],
        J_decrease_percent=100*(raw["base_J"]-repeat["objective"])/raw["base_J"],
        R_decrease_percent=100*(raw["base_R"]-repeat["F_squared"])/raw["base_R"],
        residual_norm=repeat["F_squared"]**0.5, mixed_gradient_max=float(mixed.abs().max()),
        coefficients=candidate["coefficients"], chart_alpha=candidate["chart_alpha"],
        actual_path_norm=float(delta.norm()), fidelity=fidelity,
        branch_matches_base=candidate["branch_signature_matches_base"]))
winner = config.trial.comparison.choose_repeated_arm(raw["arms"], torch.float64)
checks["actual_selection"] = winner is not None and winner["name"] == raw["selected_direction"]
checks["winner_hash"] = config.trial.tangent._tensor_sha(torch.tensor(winner["repeat"]["control"], dtype=torch.float64)) == raw["current_control_sha256"]
metrics = dict(execution_status=raw["execution_status"], numerical_status=raw["numerical_status"],
    base_control_sha256=config.BASE_SHA, control_sha256=raw["current_control_sha256"],
    base_J=raw["base_J"], base_R=raw["base_R"], selected_direction=raw["selected_direction"],
    arms=arms, candidates=raw["candidate_evaluations_completed"], commits=raw["optimizer_steps_applied"],
    new_HVP_completed=0, cached_HVP_reused=4, GN_rows=0, GN_solves=0, score_calls=0,
    child_seconds=raw["elapsed_seconds"], guard_seconds=run["resource"]["elapsed_seconds"],
    charged_prior_seconds=config.PRIOR_GUARDED_SECONDS, total_guard_seconds=run["total_guarded_seconds"],
    sampled_peak_rss_bytes=run["resource"]["sampled_peak_rss_bytes"],
    scope="same2a4d alternative branch; prior fb0a preserved; no newpoint operators/score/eligible minimum")
verification = dict(checks=checks, all_checks_passed=all(checks.values()),
    plan_sha256=digest, raw_sha256=sha(WORK / "attempt1/step.json"),
    scope="stored-source/receipt/vector arithmetic; no independent production FV/HVP regeneration")
(WORK / "METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n")
(WORK / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
print(json.dumps(verification))
if not verification["all_checks_passed"]:
    raise SystemExit(1)
