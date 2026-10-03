"""Stored-control discrete-curl arithmetic only: no trajectory or optimizer."""
from pathlib import Path
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
RAW = DIRECTORY / "point_active_face_stationarity_attempt1/stationarity.json"


def flux(t):
    y, x = torch.meshgrid(torch.arange(5, dtype=torch.float64),
                          torch.arange(6, dtype=torch.float64), indexing="ij")
    basis = torch.stack((y, x, x*y, .5*(x*x-y*y), x*x*y))
    w = torch.tensor([0., 4., -3.5, 16.], dtype=torch.float64)
    projected = basis[1:] - w[:, None, None]*basis[0]
    limits = torch.tensor([.08, .07, .04, .03], dtype=torch.float64)
    coefficients = limits*torch.tanh(torch.tensor(t[20:24], dtype=torch.float64))
    psi = torch.einsum("k,kij->ij", coefficients, projected)
    return torch.cat(((psi[1:]-psi[:-1]).flatten(), (-(psi[:, 1:]-psi[:, :-1])).flatten()))


if __name__ == "__main__":
    data = json.loads(RAW.read_text())
    current = flux(data["start_tangent"])
    rows = []
    for trial in data["trials"]:
        candidate = flux(trial["candidate_control"])
        changed = (current.sign() != candidate.sign()).nonzero().flatten().tolist()
        assert float(candidate[22]) == 0
        other = torch.cat((candidate[:22], candidate[23:]))
        rows.append({"iteration": trial["iteration"], "backtrack": trial["backtrack"],
                     "accepted": trial["accepted"], "branch_reason": trial.get("branch_reason"),
                     "changed_flux_flat_indices": changed,
                     "qy_3_0": float(candidate[39]),
                     "other_face_scaled_margin": float(other.abs().min()/candidate.abs().max())})
        if trial["accepted"]:
            current = candidate
    result = {"scope": "stored tangent coefficients and declared projected basis only; rejected limiter choices not replayed",
              "raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
              "rows": rows, "last_qy_3_0": float(current[39]),
              "full_root_claim": False, "normal_validation": "not_performed"}
    target = DIRECTORY / "R2_POINT_ACTIVE_FACE_FLOW_POSTCHECK_20261003.json"
    if target.exists():
        raise ValueError("postcheck output must be fresh")
    target.write_text(json.dumps(result, indent=2)+"\n")
