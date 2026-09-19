import json
from dataclasses import replace
from pathlib import Path
import torch
from advar import variational as v

ROOT=Path('/Users/yhlee/ADVAR')
HERE=ROOT/'graphify-out/fv-root-cause-20260919'
impact=torch.load(HERE/'rotation240_impact_0.001.pt',weights_only=False)
refined=torch.load(HERE/'rotation240_refined.pt',weights_only=False)
control=impact['control']
y=impact['observations']
frozen=refined['frozen']
contract=replace(frozen, initial_background_dbz=y[0], input_frames_dbz=y)
objective=lambda c: v.robust_objective(c, replace(refined['observations'], dbz=y), contract)
gradient=torch.func.grad(objective)
g=gradient(control)
idx=int(torch.argmax(g.abs()))
value=control[idx]
inf=torch.tensor(float('inf'),dtype=control.dtype)
plus_value=torch.nextafter(value,inf)
minus_value=torch.nextafter(value,-inf)
ulp_plus=plus_value-value
ulp_minus=minus_value-value

def trial_gradient(value):
    trial=control.clone(); trial[idx]=value
    return trial,gradient(trial)
plus,gplus=trial_gradient(plus_value)
minus,gminus=trial_gradient(minus_value)
# One exact HVP column predicts the first-order gradient quantum.
basis=torch.zeros_like(control); basis[idx]=1
hcol=torch.func.jvp(gradient,(control,),(basis,))[1]
pred_plus=hcol*ulp_plus
pred_minus=hcol*ulp_minus
out={
 'scope':'240x240 accepted impact checkpoint FP64 dominant-coordinate ULP diagnostic; no gate change',
 'checkpoint':'rotation240_impact_0.001.pt',
 'last_step':impact['last_step'],
 'control_numel':control.numel(),
 'dominant_index':idx,
 'dominant_abs_gradient':float(g[idx].abs()),
 'dominant_gradient':float(g[idx]),
 'gradient_max':float(g.abs().max()),
 'gradient_l2':float(g.norm()),
 'control_value':float(value),
 'nextafter_plus_value':float(plus_value),
 'nextafter_minus_value':float(minus_value),
 'ulp_plus':float(ulp_plus),
 'ulp_minus':float(ulp_minus),
 'plus_gradient':float(gplus[idx]),
 'minus_gradient':float(gminus[idx]),
 'plus_gradient_max':float(gplus.abs().max()),
 'minus_gradient_max':float(gminus.abs().max()),
 'plus_dominant_delta':float(gplus[idx]-g[idx]),
 'minus_dominant_delta':float(gminus[idx]-g[idx]),
 'hessian_column_norm':float(hcol.norm()),
 'hessian_diagonal':float(hcol[idx]),
 'predicted_plus_dominant_delta':float(pred_plus[idx]),
 'predicted_minus_dominant_delta':float(pred_minus[idx]),
 'plus_actual_minus_predicted':float((gplus-g-pred_plus).norm()),
 'minus_actual_minus_predicted':float((gminus-g-pred_minus).norm()),
 'best_adjacent_side': 'plus' if abs(float(gplus[idx])) < abs(float(gminus[idx])) else 'minus',
 'best_adjacent_dominant_abs_gradient':min(abs(float(gplus[idx])),abs(float(gminus[idx]))),
 'best_adjacent_gradient_max':min(float(gplus.abs().max()),float(gminus.abs().max())),
 'source_hashes':impact['source_hashes'],
 'interpretation':'ULP response is a representability diagnostic only; no stationarity-gate relaxation or certification claim.',
}
(HERE/'rotation240_impact_0.001_ulp_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
