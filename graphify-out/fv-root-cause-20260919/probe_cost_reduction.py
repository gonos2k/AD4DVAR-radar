"""Inspect finite-precision cost reductions at a previously failed C5c endpoint."""
import json, runpy, sys, math
from dataclasses import replace
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from advar import variational as v
from advar.matrix_free import pcg
ns=runpy.run_path(str(ROOT/'graphify-out/fv-root-cause-20260913/c5c_shared_reanalysis.py'))
case=runpy.run_path(str(ROOT/'examples/weather_scenarios/fv_sensitivity_probe.py'))
obs,frozen,_,_=case['make_case']()
y=obs.dbz-5e-4*torch.sin(torch.arange(obs.dbz.numel(),dtype=obs.dbz.dtype)).reshape_as(obs.dbz)
result,frozen=ns['_solve'](y,obs,frozen)
obs=replace(obs,dbz=y); c=result.control
f=v.freeze_irls_weights(c,obs,frozen)
fn=lambda x:v.residual_vector(x,obs,f)
r,pullback=torch.func.vjp(fn,c); g=pullback(r)[0]
normal=lambda x:pullback(torch.func.jvp(fn,(c,),(x,))[1])[0]
base=v.robust_objective(c,obs,frozen)
# Stable differences of the same evaluated loss terms, not a substitute Hessian.
def difference(a,b):
 ra=v._whitened_observation_residual(a,obs,frozen)
 rb=v._whitened_observation_residual(b,obs,frozen)
 d=ra.new_tensor(frozen.analysis_config.pseudo_huber_delta)
 robust=d*(ra-rb)*((ra+rb)/(torch.hypot(ra,d)+torch.hypot(rb,d)))
 pa,pb=v._control_prior_residual(a,frozen),v._control_prior_residual(b,frozen)
 sa,sb=v._field_smoothness_residual(a,frozen),v._field_smoothness_residual(b,frozen)
 return torch.where(obs.valid_mask,robust,torch.zeros_like(robust)).sum()+.5*((pa-pb)*(pa+pb)).sum()+.5*((sa-sb)*(sa+sb)).sum()
rows=[]
for damping in (1e-6,1e-3):
 linear=pcg(lambda x:normal(x)+damping*x,-g,rtol=1e-10,max_iterations=96)
 for scale in (1.,.5,.25):
  step=scale*linear.solution; candidate=c+step
  value=v.robust_objective(candidate,obs,frozen)
  grad=torch.func.grad(v.robust_objective)(candidate,obs,frozen)
  rows.append({'damping':damping,'scale':scale,'pcg_converged':linear.converged,'predicted':float(-g@step-.5*step@normal(step)),'subtracted_cost_change':float(base-value),'factored_term_change':float(difference(c,candidate)),'gradient_norm':float(grad.norm()),'step_norm':float(step.norm())})
payload={'endpoint':ns['_diagnostics'](result),'base_cost':float(base),'trials':rows,'scope':'diagnostic only; floating residual evaluation errors remain'}
Path(__file__).with_suffix('.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
