"""Recalculate saved provenance/Armijo/side evidence without an FV run."""
from pathlib import Path
import hashlib
import json
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=HERE/'attempt1'
sha=lambda b:hashlib.sha256(b).hexdigest()
s=json.loads((OUT/'RESULT.json').read_text())
p=json.loads((OUT/'PLAN.json').read_text())
checks={}
checks['plan_hash']=sha((OUT/'PLAN.json').read_bytes())==(OUT/'PLAN_SHA256.txt').read_text().strip()==s['plan_sha256']
checks['plan_content']=p==s['plan']
with tarfile.open(OUT/'source.tar.gz') as archive:
 checks['source_archive']=all(sha(archive.extractfile(k).read())==v for k,v in p['source_hashes'].items())
checks['source_closure']=all(sha((ROOT/k).read_bytes())==v for k,v in p['source_hashes'].items()) and s['source_unchanged']
checks['input_closure']=all(sha((ROOT/k).read_bytes())==v for k,v in p['input_hashes'].items()) and s['inputs_unchanged'] and s['parameters_unchanged']
with tarfile.open(HERE/'TRACES.tar.gz') as traces:
 checks['trace_hashes']=all(sha(traces.extractfile(r['trace_path']).read())==r['trace_sha256'] for r in s['evaluations'])
control=np.asarray(s['current_control'],dtype='<f8')
checks['terminal_control_hash']=sha(control.tobytes())==s['current_control_sha256']==s['terminal']['control_sha256']
previous=p['control_sha256'];j0=s['evaluations'][0]['native_J'];previous_j=j0
armijo=True
for a in s['accepted_steps']:
 control=np.asarray(a['control'],dtype='<f8')
 armijo &= (a['from_control_sha256']==previous and sha(control.tobytes())==a['control_sha256']
  and a['native_J']<previous_j and a['slope']<0 and 0<a['effective_alpha']<=a['alpha']
  and a['armijo_threshold']==previous_j+1e-4*a['effective_alpha']*a['slope']
  and a['native_J']<=a['armijo_threshold'])
 previous=a['control_sha256'];previous_j=a['native_J']
checks['accepted_native_armijo_chain']=bool(armijo and previous==s['current_control_sha256'] and previous_j==s['terminal']['native_J'])
counts=s['counts']
checks['count_caps']=all(counts[k+'_completed']<=counts[k+'_started']<=p['budget'][k+'_cap'] for k in ('direct_J','gradient','Hessian_row','profile_forward'))
checks['cumulative_counts']=(s['cumulative_direct_J_completed']==594+counts['direct_J_completed'] and s['cumulative_gradient_completed']==163+counts['gradient_completed'])
scales=True;direction_checks=True
for f in s['side_families']:
 prior=np.asarray(f['prior_chart_gradient']);values=[np.linalg.norm(prior)+np.linalg.norm(np.asarray(r['gradient'])-prior) for r in f['records']]
 scales &= abs(f['gradient_component_scale']-max(values))<=1e-14
 for d in f['decisions']:
  event=d['event'];i=f['events'].index(event);chosen=[r for r in f['records'] if r['signs'][i]==d['side']]
  slopes=[np.asarray(r['gradient'])@np.asarray(d['direction']) for r in chosen]
  direction_checks &= np.allclose(slopes,d['stepped_direction_derivatives'],rtol=0,atol=1e-15) and abs(max(slopes)-d['derivative'])<=1e-15
  if d['decision']=='release':direction_checks &= max(slopes)<-d['error_budget'] and d['stepped_direction_valid']
checks['actual_prior_cancellation_scales']=bool(scales)
checks['exact_stepped_release_derivatives']=bool(direction_checks)
checks['claim_gates']=not s['minimum_claim'] or (s['terminal']['minimum_claim'] and s['terminal']['first_order_complete'] and s['terminal']['P2_complete'])
result={'checks':checks,'all_passed':all(checks.values()),'scope':'saved-source/input/control/trace/Armijo/derivative arithmetic; no FV, optimizer, P2 or adjoint rerun',
 'accepted_steps':len(s['accepted_steps']),'J_start':j0,'J_end':previous_j,'decrease':j0-previous_j,'relative_decrease_percent':100*(j0-previous_j)/j0,
 'result_sha256':sha((OUT/'RESULT.json').read_bytes())}
(HERE/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
assert result['all_passed']
