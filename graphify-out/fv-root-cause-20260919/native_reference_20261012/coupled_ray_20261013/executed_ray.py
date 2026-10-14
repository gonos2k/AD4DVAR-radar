from pathlib import Path
import json,hashlib,importlib.util,sys,time
import torch
R=Path('/Users/yhlee/ADVAR');sys.path.insert(0,str(R));H=R/'graphify-out/fv-root-cause-20260919/native_reference_20261012';D=H/'two_face_intersection_20261013';O=H/'coupled_ray_20261013';O.mkdir(exist_ok=True)
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
q=load('ray_q',H/'qualify_candidate.py');geo=load('ray_geo',D/'diagnose.py')
inp=D/'continuation2_20261013/native24_chord_extension_result.json'; old=D/'continuation2_20261013/native24_resume_qualification.json'
a=json.loads(inp.read_text());b=json.loads(old.read_text());c=torch.tensor(a['current_control'],dtype=torch.float64);case=q.cycle.load_fixed_case();p=case['parameters'];problem=case['problem'];f=problem.frozen.fv_transport
W,I,F,meta=geo._chart(f.coefficient_limits,f.psi_basis);lift,z,projected,_=geo._make_intersection_chart(c,W,I,F)
assert torch.equal(projected,c)
g=torch.tensor(a['current_point']['tangent_hull']['vector'],dtype=torch.float64);hh=torch.tensor(b['fresh_hessians'][-1]['hessian'],dtype=torch.float64);d=torch.linalg.solve(hh,-g);slope=float(g@d)
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
plan={'input_sha256':sha(inp),'metric_input_sha256':sha(old),'scope':'actual native J/branch along fresh-gradient and stale-SPD-metric direction; diagnosis only','metric_is_current_hessian':False,'alphas':[0.,1/1024,1/128,1/64,1/32,1/8,1.],'native_J_calls':7,'gradient_calls':0,'score_calls':0,'initial_cumulative_J':499,'final_cumulative_J':506,'seconds_cap':30,'slope':slope,'growth_ray_elapsed':5.6573785841465,'observer_elapsed':2.40698,'data_parameters_sha256':q._tensor_sha(p),'sources':{str(x.relative_to(R)):sha(x) for x in [R/'src/advar/transport.py',R/'src/advar/variational.py',Path('/tmp/native_ray.py')] if x.is_relative_to(R)}}
(O/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');start=time.monotonic();rows=[]
for alpha in plan['alphas']:
 if time.monotonic()-start>30:raise TimeoutError('ray cap')
 point=lift(z+alpha*d);audit=q._analysis_branch(problem,point,p,deadline=start+30)
 rows.append({'alpha':alpha,'native_J':audit['native_J_on_audited_path'],'linear_predicted_J':a['native_J']+alpha*slope,'branch':audit,'control':point.tolist(),'control_sha256':q._tensor_sha(point)})
 (O/'PREFIX.json').write_text(json.dumps(rows,indent=2)+'\n')
result={'status':'completed','plan':plan,'rows':rows,'elapsed_seconds':time.monotonic()-start,'input_unchanged':sha(inp)==plan['input_sha256'],'metric_input_unchanged':sha(old)==plan['metric_input_sha256'],'minimum_claim':False,'parameter_control_selection':False,'calls':{'native_J':7,'gradient':0,'Hessian':0,'score':0}}
(O/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'elapsed':result['elapsed_seconds'],'slope':slope,'rows':[{'alpha':v['alpha'],'deltaJ':v['native_J']-a['native_J'],'branch':v['branch']['actual_selector_signature_sha256'][:12],'class':v['branch']['classification']} for v in rows]}))
