"""Recalculate saved comparator arrays/receipts; no FV, AD or HVP generation."""
from pathlib import Path
import gzip,hashlib,importlib.util,json
import numpy as np
E=Path(__file__).resolve().parent;D=E/'tangent_precondition_comparison_20261009_attempt1'
sp=importlib.util.spec_from_file_location('saved_tangent',E/'TANGENT_ANALYZE_20261009.py')
assert sp and sp.loader
saved=importlib.util.module_from_spec(sp);sp.loader.exec_module(saved)

def read_raw():
    p=D/'step.json'
    return p.read_bytes() if p.exists() else gzip.decompress((D/'step.json.gz').read_bytes())

def static_flux(c):
    y,x=np.meshgrid(np.arange(5.),np.arange(6.),indexing='ij')
    basis=np.stack((y,x,x*y,.5*(x*x-y*y),x*x*y))
    psi=np.einsum('k,kij->ij',np.array([.11,.08,.07,.04,.03])*np.tanh(c[20:25]),basis)
    return psi[1:]-psi[:-1], -(psi[:,1:]-psi[:,:-1])

def trace_changes(base,new):
    changed=[i for i,(a,b) in enumerate(zip(base['choices'],new['choices'])) if a!=b]
    faces=[i for i,(a,b) in enumerate(zip(base['face_signs'],new['face_signs'])) if a!=b]
    active=[]
    for stage,(a,b) in enumerate(zip(base['choices'],new['choices'])):
        for axis in range(2):
            diff=(np.asarray(a[axis]['choose_left'])!=np.asarray(b[axis]['choose_left']))
            valid=(np.asarray(a[axis]['slope_sign'])!=0)&(np.asarray(b[axis]['slope_sign'])!=0)
            for pos in np.argwhere(diff&valid):active.append([stage,axis,*pos.tolist()])
    return {'changed_limiter_stages':len(changed),'changed_other_face_sign_stages':len(faces),
        'active_choose_left_changes':active}

def main():
    data=read_raw();r=json.loads(data);run=json.loads((D/'step.run.json').read_text());resource=json.loads((D/'step.resource.json').read_text())
    assert run['execution_status']==r['execution_status']=='completed' and run['child_sha256']==hashlib.sha256(data).hexdigest()
    assert run['resource']==resource and r['source_before']==r['source_after'] and r['fixed_input_unchanged'] and r['runtime_unchanged']
    old=json.loads(gzip.decompress((E/'tangent_resume_20261009_attempt1/step.json.gz').read_bytes()))
    last=old['iterations'][-1];base=next(t for t in last['trials'] if t['accepted'])
    p,f,nu,n=saved.point(base['control'],base['theta'],base['side_gradients'],base['objective'])
    assert p['control_sha256']==r['base_control_sha256'] and abs(p['F_squared']-r['initial_F_squared'])<1e-15
    rows=r['jacobian_row_history'];assert len(rows)==r['jacobian_rows_started']==r['jacobian_rows_completed']==24
    assert {(x['side'],x['row']) for x in rows}=={(s,k) for s in (-1,1) for k in range(12)}
    assert all(x['status']=='completed' and x['base_control_sha256']==p['control_sha256'] and x['theta']==base['theta'] for x in rows)
    A={s:np.stack([np.asarray(next(x for x in rows if x['side']==s and x['row']==k)['gradient']) for k in range(12)]) for s in (-1,1)}
    vals={s:np.array([next(x for x in rows if x['side']==s and x['row']==k)['residual_value'] for k in range(12)]) for s in (-1,1)}
    P=np.eye(26)-np.outer(nu,nu);curv=(2/np.hypot(2,vals[-1]))**3
    c=np.asarray(base['control']);w=np.array([0.,-.08,-.28,-.14,-.84]);Qcc=np.zeros(26);Qcc[20:25]=-2*w*np.tanh(c[20:25])*(1-np.tanh(c[20:25])**2)
    mu=(n@f[:26])/(n@n)
    ddiag=(1-base['theta'])*np.sum(curv[:,None]*(A[-1]@P)**2,axis=0)+base['theta']*np.sum(curv[:,None]*(A[1]@P)**2,axis=0)
    rawdiag=ddiag+np.diag(P@P)-mu*np.diag(P@np.diag(Qcc)@P);floored=np.maximum(1,rawdiag);inverse=1/floored
    gn_matrix=(1-base['theta'])*A[-1].T@(curv[:,None]*A[-1])+base['theta']*A[1].T@(curv[:,None]*A[1])+np.eye(26)
    item=r['iterations'][0];arms=[];trace0=r['initial_branch_trace']['-1'];qbase=static_flux(c)[1][3,2]
    for arm in item['model_comparisons']:
        d=np.asarray(arm['direction']);hm=np.asarray(arm['hminus_direction']);hp=np.asarray(arm['hplus_direction']);df=np.asarray(arm['residual_direction'])
        jump=np.asarray(base['side_gradients']['1'])-np.asarray(base['side_gradients']['-1'])
        expected=np.r_[(1-base['theta'])*hm+base['theta']*hp+jump*arm['delta_theta'],n@d/.84]
        np.testing.assert_allclose(df,expected,atol=2e-14,rtol=2e-12);np.testing.assert_allclose(arm['residual'],f,atol=2e-14,rtol=2e-12)
        hvps=[x for x in r['hvp_history'] if x['direction_model']==arm['name']]
        assert len(hvps)==2 and {x['side'] for x in hvps}=={-1,1}
        assert all(x['direction_sha256']==saved.control_sha(d) and x['base_control_sha256']==p['control_sha256'] and x['theta']==base['theta'] and x['phase']=='current_tangent' for x in hvps)
        if arm['name']=='robust_gn_jacobi':
            diag=arm['diagnostics'];np.testing.assert_allclose(diag['raw_diagonal'],rawdiag,atol=1e-10,rtol=2e-13);np.testing.assert_allclose(diag['inverse_diagonal'],inverse,atol=1e-14,rtol=2e-13)
            np.testing.assert_allclose(d,-P@(inverse*(P@f[:26])),atol=3e-15,rtol=3e-13)
        dot=float(f@df);norm2=float(df@df);optimal=-dot/norm2;cos2=dot*dot/(p['F_squared']*norm2)
        actual_tangent_action=P@((1-base['theta'])*hm+base['theta']*hp-mu*Qcc*d)
        gn_tangent_action=P@(gn_matrix@d-mu*Qcc*d)
        diagonal_tangent_action=P@(floored*d)
        action_norm=float(np.linalg.norm(actual_tangent_action))
        trials=[]
        for t in arm['trials']:
            if 'objective' not in t:continue
            pt,ft,nt,normal=saved.point(t['control'],t['theta'],t['side_gradients'],t['objective'])
            assert abs(pt['F_squared']-t['F_squared'])<1e-15
            alpha=t['alpha'];delta=np.asarray(t['control'])-c
            assert abs(np.linalg.norm(delta)-t['actual_path_norm'])<1e-15
            assert abs(t['theta']-base['theta']-alpha*arm['delta_theta'])<1e-15
            gm=np.asarray(t['side_gradients']['-1']);gp=np.asarray(t['side_gradients']['1']);j=gp-gm
            theta_min=float(np.clip(-(gm@j)/(j@j),0,1));mixed_min=gm+theta_min*j
            flin=f+alpha*df;flux32=static_flux(np.asarray(t['control']))[1][3,2]
            trials.append({**pt,'status':t['status'],'alpha':alpha,'actual_path_norm':t['actual_path_norm'],
                'J_armijo_passed':t['J_armijo_passed'],'F_squared_armijo_passed':t['F_squared_armijo_passed'],'branch_pair_passed':t['branch_pair_passed'],
                'actual_F_squared_decrease_percent':100*(1-pt['F_squared']/p['F_squared']),
                'predicted_F_squared':float(flin@flin),'finite_F_linearization_relative_error':float(np.linalg.norm(ft-flin)/np.linalg.norm(ft)),
                'Q_y_3_2_static':float(flux32),'branch_changes_from_base':trace_changes(trace0,t['branch_trace']['-1']),
                'unapplied_saved_gradient_theta_min':theta_min,'unapplied_saved_gradient_min_F_squared':float(mixed_min@mixed_min)})
        accepted=next(t for t in trials if t['status']=='accepted')
        arms.append({'name':arm['name'],'direction_norm':float(np.linalg.norm(d)),'model_optimal_alpha':optimal,
            'model_max_F_squared_decrease_percent':100*cos2,'model_angle_degrees':float(np.degrees(np.arccos(np.clip(-dot/np.sqrt(p['F_squared']*norm2),-1,1)))),
            'constrained_tangent_HVP_action_norm':action_norm,
            'full_GN_surrogate_action_relative_error':float(np.linalg.norm(gn_tangent_action-actual_tangent_action)/action_norm),
            'diagonal_surrogate_action_relative_error':float(np.linalg.norm(diagonal_tangent_action-actual_tangent_action)/action_norm),
            'candidate_slots':len(arm['trials']),'evaluated_candidates':len(trials),'accepted':accepted,'trials':trials})
    winner=next(a for a in arms if a['name']==item['selected_direction_model'])
    assert r['current_control_sha256']==winner['accepted']['control_sha256'] and r['current_theta']==winner['accepted']['theta']
    assert r['optimizer_steps_applied']==r['accepted_iterations']==1 and len(r['hvp_history'])==r['hvp_calls_completed']==r['hvp_calls_started']==4
    final=item['final_repeat'];trial=next(t for a in item['model_comparisons'] if a['name']==winner['name'] for t in a['trials'] if t['accepted'])
    for side in ('-1','1'):np.testing.assert_array_equal(final['side_gradients'][side],trial['side_gradients'][side])
    output={'scope':'Saved-array/receipt/static flux arithmetic; no FV/HVP/AD generation; theta-min recombinations are unapplied diagnostics',
        'plan_sha256':r['plan_sha256'],'raw_sha256':hashlib.sha256(data).hexdigest(),'base':p,'base_Q_y_3_2_static':float(qbase),'arms':arms,
        'selected_direction_model':winner['name'],'selected_control_sha256':r['current_control_sha256'],'selected':winner['accepted'],
        'J_decrease_percent':100*(1-winner['accepted']['objective']/p['objective']),
        'F_squared_decrease_percent':winner['accepted']['actual_F_squared_decrease_percent'],
        'diagonal_range':[float(rawdiag.min()),float(rawdiag.max())],'floored_components':int(np.count_nonzero(rawdiag<1)),
        'jacobian_row_reverse_products':24,'direction_HVPs':4,'resource':resource,'minimum_claim':False,'response_claim':False,'FV_external_estimate':70}
    (E/'PRECOND_RESULT_20261009.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('arms','base','selected')},indent=2))

if __name__=='__main__':main()
