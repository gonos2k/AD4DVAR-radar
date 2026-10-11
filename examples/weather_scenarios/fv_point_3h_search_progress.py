"""Persist bounded candidate-search receipts without changing acceptance math."""
from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent


def search_candidates(model: Any, current: Tensor,
        evaluate: Callable[[Tensor, float, float], dict[str, Any]], *,
        chart_candidate: Callable[[Tensor, Tensor, float], Tensor],
        progress: Callable[[list[dict[str, Any]], dict[str, Any]], None],
        radius: float = 0.05, limit: int = 24
        ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= limit <= 24:
        raise ValueError('search cap must be in [1,24]')
    alphas = tangent.candidate_alphas(model, radius=radius, limit=limit)
    meta: dict[str, Any] = {'slot_cap':limit,'computed_alpha0':alphas[0] if alphas else None,
        'minimum_computed_alpha':alphas[-1] if alphas else None,'computed_alpha_count':len(alphas),
        'evaluated_count':0,'slots_started':0,'slots_completed':0,
        'evaluations_started':0,'evaluations_completed':0,
        'first_pass_candidate_index':None,'first_pass_count':0}
    trials: list[dict[str, Any]]=[]
    progress(trials,meta)
    for index, alpha in enumerate(alphas,1):
        trial: dict[str, Any]={'index':index,'alpha':alpha,'accepted':False,'status':'slot_started'}
        trials.append(trial);meta['slots_started']=index;meta['evaluated_count']=index
        progress(trials,meta)
        try:
            candidate=chart_candidate(current,model.direction,alpha)
            movement=float(torch.linalg.vector_norm(candidate-current))
            trial['actual_path_norm']=movement
            if not math.isfinite(movement) or not bool(torch.isfinite(candidate).all()):
                trial['status']='nonfinite_chart_movement_refused'
            else:
                trial.update(control=candidate.tolist(),control_sha256=tangent._tensor_sha(candidate))
                if movement<=0 or torch.equal(candidate,current):
                    trial['status']='zero_control_movement_refused'
                elif movement>radius*(1+64*torch.finfo(current.dtype).eps):
                    trial['status']='actual_chart_radius_refused'
                else:
                    trial['status']='evaluation_started';meta['evaluations_started']+=1
                    progress(trials,meta)
                    facts=evaluate(candidate,model.theta,alpha)
                    meta['evaluations_completed']+=1
                    trial.update(facts)
                    trial['accepted']=all(facts.get(key) is True for key in (
                        'mixing_minimum_valid','J_armijo_passed','F_squared_armijo_passed',
                        'branch_pair_passed','face_audit_passed','side_objectives_match_native',
                        'side_gradients_finite','finite'))
                    trial['status']='accepted' if trial['accepted'] else 'gate_refused'
            meta['slots_completed']+=1
            if trial['accepted']:
                meta['first_pass_candidate_index']=index;meta['first_pass_count']=index
            progress(trials,meta)
        except Exception as error:
            trial['accepted']=False
            trial['status']='slot_interrupted' if isinstance(error,TimeoutError) else 'slot_failed'
            trial['error_type']=type(error).__name__;trial['error_message']=str(error)
            progress(trials,meta)
            raise
        if trial['accepted']:
            return trial,trials,meta
    return None,trials,meta
