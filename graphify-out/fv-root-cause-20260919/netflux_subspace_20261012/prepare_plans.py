"""Freeze current numerical sources and inherited evidence before one launch."""
from pathlib import Path
import json
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from examples.weather_scenarios import fv_point_3h_netflux_subspace_trial as trial
from examples.weather_scenarios import fv_point_3h_netflux_comparison_endpoint as endpoint
from examples.weather_scenarios import fv_point_3h_netflux_future_observation as observer
base=endpoint.load_endpoint();parent=base['parent_plan'];sha=trial.comparison._sha
WORK=Path(__file__).resolve().parent

def freeze(path,kind,policy,sources,extra):
    if path.exists():raise ValueError(f'plan already frozen: {path}')
    plan={'experiment_kind':kind,'numerical_version':trial.VERSION,'policy':policy,
        **extra,'source_files':{n:sha(ROOT/n) for n in sorted(set(parent['source_files'])|sources)},
        'archive_files':{**parent['archive_files'],**{p.relative_to(ROOT).as_posix():sha(p) for p in endpoint.RECEIPTS}}}
    path.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    path.with_name(path.stem+'_SHA256.txt').write_text(sha(path)+'\n')
    return plan

p=freeze(trial.PLAN,'fresh_point_two_direction_trust_radius',trial.POLICY,trial.NEW_SOURCES,
    {'base_control_sha256':trial.BASE_SHA})
with tarfile.open(WORK/'source.tar.gz','w:gz') as archive:
    for name in p['source_files']:archive.add(ROOT/name,arcname=name)
trial.load_plan(trial.PLAN,sha(trial.PLAN))
score_sources={endpoint.SELF,endpoint.TEST,observer.SELF,observer.TEST,(WORK/'observe_d19e.py').relative_to(ROOT).as_posix()}
freeze(WORK/'OBSERVATION_PLAN.json','observation_only_b647_d19e',observer.POLICY,score_sources,
    {'start_control_sha256':trial.comparison.BASE_SHA,'final_control_sha256':trial.BASE_SHA,
    'parameters_sha256':observer.PARAMETERS_SHA,'truth_sha256':observer.TRUTH_SHA,'score_profile':observer.SCORE_PROFILE})
print(json.dumps({'plan_sha256':sha(trial.PLAN),'score_plan_sha256':sha(WORK/'OBSERVATION_PLAN.json'),
    'source_bundle_sha256':sha(WORK/'source.tar.gz'),'source_count':len(p['source_files']),'archive_count':len(p['archive_files'])},indent=2))
