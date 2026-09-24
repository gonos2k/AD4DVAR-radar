"""The approved single 20-minute / sampled 2-GiB reanalysis experiment."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from examples.weather_scenarios.fv86_resource_runner import run_guarded


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    stem=args.directory/'fv86_reanalysis'
    paths=[stem.with_suffix(suffix) for suffix in ('.json','.resource.json','.log')]
    if any(path.exists() for path in paths):
        raise FileExistsError('preserve the existing reanalysis run before another approved execution')
    args.directory.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(Path(__file__).with_name('fv86_reanalysis_probe.py')),
             '--output',str(paths[0])]
    result=run_guarded(command,wall_seconds=1200,rss_bytes=2*1024**3,
                       report_path=paths[1],log_path=paths[2])
    print(result)
