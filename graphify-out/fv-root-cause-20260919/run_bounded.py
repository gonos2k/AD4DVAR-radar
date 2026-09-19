"""Run one owned numerical child with sampled RSS/time limits; no shell nesting."""
import argparse, json, os, signal, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--seconds',type=float,default=600)
p.add_argument('--gib',type=float,default=8)
p.add_argument('--record',type=Path,required=True)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
start=time.monotonic(); peak=0; limit=None
with a.record.with_suffix('.log').open('w') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    while child.poll() is None:
        rss=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True).stdout.strip()
        peak=max(peak,int(rss or '0')*1024)
        if peak>a.gib*1024**3 or time.monotonic()-start>a.seconds:
            limit='rss' if peak>a.gib*1024**3 else 'time'
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
            break
        time.sleep(1)
report=dict(command=command,returncode=child.returncode,limit_reason=limit,
            sampled_peak_rss_bytes=peak,elapsed_seconds=time.monotonic()-start)
a.record.write_text(json.dumps(report,indent=2)+'\n'); print(report)
raise SystemExit(child.returncode or (1 if limit else 0))
