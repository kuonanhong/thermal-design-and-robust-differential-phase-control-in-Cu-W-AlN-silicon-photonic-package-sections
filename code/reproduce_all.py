#!/usr/bin/env python3
"""Regenerate the entire scientific dataset with the current Python interpreter.

Run from any working directory: python /path/to/package/code/reproduce_all.py
Use --verify-only to check the supplied arrays without recomputing all fields.
Outputs are written beneath this package. No network or proprietary solver is used.
"""
from __future__ import annotations
import argparse,subprocess,sys,time,json,platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');args=p.parse_args()
 tasks=[] if args.verify_only else [
 ['code/optical_mode.py'],['code/physical.py'],['code/placement_analysis.py'],
 ['code/placement_analysis.py','--validation'],['code/optical_sensitivity.py'],
 ['code/control_analysis.py'],['code/placement_control.py'],
 ['code/placement_control.py','--inputs','results/placement_validation_inputs.npz','--out','results/placement_control_validation','--no-plots'],
 ['code/plot_physical.py'],['code/plot_placement.py'],
 ['code/workload_robust.py'],['code/vector_mode.py'],['code/vector_target_study.py']]
 tasks += [['code/control_analysis.py','--verify-only'],['research/independent_checks.py','--root',str(ROOT)],
           ['code/workload_independent_check.py','--root',str(ROOT)],
           ['code/vector_independent_check.py','--root',str(ROOT)]]
 logdir=ROOT/'results'/'reproduction_logs';logdir.mkdir(parents=True,exist_ok=True)
 records=[];start=time.time()
 for i,task in enumerate(tasks,1):
  logfile=logdir/f'{i:02d}_{Path(task[0]).stem}.log'
  print(f'[{i}/{len(tasks)}] {" ".join(task)}',flush=True);t=time.time()
  with logfile.open('w',encoding='utf8') as log:
   rc=subprocess.run([sys.executable,*task],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
  records.append({'command':[sys.executable,*task],'return_code':rc,'seconds':time.time()-t,'log':str(logfile.relative_to(ROOT))})
  if rc:
   print(logfile.read_text()[-5000:]);raise SystemExit(rc)
 # Frozen-data checks do not import the optional vector geometry dependency.
 # Record its absence without failing after all scientific checks have passed.
 environment={'python':platform.python_version()}
 for package in ['numpy','scipy','matplotlib','shapely']:
  try:environment[package]=version(package)
  except PackageNotFoundError:environment[package]='not installed'
 result=dict(all_passed=True,verify_only=args.verify_only,elapsed_seconds=time.time()-start,
  environment=environment,steps=records)
 path=ROOT/'results'/('reproduction_verify.json' if args.verify_only else 'reproduction_manifest.json');path.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='steps'},indent=2))
if __name__=='__main__':main()
