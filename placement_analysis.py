#!/usr/bin/env python3
"""Fixed-volume AlN placement ablation: no guard may overlap a TSV strip."""
from pathlib import Path
import json,csv,time,argparse,hashlib
import numpy as np
from core import *
from physical import FIELD_KEYS,optical_from_fields,dumpcsv

ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--validation',action='store_true');args=ap.parse_args();prefix='placement_validation' if args.validation else 'placement'
 out=ROOT/'results';c=json.loads((ROOT/'code/modal_coefficients.json').read_text());rr=[];values={k:[] for k in ['phi_rad','H_rad_per_mW','temperature_background_K','temperature_H_K_per_mW']};allfields={f'{p}_{k}':[] for p in ['background','heater'] for k in FIELD_KEYS}
 specs=[(d,l,r,120,1e-8,170.) for d in ['AlN','graded_W_AlN'] for l in [180.,320.,460.] for r in [600.,740.,880.]]
 if args.validation:
  specs=[(d,l,r,nx,1e-8,170.) for d in ['AlN','graded_W_AlN'] for l,r in [(320.,600.),(460.,740.),(180.,880.)] for nx in [180,240]]
  specs += [(d,320.,600.,120,R,170.) for d in ['AlN','graded_W_AlN'] for R in [1e-7,1e-6]]
  specs += [(d,320.,600.,120,R,k) for d in ['AlN','graded_W_AlN'] for R in [1e-8,1e-7] for k in [36.,260.]]
 tstart=time.time();energy_max=0.;eq_max=0.
 for design,left,right,nx,R,kAlN in specs:
  g=geometry(nx,nx//2,package='stacked',design=design,scales={'guard_centers_um':[left,right],'AlN_k':kAlN/170.});op=ThermalOperator(g,R);mech=ElasticOperator(g);h=heater_sources(g);tt=op.solve(np.concatenate([g['q'][None],h]));fs=[];stress=[]
  loads=np.concatenate([g['q'][None],h])
  for j,t in enumerate(tt):
   m=mech.solve(t);energy_max=max(energy_max,op.energy(loads[j],t)['energy_relative_error']);eq_max=max(eq_max,m['equilibrium_relative_residual']);fs.append(sample_fields(g,t,m));stress.append(float(np.quantile(m['von_mises'],.95)/1e6))
  bg=fs[0];hs={k:np.stack([z[k] for z in fs[1:]]) for k in FIELD_KEYS};phi,*_=optical_from_fields(bg,c);H,*_=optical_from_fields(hs,c);H=H.T
  meta=dict(case_id=f'{design}_L{left:g}_R{right:g}'+(f'_n{nx}_RK{R:.0e}_k{kAlN:g}' if args.validation else ''),design=design,Rb_m2K_W=R,k_AlN_W_mK=kAlN,nx=nx,ny=nx//2,kind=prefix,left_guard_um=left,right_guard_um=right)
  row=dict(**meta,Tmax_C=float(tt[0].max()+25),Tchannel_mean_C=float(bg['temperature_K'].mean()+25),phi_mean_rad=float(phi.mean()),phi_spread_rad=float(np.ptp(phi)),H_diag_mean_rad_per_mW=float(np.diag(H).mean()),stress_p95_MPa=stress[0]);rr.append(row)
  for k,v in [('phi_rad',phi),('H_rad_per_mW',H),('temperature_background_K',bg['temperature_K']),('temperature_H_K_per_mW',hs['temperature_K'].T)]:values[k].append(v)
  for k in FIELD_KEYS:allfields['background_'+k].append(bg[k]);allfields['heater_'+k].append(hs[k])
  print(row,flush=True)
 payload={k:np.array([r[k] for r in rr]) for k in meta};payload.update({k:np.array(v) for k,v in values.items()});np.savez_compressed(out/(prefix+'_inputs.npz'),**payload)
 raw={k:payload[k] for k in meta};raw.update({k:np.array(v) for k,v in allfields.items()});np.savez_compressed(out/(prefix+'_physical_inputs.npz'),**raw);dumpcsv(out/(prefix+'_summary.csv'),rr)

 metadata=dict(cases=len(specs),loads_per_case=10,case_loads=10*len(specs),energy_max_relative_error=energy_max,equilibrium_max_relative_residual=eq_max,elapsed_seconds=time.time()-tstart,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('core.py')]})
 (out/(prefix+'_metadata.json')).write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
