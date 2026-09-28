#!/usr/bin/env python3
"""Finite-mesh sensitivity and explicit 0.25-rad target of the vector cross-check.
Run after vector_mode.py. No Maxwell eigenproblem is repeated by this script.
"""
from pathlib import Path
import json,csv
import numpy as np
from physical import FIELD_KEYS,optical_from_fields
from control_analysis import optimize,write_csv
ROOT=Path(__file__).resolve().parents[1]
def run():
 out=ROOT/'results/vector';rows=[];commands=[]
 for mesh in [10.,5.,2.5]:
  c=json.loads((out/f'vector_coefficients_{mesh:g}nm_0.0001.json').read_text())
  for name in ['physical_inputs.npz','placement_physical_inputs.npz']:
   source=np.load(ROOT/'results'/name)
   for i in range(len(source['case_id'])):
    design=str(source['design'][i]);R=float(source['Rb_m2K_W'][i]);cid=str(source['case_id'][i])
    if name=='physical_inputs.npz' and not(source['nx'][i]==120 and source['k_AlN_W_mK'][i]==170 and (R==1e-8 if 'AlN' in design else R==0)):continue
    bg={k:source['background_'+k][i] for k in FIELD_KEYS};hh={k:source['heater_'+k][i] for k in FIELD_KEYS}
    phi,*_=optical_from_fields(bg,c);H,*_=optical_from_fields(hh,c);H=H.T
    reserve=9*20/255/2
    r=optimize(phi,H,20,100-reserve,robust=True,bits=8,eta=.02,eps_rad=.002,target_rad=.25)
    row=dict(mesh_nm=mesh,case_id=cid,passive_spread_rad=float(np.ptp(phi)),zero_power_Cpost_rad=float(np.ptp(phi)+.004),
      zero_power_meets_0p25=bool(np.ptp(phi)+.004<=.25),target_rad=.25,minimum_power_feasible=r['feasible'])
    if r['feasible']:
     row.update(quantized_power_mW=r['quantized_sum_mW'],Cpost_rad=r['posterior_certificate_rad'],Cpre_rad=r['certificate']['bound_rad'])
     for j,(p,pq) in enumerate(zip(r['p_mW'],r['p_quantized_mW'])):commands.append(dict(mesh_nm=mesh,case_id=cid,heater=j,continuous_mW=p,quantized_mW=pq))
    rows.append(row)
 write_csv(out/'vector_target025.csv',rows);write_csv(out/'vector_target025_commands.csv',commands)
 selected=[r for r in rows if r['case_id']=='AlN_L320_R600'];s=dict(target_rad=.25,selected_layout_mesh_sensitivity=selected,
   zero_power_target_fails_all_three_meshes=all(not r['zero_power_meets_0p25'] for r in selected),
   statement='The EIM zero-power 0.25-rad result is not preserved by the independent full-vector optical model at any of the three derivative meshes; active heating is necessary under the stated sensing margin and branch.')
 (out/'vector_target025_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
if __name__=='__main__':run()
