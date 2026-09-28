#!/usr/bin/env python3
"""Reproduce package thermomechanical fields and physical heater influence.

python physical.py --out ../results
No proprietary FEM, GPU, commercial mesher or empirical phase fit is required.
All powers are per invariant z length; the optical length is 1 mm.
"""
from __future__ import annotations
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
from core import *

FIELD_KEYS=['temperature_K','total_strain_x','total_strain_y','total_strain_z','elastic_strain_x','elastic_strain_y','elastic_strain_z','stress_x_Pa','stress_y_Pa','stress_z_Pa']
R_VALUES=[0.,1e-10,1e-9,1e-8,3e-8,1e-7,3e-7,1e-6]

def dumpcsv(path,rows):
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def optical_from_fields(fields,c):
 """Linear modal perturbation of n_eff; no longitudinal length strain."""
 factor=2*np.pi*c.get('path_length_m',1e-3)/c.get('wavelength_m',1550e-9)
 thermal=c['dn_eff_dT_K']*fields['temperature_K']
 photo=sum(c['dn_eff_d_elastic_strain_'+v]*fields['elastic_strain_'+v] for v in 'xyz')
 shape=sum(c['dn_eff_d_total_strain_'+v]*fields['total_strain_'+v] for v in 'xy')
 return factor*(thermal+photo+shape),factor*thermal,factor*photo,factor*shape

def verify():
 # Exact resistance ladder, power deposited in topmost cell, adiabatic top.
 g=geometry(12,80);g['k'][:]=130.;g['labels'][:40,:]='Si';g['labels'][40:,:]='AlN';g['q'][:]=0
 flux=1e4;g['q'][-1,:]=flux/g['dy'];R=3e-7
 op=ThermalOperator(g,R,h_top=0.);t=op.solve(g['q'])
 exact=flux*(1/CONFIG['h_bottom_W_m2K']+(CONFIG['height_m']-g['dy']/2)/130.+R)
 err=float(np.max(abs(t[-1,:]-exact))/exact)
 # Recover previous ideal-interface solver, then stress/strain solution.
 g=geometry(40,20,design='graded_W_AlN');op=ThermalOperator(g,0);t=op.solve(g['q']);old,e=thermal(g)
 tr=float(np.max(abs(t-old))/np.max(abs(old)));m=ElasticOperator(g).solve(t);oldm=elasticity(g,t)
 mr=float(np.max(abs(m['u']-oldm['u']))/np.max(abs(oldm['u'])))
 p=geometry(12,8)
 for key in ['E','nu','alpha']:p[key][:]=MATERIALS['Si'][key]
 patch=elasticity(p,np.zeros((8,12)),patch=True)
 heat=ElasticOperator(p).solve(np.ones((8,12))*10)
 expect=(1+MATERIALS['Si']['nu'])*MATERIALS['Si']['alpha']*10
 te=float(np.max(abs(heat['strain'][:,:,:2]-expect))/expect)
 # Superposition and physical 1 W/m source normalization.
 h=heater_sources(g);u=np.arange(1,10)*.1;t0=op.solve(g['q']);tj=op.solve(h)
 td=op.solve(g['q']+np.einsum('j,jyx->yx',u,h));tl=t0+np.einsum('j,jyx->yx',u,tj)
 se=float(np.max(abs(td-tl))/np.max(abs(td)))
 v=dict(interface_resistance_analytic_relative_error=err,ideal_interface_vs_original_relative_error=tr,
  cached_mechanics_vs_original_relative_error=mr,affine_patch_relative_error=patch['patch_displacement_relative_error'],
  free_plane_strain_uniform_heating_relative_error=te,linear_thermal_superposition_relative_error=se,
  heater_normalization_max_absolute_error_W_per_m=float(np.max(abs(h.sum(axis=(1,2))*g['dx']*g['dy']-1.))))
 assert max(v.values())<1e-9,v
 return v

def cases():
 a=[]
 def add(design,R,k=170.,nx=120,kind='interface'):
  key=(design,R,k,nx)
  if any(z['_key']==key for z in a):return
  a.append(dict(_key=key,case_id=f'{design}_R{R:.0e}_k{k:g}_n{nx}',design=design,Rb_m2K_W=R,k_AlN_W_mK=k,nx=nx,ny=nx//2,kind=kind))
 for d in DESIGNS:
  for R in R_VALUES if 'AlN' in d else [0.]:add(d,R)
 for d in ['AlN','graded_W_AlN']:
  for k in [36.,80.,260.]:
   for R in [1e-8,1e-7,1e-6]:add(d,R,k,kind='conductivity')
 for nx in [60,180,240]:
  for d in DESIGNS:add(d,1e-8 if 'AlN' in d else 0.,nx=nx,kind='mesh')
 for z in a:z.pop('_key')
 return a

def derive(out,c):
 data=np.load(out/'physical_inputs.npz');bg={k:data['background_'+k] for k in FIELD_KEYS};hh={k:data['heater_'+k] for k in FIELD_KEYS}
 phi,th,pe,sh=optical_from_fields(bg,c);H,Hth,Hpe,Hsh=optical_from_fields(hh,c)
 # Inputs heater axis is case,source,observation; desired H is case,observation,source.
 payload={k:data[k] for k in ['case_id','design','Rb_m2K_W','k_AlN_W_mK','nx','ny','kind']}
 payload.update(phi_rad=phi,H_rad_per_mW=H.transpose(0,2,1),phi_thermal_rad=th,phi_photoelastic_rad=pe,phi_shape_rad=sh,
  H_thermal_rad_per_mW=Hth.transpose(0,2,1),H_photoelastic_rad_per_mW=Hpe.transpose(0,2,1),H_shape_rad_per_mW=Hsh.transpose(0,2,1),
  temperature_background_K=bg['temperature_K'],temperature_H_K_per_mW=hh['temperature_K'].transpose(0,2,1))
 np.savez_compressed(out/'control_inputs.npz',**payload)
 rows=[]
 for i,cid in enumerate(data['case_id']):
  row={k:data[k][i].item() for k in ['case_id','design','Rb_m2K_W','k_AlN_W_mK','nx','ny','kind']}
  hi=payload['H_rad_per_mW'][i]
  row.update(phi_mean_rad=float(phi[i].mean()),phi_spread_rad=float(np.ptp(phi[i])),phase_thermal_mean_rad=float(th[i].mean()),
   phase_photoelastic_mean_rad=float(pe[i].mean()),phase_shape_mean_rad=float(sh[i].mean()),
   H_diag_mean_rad_per_mW=float(np.diag(hi).mean()),H_smallest_singular_value=float(np.linalg.svd(hi,compute_uv=False)[-1]),
   H_condition_number=float(np.linalg.cond(hi)))
  rows.append(row)
 dumpcsv(out/'optical_physical_summary.csv',rows)
 (out/'used_modal_coefficients.json').write_text(json.dumps(c,indent=2))
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results');ap.add_argument('--coefficients',type=Path,default=Path(__file__).with_name('modal_coefficients.json'));ap.add_argument('--derive-only',action='store_true');args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
 if args.derive_only:
  derive(out,json.loads(args.coefficients.read_text()));return
 ver=verify();(out/'physical_verification.json').write_text(json.dumps(ver,indent=2)); print('VERIFIED',ver,flush=True)
 cc=cases();samples={f'{p}_{k}':[] for p in ['background','heater'] for k in FIELD_KEYS};summaries=[];cache={};tstart=time.time()
 for case in cc:
  d,R,k,nx,ny=[case[v] for v in ['design','Rb_m2K_W','k_AlN_W_mK','nx','ny']]
  key=(d,k,nx)
  if key not in cache:
   # Keep few cached stiffness factors: factorization is the major memory cost.
   cache.clear();g=geometry(nx,ny,package='stacked',design=d,scales={'AlN_k':k/170.});cache[key]=(g,ElasticOperator(g))
  g,mech=cache[key];op=ThermalOperator(g,R);h=heater_sources(g);loads=np.concatenate([g['q'][None,:,:],h],axis=0);tt=op.solve(loads)
  sampled=[];energy=[];eq=[]
  for j,t in enumerate(tt):
   m=mech.solve(t);sampled.append(sample_fields(g,t,m));energy.append(op.energy(loads[j],t)['energy_relative_error']);eq.append(m['equilibrium_relative_residual'])
   if j==0:
    summary=dict(**case,Tmax_C=float(t.max()+25),Tchannel_mean_C=float(sampled[-1]['temperature_K'].mean()+25),Tchannel_spread_K=float(np.ptp(sampled[-1]['temperature_K'])),
     stress_max_MPa=float(m['von_mises'].max()/1e6),stress_p95_MPa=float(np.quantile(m['von_mises'],.95)/1e6),
     top_displacement_range_um=float(np.ptp(m['u'][-1,:,1])*1e6))
    if nx==120 and k==170 and R in [0.,1e-8,1e-6]:
     np.savez_compressed(out/f'fields_{case["case_id"]}.npz',x_m=g['x'],y_m=g['y'],labels=g['labels'],temperature_C=t+25,stress_von_mises_Pa=m['von_mises'],strain=m['strain'],stress=m['stress'],sigma_z=m['sigma_z'],u=m['u'])
  for fld in FIELD_KEYS:
   samples['background_'+fld].append(sampled[0][fld]);samples['heater_'+fld].append(np.stack([ss[fld] for ss in sampled[1:]]))
  summary.update(energy_max_relative_error=max(energy),equilibrium_max_relative_residual=max(eq));summaries.append(summary)
  print(f'{len(summaries)}/{len(cc)} {case["case_id"]} Tmax={summary["Tmax_C"]:.5f} elapsed={time.time()-tstart:.1f}s',flush=True)
 # NPZ stores mechanics and thermal data independent of modal assumptions.
 payload={k:np.array([c[k] for c in cc]) for k in cc[0]};payload.update({k:np.array(v) for k,v in samples.items()});payload['channel_x_m']=CHANNEL_X;payload['channel_y_m']=CHANNEL_Y
 np.savez_compressed(out/'physical_inputs.npz',**payload);dumpcsv(out/'physical_summary.csv',summaries)
 meta=dict(model='2D x-y thermal + plane-strain elasticity; invariant z optical propagation',path_length_m=1e-3,
  power_normalization='1 W/m line heating equals 1 mW over the assumed 1 mm invariant length',
  source_shape=dict(sigma_x_m=HEATER_SIGMA_X,y_bottom_m=HEATER_BOTTOM,y_top_m=HEATER_TOP),
  contact_resistance='AlN versus non-AlN cell faces only; one R_K contribution per face',
  cases=len(cc),loads_per_case=10,physics_case_loads=len(cc)*10,elapsed_seconds=time.time()-tstart,
  material_properties=MATERIALS,geometry_config=CONFIG,
  energy_max_relative_error=max(z['energy_max_relative_error'] for z in summaries),equilibrium_max_relative_residual=max(z['equilibrium_max_relative_residual'] for z in summaries),
  verification=ver,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('core.py')]})
 (out/'physical_metadata.json').write_text(json.dumps(meta,indent=2))
 if args.coefficients.exists():derive(out,json.loads(args.coefficients.read_text()))
 else:print('Modal coefficient file not available: run --derive-only after it is supplied.',flush=True)

if __name__=='__main__':main()
