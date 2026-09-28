#!/usr/bin/env python3
"""Optical model-form and photoelastic sensitivity without re-solving FEM.

Two effective-index orderings × three photoelastic scales × (four nominal
material ablations + eighteen placements) = 132 reported evaluations.
Nominal placement(320,880) repeats the main AlN geometry for a transparent
within-study cross-check; hence 120 unique physical/optical combinations.
"""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from physical import FIELD_KEYS,optical_from_fields,dumpcsv
ROOT=Path(__file__).resolve().parents[1]

def main():
 out=ROOT/'results';main=np.load(out/'physical_inputs.npz');placement=np.load(out/'placement_physical_inputs.npz');spec=[]
 for i in range(len(main['case_id'])):
  R=main['Rb_m2K_W'][i];design=str(main['design'][i]);okR=R==1e-8 if 'AlN' in design else R==0
  if main['nx'][i]==120 and main['k_AlN_W_mK'][i]==170 and okR:spec.append((main,i))
 spec.extend((placement,i) for i in range(len(placement['case_id'])))
 values={k:[] for k in ['phi_rad','H_rad_per_mW','phi_thermal_rad','phi_photoelastic_rad','phi_shape_rad','temperature_background_K','temperature_H_K_per_mW']};rr=[];metadata=[]
 for mode in ['vertical_first','horizontal_first']:
  coeff_path=ROOT/'code'/('modal_coefficients.json' if mode=='vertical_first' else 'modal_coefficients_horizontal_first.json');original=json.loads(coeff_path.read_text())
  for scale in [.5,1.,1.5]:
   c=dict(original)
   for axis in 'xyz':c['dn_eff_d_elastic_strain_'+axis]*=scale
   for data,i in spec:
    bg={k:data['background_'+k][i] for k in FIELD_KEYS};hh={k:data['heater_'+k][i] for k in FIELD_KEYS};phi,th,pe,sh=optical_from_fields(bg,c);H,*_=optical_from_fields(hh,c);H=H.T
    src=str(data['case_id'][i]);meta=dict(case_id=src+f'__{mode}_pe{scale:g}',physical_case_id=src,design=str(data['design'][i]),Rb_m2K_W=float(data['Rb_m2K_W'][i]),k_AlN_W_mK=float(data['k_AlN_W_mK'][i]),nx=int(data['nx'][i]),ny=int(data['ny'][i]),kind='optical_sensitivity',mode_order=mode,photoelastic_scale=scale)
    metadata.append(meta);rr.append(dict(**meta,phi_mean_rad=float(phi.mean()),phi_spread_rad=float(np.ptp(phi)),thermal_mean_rad=float(th.mean()),photoelastic_mean_rad=float(pe.mean()),shape_mean_rad=float(sh.mean()),H_diag_mean_rad_per_mW=float(np.diag(H).mean())))
    for k,v in [('phi_rad',phi),('H_rad_per_mW',H),('phi_thermal_rad',th),('phi_photoelastic_rad',pe),('phi_shape_rad',sh),('temperature_background_K',bg['temperature_K']),('temperature_H_K_per_mW',hh['temperature_K'].T)]:values[k].append(v)
 payload={k:np.array([r[k] for r in metadata]) for k in metadata[0]};payload.update({k:np.array(v) for k,v in values.items()});np.savez_compressed(out/'optical_sensitivity_inputs.npz',**payload);dumpcsv(out/'optical_sensitivity_summary.csv',rr)
 comparisons=[]
 for mode in ['vertical_first','horizontal_first']:
  for scale in [.5,1,1.5]:
   rows=[r for r in rr if r['mode_order']==mode and r['photoelastic_scale']==scale]
   find=lambda cid:next(r for r in rows if r['physical_case_id']==cid)
   cu=find('Cu_R0e+00_k170_n120');best=find('AlN_L320_R600');bw=find('graded_W_AlN_L320_R600');placements=[r for r in rows if '_L' in r['physical_case_id']];winner=min(placements,key=lambda r:r['phi_spread_rad'])
   comparisons.append(dict(mode_order=mode,photoelastic_scale=scale,Cu_spread_rad=cu['phi_spread_rad'],AlN_320_600_spread_rad=best['phi_spread_rad'],WAlN_320_600_spread_rad=bw['phi_spread_rad'],AlN_320_600_vs_Cu_reduction_percent=100*(1-best['phi_spread_rad']/cu['phi_spread_rad']),minimum_spread_placement=winner['physical_case_id'],minimum_spread_rad=winner['phi_spread_rad']))
 dumpcsv(out/'optical_sensitivity_comparisons.csv',comparisons)
 report=dict(evaluations=len(rr),source_cases=len(spec),optical_assumptions=6,scope='Deterministic model-form and parameter sensitivity, not a calibrated uncertainty distribution or probability',AlN_320_600_beats_Cu_all_six=all(r['AlN_320_600_spread_rad']<r['Cu_spread_rad'] for r in comparisons),comparisons=comparisons,
  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'code/modal_coefficients.json',ROOT/'code/modal_coefficients_horizontal_first.json',out/'physical_inputs.npz',out/'placement_physical_inputs.npz']})
 (out/'optical_sensitivity_metadata.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
