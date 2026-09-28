#!/usr/bin/env python3
"""Guard-placement optical-control study using placement_analysis.py loads.

The same calibrated-model assumptions, power limits, phase uncertainty and
finite DAC are used for every geometry. Geometry optimization is a finite
3x3 design-grid comparison, not a claimed global continuous optimum.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from control_analysis import optimize, pair_range, write_csv, mzi_penalty_dB


def run(inputs,out,make_plots=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=np.load(inputs,allow_pickle=False);rows=[];commands=[]
    pmax=20.;bits=8;eta=.02;eps=.002;reserve=9*pmax/(2**bits-1)/2
    for j in range(len(d['case_id'])):
        phi,H=d['phi_rad'][j],d['H_rad_per_mW'][j]
        meta=dict(case_id=str(d['case_id'][j]),design=str(d['design'][j]),
            left_guard_um=float(d['left_guard_um'][j]),right_guard_um=float(d['right_guard_um'][j]),
            Rb_m2K_W=float(d['Rb_m2K_W'][j]),k_AlN_W_mK=float(d['k_AlN_W_mK'][j]),
            nx=int(d['nx'][j]),passive_pair_rad=pair_range(phi),
            passive_posterior_certificate_rad=pair_range(phi)+2*eps,
            background_mean_phase_rad=float(phi.mean()),
            background_mean_temperature_rise_K=float(d['temperature_background_K'][j].mean()),
            per_heater_max_mW=pmax,DAC_bits=bits,relative_H_bound=eta,sensor_bound_rad=eps)
        tasks=[('minimax',budget,None,robust) for budget in [25.,50.,100.] for robust in [False,True]]
        tasks += [('minimum_power',9*pmax,target,True) for target in [.05,.10,.25,.50]]
        for objective,budget,target,robust in tasks:
            r=optimize(phi,H,pmax,budget-reserve,robust=robust,bits=bits,eta=eta,
                       eps_rad=eps,target_rad=target)
            row=dict(**meta,objective=objective,mode='robust' if robust else 'nominal',
                     total_budget_mW=budget,target_pair_rad=target,feasible=r['feasible'])
            if r['feasible']:
                row.update({k:r[k] for k in ['continuous_sum_mW','quantized_sum_mW','nominal_continuous_pair_rad',
                    'nominal_quantized_pair_rad','posterior_certificate_rad','LP_constraint_violation_rad']})
                row.update(certificate_rad=r['certificate']['bound_rad'],
                    ideal_MZI_penalty_bound_dB=mzi_penalty_dB(r['posterior_certificate_rad']),
                    mean_temperature_rise_with_heating_K=float(np.mean(d['temperature_background_K'][j]+d['temperature_H_K_per_mW'][j]@r['p_quantized_mW'])),
                    rounded_budget_violation_mW=max(0.,r['quantized_sum_mW']-budget))
                for h,(p,pq) in enumerate(zip(r['p_mW'],r['p_quantized_mW'])):
                    commands.append(dict(case_id=meta['case_id'],objective=objective,mode=row['mode'],total_budget_mW=budget,
                        target_pair_rad=target,heater=h,continuous_power_mW=float(p),quantized_power_mW=float(pq)))
            rows.append(row)
    write_csv(out/'placement_control.csv',rows);write_csv(out/'placement_control_commands.csv',commands)
    primary=[r for r in rows if r['objective']=='minimax' and r['mode']=='robust' and r['total_budget_mW']==100.]
    summary=dict(number_of_layouts=len(d['case_id']),optimization_rows=len(rows),command_rows=len(commands),
        best_temperature_layout=min(primary,key=lambda r:r['background_mean_temperature_rise_K']),
        best_passive_phase_layout=min(primary,key=lambda r:r['passive_pair_rad']),
        best_control_certificate_layout=min(primary,key=lambda r:r['posterior_certificate_rad']),
        zero_power_meets_0p25_certificate=[r['case_id'] for r in rows if r['objective']=='minimum_power' and
            r['target_pair_rad']==.25 and r['feasible'] and r['quantized_sum_mW']<1e-9],
        maximum_rounded_budget_violation_mW=max(r.get('rounded_budget_violation_mW',0) for r in rows),
        maximum_LP_constraint_violation_rad=max(r.get('LP_constraint_violation_rad',0) for r in rows))
    (out/'placement_control_summary.json').write_text(json.dumps(summary,indent=2))
    if make_plots:plot(primary,out)
    return summary


def plot(primary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11.5,'axes.labelsize':11.5,'xtick.labelsize':11.5,'ytick.labelsize':11.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    fig,axs=plt.subplots(2,3,figsize=(10.5,7.5))
    keys=['background_mean_temperature_rise_K','passive_pair_rad','posterior_certificate_rad']
    labels=['mean ΔT (K)','passive phase (rad)','$C_{post}$ (rad)']
    for ii,design in enumerate(['AlN','graded_W_AlN']):
        rr=[r for r in primary if r['design']==design]
        left=sorted(set(r['left_guard_um'] for r in rr));right=sorted(set(r['right_guard_um'] for r in rr))
        for jj,(key,label) in enumerate(zip(keys,labels)):
            ax=axs[ii,jj];A=np.full((len(right),len(left)),np.nan)
            for r in rr:A[right.index(r['right_guard_um']),left.index(r['left_guard_um'])]=r[key]
            low=min(r[key] for r in primary);high=max(r[key] for r in primary)
            im=ax.imshow(A,origin='lower',cmap='viridis',vmin=low,vmax=high,aspect='equal')
            for a in range(len(right)):
                for b in range(len(left)):
                    color='white' if A[a,b]<(low+high)/2 else 'black'
                    ax.text(b,a,f'{A[a,b]:.3f}',ha='center',va='center',color=color,fontsize=12)
            best=np.unravel_index(np.argmin(A),A.shape);ax.plot(best[1],best[0],'s',ms=38,mfc='none',mec='#ef7d00',mew=2)
            ax.set(xticks=range(len(left)),xticklabels=[f'{x:g}' for x in left],
                   yticks=range(len(right)),yticklabels=[f'{x:g}' for x in right],xlabel='Left guard center (µm)',ylabel='Right guard center (µm)')
            ax.set_title(f"({'abcdef'[ii*3+jj]}) {'AlN' if design=='AlN' else 'W + AlN'}: {label}",fontsize=11.5)
            fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'placement_optical_control.{ext}',dpi=230)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();base=Path(__file__).resolve().parents[1]
    p.add_argument('--inputs',type=Path,default=base/'results/placement_inputs.npz');p.add_argument('--out',type=Path,default=base/'results');p.add_argument('--no-plots',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.inputs,a.out,not a.no_plots),indent=2))
if __name__=='__main__':main()
