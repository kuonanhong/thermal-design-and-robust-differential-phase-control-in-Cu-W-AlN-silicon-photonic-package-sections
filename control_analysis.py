#!/usr/bin/env python3
"""Phase-difference control with material-dependent physical heater responses.

Uses the linear package responses supplied by physical.py, in rad/mW.
All optimization is on one continuous, unwrapped phase-locking branch. No
integer/modulo phase search is claimed. Robustness is relative to explicit
independent interval uncertainties, not an experimentally measured yield.

Run: python control_analysis.py --inputs ../results/control_inputs.npz --out ../results
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog


def pair_operators(phi, H):
    phi, H = np.asarray(phi, float), np.asarray(H, float)
    if phi.ndim != 1 or H.shape[0] != len(phi) or H.ndim != 2:
        raise ValueError('Expected phi[n] and H[n, number_of_heaters].')
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(H)):
        raise ValueError('Finite phase and response entries are required.')
    i, k = np.triu_indices(len(phi), 1)
    return i, k, phi[i] - phi[k], H[i] - H[k], abs(H[i]) + abs(H[k])


def uncertainty_terms(phi, H, pmax_mW, bits=8, eta=0.02, eps_rad=0.002, uncertainty_mode="entrywise"):
    """Return pair LP terms, including gain x quantization cross term.

    Each sensed background phase has absolute error <= eps_rad. Every true
    response entry lies in H_ij +/- eta*abs(H_ij), independently. Endpoint DAC
    spacing q_j=Pmax_j/(2**bits-1) ensures |round(p)_j-p_j| <= q_j/2.
    """
    i, k, b, D, S = pair_operators(phi, H)
    upper = np.broadcast_to(np.asarray(pmax_mW, float), H.shape[1]).copy()
    if np.any(upper <= 0) or bits < 1 or eta < 0 or eps_rad < 0:
        raise ValueError('Positive heater bounds, bits>=1 and nonnegative errors required.')
    q = upper / (2**bits - 1)
    if uncertainty_mode not in ['entrywise','column_gain']:
        raise ValueError('uncertainty_mode must be entrywise or column_gain')
    error_response = S if uncertainty_mode == 'entrywise' else abs(D)
    margin_const = 2 * eps_rad + abs(D) @ (q / 2) + eta * error_response @ (q / 2)
    margin_gain = eta * error_response
    return dict(i=i, k=k, b=b, D=D, S=S, upper=upper, q=q,
                constant=margin_const, gain=margin_gain,error_response=error_response)


def quantize(p, upper, bits):
    upper = np.broadcast_to(np.asarray(upper, float), len(p))
    q = upper / (2**bits - 1)
    return np.clip(np.rint(np.asarray(p) / q) * q, 0, upper)


def pair_certificate(phi, H, p, pmax_mW, bits=8, eta=0.02, eps_rad=0.002, uncertainty_mode="entrywise"):
    terms = uncertainty_terms(phi, H, pmax_mW, bits, eta, eps_rad, uncertainty_mode)
    nominal = terms['b'] + terms['D'] @ p
    bound_by_pair = abs(nominal) + terms['gain'] @ p + terms['constant']
    j = int(np.argmax(bound_by_pair))
    return dict(bound_rad=float(bound_by_pair[j]),
                nominal_pair_rad=float(np.max(abs(nominal))),
                worst_pair=[int(terms['i'][j]), int(terms['k'][j])],
                sensing_margin_rad=float(2 * eps_rad),
                gain_margin_rad=float((terms['gain'] @ p)[j]),
                dac_margin_rad=float((abs(terms['D']) @ (terms['q']/2))[j]),
                gain_dac_cross_margin_rad=float((eta*terms['error_response'] @ (terms['q']/2))[j]))


def pair_range(phi):
    return float(np.max(phi) - np.min(phi))


def mzi_penalty_dB(phase_difference_rad):
    """Worst balanced, lossless MZI excess penalty for |delta|<=bound<=pi.

    The transfer cos(delta/2)**2 is normalized to the constructive port.
    Larger bounds include a dark port and have no finite worst-case bound.
    This is not waveguide propagation loss or total package insertion loss.
    """
    b = abs(float(phase_difference_rad))
    if b >= np.pi:
        return float('inf')
    return float(-20 * np.log10(np.cos(b / 2)))


def optimize(phi, H, pmax_mW=20., total_budget_mW=100., *,
             robust=False, bits=8, eta=0.02, eps_rad=0.002,
             target_rad=None, secondary_tolerance_rad=1e-8, uncertainty_mode='entrywise'):
    """Minimax pair phase (then minimum power), or minimum power for target.

    'robust=False' optimizes continuous nominal response only. Its rounded
    commands are still evaluated with the same uncertainty certificate outside
    this solver. 'robust=True' includes sensor, calibration, DAC and cross-term
    margins directly in the LP. In neither mode is the rounded discrete policy
    asserted to solve a mixed-integer global optimum.
    """
    H = np.asarray(H, float); phi = np.asarray(phi, float)
    t = uncertainty_terms(phi, H, pmax_mW, bits, eta, eps_rad, uncertainty_mode)
    m = H.shape[1]
    gain = t['gain'] if robust else np.zeros_like(t['gain'])
    const = t['constant'] if robust else np.zeros_like(t['constant'])
    power_rows = np.vstack([t['D'] + gain, -t['D'] + gain])
    phase_rhs = np.r_[-t['b'] - const, t['b'] - const]
    bounds = [(0., float(u)) for u in t['upper']]
    A_power = np.ones((1,m)); B_power = np.array([float(total_budget_mW)])
    opts = {'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9}
    if target_rad is not None:
        A = np.vstack([power_rows, A_power])
        b = np.r_[phase_rhs + float(target_rad), B_power]
        solution = linprog(np.ones(m), A_ub=A, b_ub=b, bounds=bounds,
                           method='highs', options=opts)
        if not solution.success:
            return dict(feasible=False, status=int(solution.status), message=solution.message)
        p = np.clip(solution.x, 0, t['upper'])
        z_star = float(target_rad)
    else:
        A = np.vstack([np.column_stack([power_rows, -np.ones(len(power_rows))]),
                       np.r_[np.ones(m), 0.][None,:]])
        b = np.r_[phase_rhs, B_power]
        solution = linprog(np.r_[np.zeros(m),1.], A_ub=A, b_ub=b,
                           bounds=bounds+[(0.,None)], method='highs',options=opts)
        if not solution.success:
            return dict(feasible=False,status=int(solution.status),message=solution.message)
        z_star = float(solution.x[-1])
        # Lexicographic second objective avoids gratuitous common-mode heating.
        A2 = np.vstack([power_rows, A_power]); b2 = np.r_[phase_rhs+z_star+secondary_tolerance_rad,B_power]
        secondary = linprog(np.ones(m), A_ub=A2, b_ub=b2, bounds=bounds,
                            method='highs', options=opts)
        if not secondary.success:
            raise RuntimeError('Lexicographic minimum-power LP failed: '+secondary.message)
        p = np.clip(secondary.x,0,t['upper'])
        A,b = A2,b2
    constrained = abs(t['b']+t['D']@p) + gain@p + const
    constraint_error = float(max(0., np.max(constrained)-(float(target_rad) if target_rad is not None else z_star+secondary_tolerance_rad),
                                 np.sum(p)-total_budget_mW))
    pq = quantize(p,t['upper'],bits)
    cert = pair_certificate(phi,H,p,t['upper'],bits,eta,eps_rad,uncertainty_mode)
    posterior = float(np.max(abs(t['b']+t['D']@pq) + t['gain']@pq + 2*eps_rad))
    if posterior > cert['bound_rad'] + 1e-8:
        raise AssertionError('Posterior interval bound exceeds pre-round certificate.')
    return dict(feasible=True, p_mW=p, p_quantized_mW=pq, optimal_bound_rad=z_star,
                objective_achieved_rad=float(np.max(constrained)),
                continuous_sum_mW=float(p.sum()), quantized_sum_mW=float(pq.sum()),
                # A rounded sum may exceed continuous budget by at most sum(q/2).
                # Designs needing an exact physical total budget reserve that
                # margin before solving; see run_analysis below.
                nominal_continuous_pair_rad=pair_range(phi+H@p),
                nominal_quantized_pair_rad=pair_range(phi+H@pq),
                certificate=cert,posterior_certificate_rad=posterior, LP_constraint_violation_rad=constraint_error,
                quantization_error_mW=float(np.max(abs(pq-p))))


def write_csv(path, rows):
    if not rows:
        return
    columns=[]
    for row in rows:
        for key in row:
            if key not in columns: columns.append(key)
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def verify():
    """Independent analytic examples plus adversarial interval checks."""
    checks=[]
    # Known 2-lane solution: heat lower phase by 1 rad at 0.1 rad/mW.
    phi=np.array([1.,0.]);H=.1*np.eye(2)
    o=optimize(phi,H,20.,30.,bits=12,eta=0.,eps_rad=0.)
    checks.append(('analytic_two_lane_power',abs(o['p_mW'][0])+abs(o['p_mW'][1]-10.)<2e-6))
    o2=optimize(phi,H,20.,5.,bits=12,eta=0.,eps_rad=0.)
    checks.append(('analytic_budget_residual',abs(o2['optimal_bound_rad']-.5)<1e-10))
    # Equal rows: heaters cannot affect differential phase, independently of power.
    oo=optimize(phi,np.full((2,2),.1),20.,30.,bits=12,eta=0.,eps_rad=0.)
    checks.append(('commonmode_heating_null',abs(oo['optimal_bound_rad']-1.)<1e-10 and oo['continuous_sum_mW']<1e-8))
    # Scalar reference shift must never alter p or the optimum.
    o3=optimize(phi+900.,H,20.,30.,bits=12,eta=0.,eps_rad=0.)
    checks.append(('commonphase_shift_invariance',np.max(abs(o3['p_mW']-o['p_mW']))<1e-8))
    # Direct pair LP equals nominal floating-target radius*2.
    rng=np.random.default_rng(260925);max_equivalence_error=0.;max_violation=-np.inf;trials=0
    for repeat in range(60):
        n=2+repeat%4;phi=rng.normal(0,.2,n);H=rng.uniform(.005,.03,(n,n))+.12*np.eye(n)
        u=rng.uniform(4,15);budget=.65*n*u;eta=.03;eps=.003;bits=6
        o=optimize(phi,H,u,budget,robust=False,bits=bits,eta=eta,eps_rad=eps)
        # Variables [p(n), common target, radius]. Independent epigraph.
        Af=np.vstack([np.column_stack([H,-np.ones(n),-np.ones(n)]),
                      np.column_stack([-H,np.ones(n),-np.ones(n)]),
                      np.r_[np.ones(n),0.,0.][None,:]])
        bf=np.r_[-phi,phi,budget]
        f=linprog(np.r_[np.zeros(n+1),1.],A_ub=Af,b_ub=bf,
                  bounds=[(0.,u)]*n+[(None,None),(0.,None)],method='highs')
        max_equivalence_error=max(max_equivalence_error,abs(2*f.x[-1]-o['optimal_bound_rad']))
        ro=optimize(phi,H,u,budget,robust=True,bits=bits,eta=eta,eps_rad=eps)
        cert=ro['certificate']['bound_rad'];pq=ro['p_quantized_mW']
        # Random draws AND each pair's exact adversarial gain/sensor corner for
        # the actual quantized command. Corners are stronger than Monte Carlo.
        i,k,b,D,S=pair_operators(phi,H)
        for pair in range(len(i)):
            for s in [-1.,1.]:
                e=np.zeros(n);e[i[pair]]=s*eps;e[k[pair]]=-s*eps
                dH=np.zeros_like(H);dH[i[pair]]=s*eta*abs(H[i[pair]])
                dH[k[pair]]=-s*eta*abs(H[k[pair]])
                actual=pair_range(phi+e+(H+dH)@pq)
                max_violation=max(max_violation,actual-cert);trials+=1
        for draw in range(20):
            e=rng.uniform(-eps,eps,n);dH=eta*abs(H)*rng.uniform(-1,1,H.shape)
            max_violation=max(max_violation,pair_range(phi+e+(H+dH)@pq)-cert);trials+=1
    checks.append(('floating_target_equivalence',max_equivalence_error<1e-8))
    checks.append(('quantized_interval_certificate',max_violation<1e-9))
    result=dict(checks=[dict(name=n,passed=bool(p)) for n,p in checks],
                max_floating_target_equivalence_error_rad=max_equivalence_error,
                worst_certificate_violation_rad=float(max_violation),adversarial_and_random_cases=trials,
                all_passed=all(p for _,p in checks))
    if not result['all_passed']:
        raise AssertionError(json.dumps(result,indent=2))
    return result


def input_cases(path):
    """Read the documented physical-response schema with pickle disabled."""
    data=np.load(path,allow_pickle=False)
    phi=np.asarray(data['phi_rad']);H=np.asarray(data['H_rad_per_mW'])
    if phi.ndim==1:phi=phi[None,:];H=H[None,:,:]
    count=len(phi)
    def at(key,index,default):
        if key not in data:return default
        a=np.asarray(data[key]);return a.item() if a.ndim==0 else a[index].item()
    return [dict(case_id=str(at('case_id',j,f'case_{j:03d}')),
                 design=str(at('design',j,f'case_{j:03d}')),
                 Rb_m2K_W=float(at('Rb_m2K_W',j,0.)),
                 nx=int(at('nx',j,120)),k_AlN_W_mK=float(at('k_AlN_W_mK',j,170.)),kind=str(at('kind',j,'interface')),phi=phi[j],H=H[j]) for j in range(count)]


def run_analysis(inputs, out, main_nx=120):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    verification=verify();(out/'control_validation.json').write_text(json.dumps(verification,indent=2))
    cases=input_cases(inputs);rows=[];powers=[];errors=[]
    seed=260925;rng=np.random.default_rng(seed)
    # Identical perturbation shapes across materials: paired comparisons.
    draws=128;n=cases[0]['H'].shape[0];m=cases[0]['H'].shape[1]
    sensor_shapes=rng.uniform(-1,1,(draws,n));gain_shapes=rng.uniform(-1,1,(draws,n,m))
    np.savez_compressed(out/'control_perturbations.npz',sensor_shapes=sensor_shapes,gain_shapes=gain_shapes,seed=seed)
    for case in cases:
        if case['nx'] != main_nx: continue
        phi,H=case['phi'],case['H'];meta={k:case[k] for k in ['case_id','design','Rb_m2K_W','nx','k_AlN_W_mK','kind']}
        # All budget constraints include a reserve to guarantee the rounded sum.
        for pmax in [10.,20.]:
          for bits in [6,8,10]:
            reserve=H.shape[1]*pmax/(2**bits-1)/2
            for eta in [0.,.02,.05]:
              eps=.002
              for budget in [25.,50.,100.]:
                for robust in [False,True]:
                  solution=optimize(phi,H,pmax,budget-reserve,robust=robust,bits=bits,eta=eta,eps_rad=eps)
                  if not solution['feasible']:raise RuntimeError('Minimax always admits p=0.')
                  row=dict(**meta,mode='robust' if robust else 'nominal',objective='minimax',feasible=True,
                    per_heater_max_mW=pmax,total_budget_mW=budget,rounding_power_reserve_mW=reserve,
                    DAC_bits=bits,relative_H_bound=eta,sensor_bound_rad=eps,
                    passive_pair_rad=pair_range(phi),**{k:solution[k] for k in ['continuous_sum_mW','quantized_sum_mW',
                    'optimal_bound_rad','objective_achieved_rad','nominal_continuous_pair_rad','nominal_quantized_pair_rad','LP_constraint_violation_rad']},
                    certificate_rad=solution['certificate']['bound_rad'],posterior_certificate_rad=solution['posterior_certificate_rad'],
                    ideal_MZI_penalty_bound_dB=mzi_penalty_dB(solution['posterior_certificate_rad']))
                  pq=solution['p_quantized_mW'];actual=phi[None,:]+eps*sensor_shapes+np.einsum('ij,bj->bi',H,np.broadcast_to(pq,(draws,m)))+eta*np.einsum('ij,bij,j->bi',abs(H),gain_shapes,pq)
                  spread=np.ptp(actual,axis=1)
                  row.update(perturbation_draws=draws,perturbed_p95_pair_rad=float(np.quantile(spread,.95)),
                      perturbed_max_pair_rad=float(spread.max()),draw_count_below_0p05_rad=int(np.sum(spread<=.05)),
                      draw_count_below_0p10_rad=int(np.sum(spread<=.10)),
                      rounded_budget_violation_mW=max(0.,float(pq.sum())-budget))
                  rows.append(row)
                  for j,(p,pqj) in enumerate(zip(solution['p_mW'],pq)):
                    powers.append(dict(**meta,mode=row['mode'],objective='minimax',per_heater_max_mW=pmax,
                         total_budget_mW=budget,DAC_bits=bits,relative_H_bound=eta,heater=j,
                         continuous_power_mW=float(p),quantized_power_mW=float(pqj)))
                  if float(spread.max())>solution['certificate']['bound_rad']+1e-8:
                    raise AssertionError('Interval certificate violated')
              # Target-constrained minimum power uses a generous 9*Pmax ceiling;
              # specific practical total budgets are tested via minimax above.
              for target in [.05,.10]:
                for robust in [False,True]:
                  budget=H.shape[1]*pmax
                  solution=optimize(phi,H,pmax,budget-reserve,robust=robust,bits=bits,eta=eta,eps_rad=eps,target_rad=target)
                  row=dict(**meta,mode='robust' if robust else 'nominal',objective='minimum_power',
                    per_heater_max_mW=pmax,total_budget_mW=budget,rounding_power_reserve_mW=reserve,
                    DAC_bits=bits,relative_H_bound=eta,sensor_bound_rad=eps,target_pair_rad=target,
                    feasible=solution['feasible'],passive_pair_rad=pair_range(phi))
                  if solution['feasible']:
                    row.update({k:solution[k] for k in ['continuous_sum_mW','quantized_sum_mW','nominal_continuous_pair_rad','nominal_quantized_pair_rad','LP_constraint_violation_rad']})
                    row.update(certificate_rad=solution['certificate']['bound_rad'],posterior_certificate_rad=solution['posterior_certificate_rad'],ideal_MZI_penalty_bound_dB=mzi_penalty_dB(solution['posterior_certificate_rad']))
                    for j,(p,pqj) in enumerate(zip(solution['p_mW'],solution['p_quantized_mW'])):
                      powers.append(dict(**meta,mode=row['mode'],objective='minimum_power',per_heater_max_mW=pmax,
                           total_budget_mW=budget,DAC_bits=bits,relative_H_bound=eta,target_pair_rad=target,heater=j,
                           continuous_power_mW=float(p),quantized_power_mW=float(pqj)))
                  rows.append(row)
    write_csv(out/'control_results.csv',rows);write_csv(out/'control_commands.csv',powers)
    key_rows=[r for r in rows if r['objective']=='minimax' and r['DAC_bits']==8 and r['per_heater_max_mW']==20 and r['relative_H_bound']==.02 and r['total_budget_mW']==100 and r['kind']=='interface' and (r['design'] in ['Cu','graded_W'] or r['Rb_m2K_W']==1e-8)]
    write_csv(out/'control_key_results.csv',key_rows)
    summary=dict(input_sha256=hashlib.sha256(Path(inputs).read_bytes()).hexdigest(),number_of_cases=len({r['case_id'] for r in rows}),
       optimization_rows=len(rows),command_rows=len(powers),main_nx=main_nx,perturbation_draws_per_minimax=draws,
       maximum_LP_constraint_violation_rad=max(r.get('LP_constraint_violation_rad',0) for r in rows),
       maximum_rounded_budget_violation_mW=max(r.get('rounded_budget_violation_mW',0) for r in rows),
       validation=verification,
       assumptions=['Linear physical H has units rad/mW and is recomputed for each material/interface case.',
         'Each optical lane is an assumed 1 mm uniform path; no optical experiments are implied.',
         'No modulo branch changes; equalization is on the unwrapped phase-locking branch.',
         'Heater powers are rounded to endpoint-inclusive finite-bit DAC levels; continuous LP is not an integer optimum.',
         'Independent entrywise H intervals and per-lane sensing intervals are explicit assumed uncertainty sets.',
         'Reported MZI penalty excludes propagation, coupling, reflection and absorption loss.',
         'Adversarial/Monte Carlo tests verify the bound inside the assumed model, not fabrication yield.'])
    (out/'control_summary.json').write_text(json.dumps(summary,indent=2))
    plot_results(rows,out)
    summary['uncertainty_structure_sensitivity']=run_sensitivity(inputs,out)
    (out/'control_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def run_sensitivity(inputs,out):
    """Separate uncertainty-structure and minimum-power sensitivity table.

    column_gain means H_true = H @ diag(1+delta), |delta_j|<=eta.
    Its exact pair calibration term is eta*abs(H_i-H_k)@p. It preserves
    common-mode heater gains, unlike the broader independent-entrywise set.
    Neither structure is selected by experimental calibration here.
    """
    cases=[c for c in input_cases(inputs) if c['nx']==120 and c['kind']=='interface' and
       c['k_AlN_W_mK']==170. and (c['Rb_m2K_W']==1e-8 or c['design'] in ['Cu','graded_W'])]
    rows=[];commands=[];max_violation=-np.inf;corners=0
    for c in cases:
      phi,H=c['phi'],c['H'];meta={k:c[k] for k in ['case_id','design','Rb_m2K_W','nx','k_AlN_W_mK','kind']}
      for structure in ['entrywise','column_gain']:
        for pmax in [20.,50.,100.]:
          bits=8;eta=.02;eps=.002;reserve=H.shape[1]*pmax/(2**bits-1)/2
          tasks=[('minimax',100.,None)]+[('minimum_power',H.shape[1]*pmax,t) for t in [.05,.10,.25,.50]]
          for objective,budget,target in tasks:
            o=optimize(phi,H,pmax,budget-reserve,robust=True,bits=bits,eta=eta,eps_rad=eps,
                 target_rad=target,uncertainty_mode=structure)
            row=dict(**meta,uncertainty_structure=structure,objective=objective,
                  per_heater_max_mW=pmax,total_budget_mW=budget,target_pair_rad=target,
                  DAC_bits=bits,relative_H_bound=eta,sensor_bound_rad=eps,feasible=o['feasible'])
            if o['feasible']:
              row.update({k:o[k] for k in ['continuous_sum_mW','quantized_sum_mW','nominal_quantized_pair_rad','posterior_certificate_rad']})
              row['certificate_rad']=o['certificate']['bound_rad']
              i,k,b,D,S=pair_operators(phi,H);pq=o['p_quantized_mW'];post=o['posterior_certificate_rad']
              # Exact adversarial pair corners for each uncertainty structure.
              for pair in range(len(i)):
                for sign in [-1.,1.]:
                  e=np.zeros(len(phi));e[i[pair]]=sign*eps;e[k[pair]]=-sign*eps
                  if structure=='column_gain':
                    delta=sign*eta*np.sign(D[pair]);Ht=H*(1+delta[None,:])
                  else:
                    Ht=H.copy();Ht[i[pair]]+=sign*eta*abs(H[i[pair]]);Ht[k[pair]]-=sign*eta*abs(H[k[pair]])
                  violation=pair_range(phi+e+Ht@pq)-post
                  max_violation=max(max_violation,violation);corners+=1
              for j,(p,pqj) in enumerate(zip(o['p_mW'],pq)):
                commands.append(dict(**meta,uncertainty_structure=structure,objective=objective,
                   per_heater_max_mW=pmax,total_budget_mW=budget,target_pair_rad=target,
                   heater=j,continuous_power_mW=float(p),quantized_power_mW=float(pqj)))
            rows.append(row)
    if max_violation>1e-8:raise AssertionError('Sensitivity exact interval certificate violated')
    write_csv(Path(out)/'control_sensitivity.csv',rows)
    write_csv(Path(out)/'control_sensitivity_commands.csv',commands)
    checks=dict(rows=len(rows),adversarial_corner_cases=corners,
                maximum_exact_interval_bound_violation_rad=float(max_violation),
                all_passed=bool(max_violation<=1e-8))
    (Path(out)/'control_sensitivity_validation.json').write_text(json.dumps(checks,indent=2))
    return checks


def plot_results(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    # Main: 8-bit 20 mW heaters, eta 2%, all cases incl finite interfaces.
    selected=[r for r in rows if r['DAC_bits']==8 and r['per_heater_max_mW']==20 and r['relative_H_bound']==.02 and r['kind']=='interface' and r['k_AlN_W_mK']==170. and (r['Rb_m2K_W']==1e-8 or r['design'] in ['Cu','graded_W'])]
    ids=list(dict.fromkeys(r['case_id'] for r in selected));colors=plt.cm.viridis(np.linspace(.1,.9,max(len(ids),1)))
    fig,axes=plt.subplots(1,3,figsize=(11,4.2))
    for cid,col in zip(ids,colors):
        rr=[r for r in selected if r['case_id']==cid and r['objective']=='minimax' and r['mode']=='robust']
        rr=sorted(rr,key=lambda r:r['total_budget_mW'])
        if not rr:continue
        label={'Cu':'Cu','graded_W':'W','AlN':'AlN','graded_W_AlN':'W + AlN'}[rr[0]['design']]
        axes[0].plot([r['total_budget_mW'] for r in rr],[r['certificate_rad'] for r in rr],'-o',label=label,color=col,ms=3)
        axes[1].plot([r['total_budget_mW'] for r in rr],[r['quantized_sum_mW'] for r in rr],'-o',color=col,ms=3)
        nominal=[r for r in selected if r['case_id']==cid and r['objective']=='minimax' and r['mode']=='nominal' and r['total_budget_mW']==100]
        robust=[r for r in rr if r['total_budget_mW']==100]
        if nominal and robust:axes[2].plot([0,1],[nominal[0]['certificate_rad'],robust[0]['certificate_rad']],'-o',color=col,ms=3)
    axes[0].set(xlabel='Available total heater power (mW)',ylabel='Pre-round certificate $C_{pre}$ (rad)')
    axes[1].set(xlabel='Available total heater power (mW)',ylabel='Quantized heater power used (mW)')
    axes[2].set(xticks=[0,1],xticklabels=['Nominal LP','Robust LP'],ylabel='$C_{pre}$ at 100 mW budget (rad)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.04),ncol=4,frameon=False,fontsize=11)
    for j,ax in enumerate(axes):ax.set_title('('+chr(97+j)+')',loc='left')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'control_tradeoff.{ext}',dpi=220)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parents[1]/'results/control_inputs.npz')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    p.add_argument('--main-nx',type=int,default=120);p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    result=verify() if a.verify_only else run_analysis(a.inputs,a.out,a.main_nx)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
