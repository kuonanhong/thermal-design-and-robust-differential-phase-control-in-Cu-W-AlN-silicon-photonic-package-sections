#!/usr/bin/env python3
"""Source-resolved static workload robustness for photonic package phase control.

Adds two independently normalized EIC/PIC background-source columns to the
existing linear package model without modifying its thermal/mechanical core.
The declared workload box is a deterministic sensitivity envelope, not a
measured workload distribution. One common fixed heater command is compared
with a command tuned at nominal load and separate, clairvoyant vertex commands.
No transient feedback or operating-policy implementation is claimed.

python code/workload_robust.py
Outputs: results/workload_robust_*.{npz,csv,json,pdf,png}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog, brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core import geometry, ThermalOperator, ElasticOperator, heater_sources, sample_fields
from physical import optical_from_fields, FIELD_KEYS
from control_analysis import optimize, quantize, mzi_penalty_dB

ROOT = Path(__file__).resolve().parents[1]
LOWER = np.array([200., 20.])
UPPER = np.array([300., 30.])
NOMINAL = np.array([250., 25.])
VERTICES = np.array(list(itertools.product(*zip(LOWER, UPPER))))
PMAX = 20.
BUDGET = 100.
BITS = 8
ETA = .02
EPS = .002
Q = PMAX / (2**BITS - 1)
CONTINUOUS_BUDGET = BUDGET - 9*Q/2
DESIGNS = [
    dict(case_id='Cu', design='Cu', centers=[320., 880.], label='Cu'),
    dict(case_id='W', design='graded_W', centers=[320., 880.], label='W'),
    dict(case_id='AlN_original', design='AlN', centers=[320., 880.], label='AlN 320/880'),
    dict(case_id='AlN_passive', design='AlN', centers=[320., 600.], label='AlN 320/600'),
    dict(case_id='WAlN_control', design='graded_W_AlN', centers=[180., 880.], label='W+AlN 180/880'),
]


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pairs(G, H):
    i, k = np.triu_indices(H.shape[0], 1)
    return i, k, G[i]-G[k], H[i]-H[k], abs(H[i])+abs(H[k])


def fixed_certificate(G, H, p, loads=VERTICES):
    """Exact load-box x independent calibration-box certificate for fixed p.

    Linear source dependence and fixed H imply that checking the four source
    vertices suffices for every interior load. Calibration errors remain the
    same explicit mathematical error set used in the principal analysis.
    """
    i, k, A, D, S = pairs(G, H)
    b = np.asarray(loads) @ A.T
    nominal = b + (D @ p)[None, :]
    margins = 2*EPS + ETA*S@p
    pair_bounds = abs(nominal) + margins[None, :]
    v, pair = np.unravel_index(np.argmax(pair_bounds), pair_bounds.shape)
    return dict(certificate_rad=float(pair_bounds[v, pair]),
                corner_certificates_rad=np.max(pair_bounds, axis=1),
                worst_vertex=int(v), worst_pair=[int(i[pair]), int(k[pair])],
                nominal_spreads_rad=np.max(abs(nominal), axis=1))


def source_box_support_certificate(G, H, p, lower=LOWER, upper=UPPER):
    """Equivalent exact support-function form, with no vertex enumeration."""
    _, _, A, D, S = pairs(G,H)
    center=(np.asarray(lower)+np.asarray(upper))/2
    radius=(np.asarray(upper)-np.asarray(lower))/2
    return float(np.max(abs(A@center+D@p)+abs(A)@radius+2*EPS+ETA*S@p))


def passive_target_tolerance(G, target=.25):
    """Largest common fractional source-box halfwidth meeting target at p=0.

    This exact expression applies to nonnegative load boxes with halfwidth
    delta<=1. Negative raw values mean the nominal case already fails.
    """
    i,k=np.triu_indices(G.shape[0],1)
    A=G[i]-G[k]; b=A@NOMINAL; slope=abs(A)@NOMINAL
    margin=target-2*EPS-abs(b)
    ratios=np.divide(margin,slope,out=np.full_like(margin,np.inf),where=slope>1e-15)
    j=int(np.argmin(ratios)); raw=float(ratios[j])
    feasible=bool(np.all(margin>=0))
    return dict(target_rad=target,nominal_passes=feasible,
                fractional_halfwidth=(min(raw,1.) if feasible else 0.),
                raw_algebraic_halfwidth=raw,limiting_pair_i=int(i[j]),limiting_pair_k=int(k[j]))


def common_command(G, H):
    """Minimax robust LP for one command shared by every source-box vertex."""
    i, k, A, D, S = pairs(G, H)
    bs = VERTICES @ A.T
    gain = ETA*S
    const = 2*EPS + np.sum(abs(D)*Q/2, axis=1) + np.sum(gain*Q/2, axis=1)
    single_rows = np.vstack([D+gain, -D+gain])
    phase_rows = np.tile(single_rows, (len(VERTICES), 1))
    phase_rhs = np.concatenate([np.r_[-b-const, b-const] for b in bs])
    n = H.shape[1]
    A_ub = np.vstack([np.column_stack([phase_rows, -np.ones(len(phase_rows))]),
                      np.r_[np.ones(n), 0.][None, :]])
    b_ub = np.r_[phase_rhs, CONTINUOUS_BUDGET]
    bounds = [(0., PMAX)]*n + [(0., None)]
    opts = {'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9}
    result = linprog(np.r_[np.zeros(n), 1.], A_ub=A_ub, b_ub=b_ub,
                     bounds=bounds, method='highs', options=opts)
    if not result.success:
        raise RuntimeError(result.message)
    optimum = float(result.x[-1])
    secondary = linprog(np.ones(n),
        A_ub=np.vstack([phase_rows, np.ones((1,n))]),
        b_ub=np.r_[phase_rhs+optimum+1e-8, CONTINUOUS_BUDGET],
        bounds=bounds[:-1], method='highs', options=opts)
    if not secondary.success:
        raise RuntimeError(secondary.message)
    p = np.clip(secondary.x, 0, PMAX)
    pq = quantize(p, PMAX, BITS)
    post = fixed_certificate(G, H, pq)
    if post['certificate_rad'] > optimum + 1.1e-8:
        raise AssertionError('Post-round certificate exceeds robust pre-round optimum.')
    return dict(p_mW=p, p_quantized_mW=pq, pre_rad=optimum, **post)


def continuous_relaxation_lower_bound(G,H):
    """Exact continuous minimax; a lower bound for every quantized command.

    DAC margins and the DAC budget reserve are removed. The feasible set thus
    contains every physically admissible quantized command. An optimum above a
    target proves that no quantized fixed command can certify that target under
    the declared source/calibration box, independent of the sufficient LP used
    to construct a command.
    """
    _,_,A,D,S=pairs(G,H)
    bs=VERTICES@A.T
    phase_rows=np.tile(np.vstack([D+ETA*S,-D+ETA*S]),(4,1))
    rhs=np.concatenate([np.r_[-b-2*EPS,b-2*EPS] for b in bs])
    A_ub=np.vstack([np.column_stack([phase_rows,-np.ones(len(phase_rows))]),np.r_[np.ones(9),0.][None,:]])
    result=linprog(np.r_[np.zeros(9),1.],A_ub=A_ub,b_ub=np.r_[rhs,BUDGET],
        bounds=[(0.,PMAX)]*9+[(0.,None)],method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:raise RuntimeError(result.message)
    return float(result.fun)


def combiner_efficiency(phase):
    """Normalized ideal equal-amplitude N-input coherent-combiner output."""
    return float(abs(np.mean(np.exp(1j*np.asarray(phase))))**2)


def combiner_lower_bound(bound):
    # Common-phase rotation places every phasor within an arc of width C.
    # The average real part is >= cos(C/2), hence eta >= cos(C/2)^2.
    return float(np.cos(bound/2)**2) if bound < np.pi else 0.


def reference_checks(case, G, H, out):
    """Confirm nominal basis reassembly agrees with existing principal data."""
    if case['case_id'] in ['Cu', 'W', 'AlN_original']:
        ref = np.load(out/'control_inputs.npz', allow_pickle=False)
        mask = (ref['design']==case['design']) & (ref['nx']==120) & (ref['k_AlN_W_mK']==170.)
        mask &= np.isclose(ref['Rb_m2K_W'], 1e-8 if 'AlN' in case['design'] else 0., atol=1e-14, rtol=0)
    else:
        ref = np.load(out/'placement_inputs.npz', allow_pickle=False)
        mask = (ref['design']==case['design']) & (ref['left_guard_um']==case['centers'][0]) & (ref['right_guard_um']==case['centers'][1])
    selected = np.flatnonzero(mask)
    if len(selected) != 1:
        raise ValueError((case['case_id'], selected))
    j = selected[0]
    return dict(nominal_phase_reference_max_error_rad=float(np.max(abs(G@NOMINAL-ref['phi_rad'][j]))),
                heater_reference_max_error_rad_per_mW=float(np.max(abs(H-ref['H_rad_per_mW'][j]))))


def run(out, make_plots=True):
    out.mkdir(parents=True, exist_ok=True)
    coefficients = json.loads((ROOT/'code/modal_coefficients.json').read_text())
    start = time.time()
    basis_payload = {k:[] for k in ['G_rad_per_mW', 'H_rad_per_mW', 'nominal_phase_rad',
        'source_temperature_K_per_mW', 'heater_temperature_K_per_mW',
        'commands_nominal_mW', 'commands_common_mW', 'commands_adaptive_mW',
        'continuous_commands_nominal_mW', 'continuous_commands_common_mW', 'continuous_commands_adaptive_mW',
        'nominal_pre_rad', 'common_pre_rad', 'adaptive_pre_rad',
        'continuous_relaxation_lower_bound_rad',
        'fixed_nominal_corner_certificates_rad', 'common_corner_certificates_rad', 'adaptive_corner_certificates_rad']}
    field_payload = {k:[] for k in FIELD_KEYS}
    rows, detail_rows, source_rows, reference_rows, ratio_rows, tolerance_rows = [], [], [], [], [], []
    max_energy = max_equilibrium = max_normalization = max_source_superposition = 0.
    max_nominal_source_reconstruction = max_exact_control_corner_violation = max_interior_violation = 0.
    max_support_identity_error = 0.
    rng = np.random.default_rng(20260926)
    for case in DESIGNS:
        g = geometry(120, 60, package='stacked', design=case['design'],
                     scales={'guard_centers_um':case['centers']})
        source = np.stack([g[key]/(g[key].sum()*g['dx']*g['dy']) for key in ['eic','pic']])
        heaters = heater_sources(g)
        # Two background bases, nine heaters, and an independently assembled
        # nominal background are solved through the same cached operators.
        loads = np.concatenate([source, heaters, g['q'][None]])
        op = ThermalOperator(g, 1e-8 if 'AlN' in case['design'] else 0.)
        mech = ElasticOperator(g)
        temp = op.solve(loads)
        sampled = []
        max_normalization = max(max_normalization, float(np.max(abs(loads[:11].sum(axis=(1,2))*g['dx']*g['dy']-1))))
        for j, t in enumerate(temp):
            mechanical = mech.solve(t)
            sampled.append(sample_fields(g, t, mechanical))
            max_energy = max(max_energy, op.energy(loads[j],t)['energy_relative_error'])
            max_equilibrium = max(max_equilibrium, mechanical['equilibrium_relative_residual'])
        all_fields = {k:np.stack([v[k] for v in sampled]) for k in FIELD_KEYS}
        all_phase, *_ = optical_from_fields(all_fields, coefficients)
        G, H = all_phase[:2].T, all_phase[2:11].T
        max_nominal_source_reconstruction = max(max_nominal_source_reconstruction,
            float(np.max(abs(G@NOMINAL-all_phase[-1]))))
        max_source_superposition = max(max_source_superposition,
            float(np.max(abs(np.einsum('s,syx->yx',NOMINAL,temp[:2])-temp[-1])) / np.max(abs(temp[-1]))))
        reference = reference_checks(case,G,H,out)
        reference_rows.append(dict(case_id=case['case_id'], **reference))
        nominal = optimize(G@NOMINAL, H, PMAX, CONTINUOUS_BUDGET,
                           robust=True, bits=BITS, eta=ETA, eps_rad=EPS)
        common = common_command(G,H)
        relaxation = continuous_relaxation_lower_bound(G,H)
        adaptive = [optimize(G@load,H,PMAX,CONTINUOUS_BUDGET,
                    robust=True,bits=BITS,eta=ETA,eps_rad=EPS) for load in VERTICES]
        npost = fixed_certificate(G,H,nominal['p_quantized_mW'])
        apost = np.array([fixed_certificate(G,H,a['p_quantized_mW'],[load])['certificate_rad']
                         for load,a in zip(VERTICES,adaptive)])
        pure = fixed_certificate(G,H,np.zeros(9))
        tolerance_rows.append(dict(case_id=case['case_id'],**passive_target_tolerance(G)))
        # Existing controller is verified independently elsewhere. These checks
        # explicitly reconstruct attainable worst uncertainty corners for the
        # present two-source data, including all sampled interior workloads.
        for name,p in [('nominal',nominal['p_quantized_mW']),('common',common['p_quantized_mW'])]:
            cert = fixed_certificate(G,H,p)['certificate_rad']
            max_support_identity_error=max(max_support_identity_error,
                abs(cert-source_box_support_certificate(G,H,p)))
            i,k,A,D,S = pairs(G,H)
            for v in VERTICES:
                for pair in range(len(i)):
                    for sign in [-1.,1.]:
                        e = np.zeros(9); e[i[pair]]=sign*EPS; e[k[pair]]=-sign*EPS
                        dh = np.zeros_like(H)
                        dh[i[pair]]=sign*ETA*abs(H[i[pair]])
                        dh[k[pair]]=-sign*ETA*abs(H[k[pair]])
                        actual = np.ptp(G@v + e + (H+dh)@p)
                        max_exact_control_corner_violation = max(max_exact_control_corner_violation,float(actual-cert))
            interior = rng.uniform(LOWER,UPPER,(1000,2))
            interior_bounds = fixed_certificate(G,H,p,interior)['corner_certificates_rad']
            max_interior_violation = max(max_interior_violation,float(np.max(interior_bounds)-cert))
        policies = [('passive',np.zeros(9),pure),
                    ('nominal_fixed',nominal['p_quantized_mW'],npost),
                    ('common_robust_fixed',common['p_quantized_mW'],common)]
        for name,p,certification in policies:
            efficiencies = []
            for vertex,(load,certificate) in enumerate(zip(VERTICES,certification['corner_certificates_rad'])):
                phase = G@load + H@p
                efficiency = combiner_efficiency(phase)
                efficiencies.append(efficiency)
                detail_rows.append(dict(case_id=case['case_id'],policy=name,vertex=vertex,
                    EIC_mW=load[0],PIC_mW=load[1],sum_power_mW=float(p.sum()),
                    phase_range_rad=float(np.ptp(phase)),certificate_rad=float(certificate),
                    nominal_combiner_efficiency=efficiency,
                    certified_combiner_efficiency_lower_bound=combiner_lower_bound(certificate),
                    certified_phase_only_penalty_dB=mzi_penalty_dB(certificate)))
            rows.append(dict(case_id=case['case_id'],policy=name,
                nominal_load_phase_range_rad=float(np.ptp(G@NOMINAL+H@p)),
                worst_corner_nominal_phase_range_rad=float(max(certification['nominal_spreads_rad'])),
                load_box_certificate_rad=certification['certificate_rad'],
                continuous_relaxation_lower_bound_rad=relaxation,
                pre_round_objective_rad=(nominal['optimal_bound_rad'] if name=='nominal_fixed' else common['pre_rad'] if name=='common_robust_fixed' else ''),
                sum_power_mW=float(p.sum()),worst_vertex=certification['worst_vertex'],
                minimum_nominal_vertex_combiner_efficiency=float(min(efficiencies)),
                certified_combiner_efficiency_lower_bound=combiner_lower_bound(certification['certificate_rad']),
                certified_phase_only_penalty_dB=mzi_penalty_dB(certification['certificate_rad'])))
        # Adaptive means four independently optimized steady commands, not a
        # physical feedback law. It is not labeled as a quantized box guarantee.
        ae = []
        for vertex,(load,control,certificate) in enumerate(zip(VERTICES,adaptive,apost)):
            p = control['p_quantized_mW']; phase = G@load+H@p
            ae.append(combiner_efficiency(phase))
            detail_rows.append(dict(case_id=case['case_id'],policy='adaptive_vertex',vertex=vertex,
                EIC_mW=load[0],PIC_mW=load[1],sum_power_mW=float(p.sum()),
                phase_range_rad=float(np.ptp(phase)),certificate_rad=float(certificate),
                nominal_combiner_efficiency=ae[-1],
                certified_combiner_efficiency_lower_bound=combiner_lower_bound(certificate),
                certified_phase_only_penalty_dB=mzi_penalty_dB(certificate)))
        rows.append(dict(case_id=case['case_id'],policy='adaptive_vertices_only',
                nominal_load_phase_range_rad='',worst_corner_nominal_phase_range_rad=max(v['nominal_quantized_pair_rad'] for v in adaptive),
                load_box_certificate_rad='',continuous_relaxation_lower_bound_rad='',pre_round_objective_rad=float(max(v['optimal_bound_rad'] for v in adaptive)),
                sum_power_mW=float(max(v['quantized_sum_mW'] for v in adaptive)),worst_vertex=int(np.argmax(apost)),
                minimum_nominal_vertex_combiner_efficiency=float(min(ae)),
                certified_combiner_efficiency_lower_bound=float(min(combiner_lower_bound(c) for c in apost)),
                certified_phase_only_penalty_dB=mzi_penalty_dB(max(apost))))
        for j,name in enumerate(['EIC','PIC']):
            source_rows.append(dict(case_id=case['case_id'],source=name,
                mean_phase_sensitivity_rad_per_mW=float(G[:,j].mean()),
                phase_spread_sensitivity_rad_per_mW=float(np.ptp(G[:,j])),
                mean_temperature_sensitivity_K_per_mW=float(all_fields['temperature_K'][j].mean())))
        for eic in np.linspace(100,400,31):
            for pic in np.linspace(10,150,29):
                phase = G@np.array([eic,pic])
                ratio_rows.append(dict(case_id=case['case_id'],EIC_mW=eic,PIC_mW=pic,
                    passive_phase_range_rad=float(np.ptp(phase)),
                    nominal_fixed_certificate_rad=fixed_certificate(G,H,nominal['p_quantized_mW'],[[eic,pic]])['certificate_rad'],
                    common_fixed_certificate_rad=fixed_certificate(G,H,common['p_quantized_mW'],[[eic,pic]])['certificate_rad']))
        data = dict(G_rad_per_mW=G,H_rad_per_mW=H,nominal_phase_rad=G@NOMINAL,
            source_temperature_K_per_mW=all_fields['temperature_K'][:2].T,
            heater_temperature_K_per_mW=all_fields['temperature_K'][2:11].T,
            commands_nominal_mW=nominal['p_quantized_mW'],commands_common_mW=common['p_quantized_mW'],
            commands_adaptive_mW=np.stack([v['p_quantized_mW'] for v in adaptive]),
            continuous_commands_nominal_mW=nominal['p_mW'],continuous_commands_common_mW=common['p_mW'],
            continuous_commands_adaptive_mW=np.stack([v['p_mW'] for v in adaptive]),
            nominal_pre_rad=nominal['optimal_bound_rad'],common_pre_rad=common['pre_rad'],
            continuous_relaxation_lower_bound_rad=relaxation,
            adaptive_pre_rad=np.array([v['optimal_bound_rad'] for v in adaptive]),
            fixed_nominal_corner_certificates_rad=npost['corner_certificates_rad'],
            common_corner_certificates_rad=common['corner_certificates_rad'],adaptive_corner_certificates_rad=apost)
        for key,value in data.items():basis_payload[key].append(value)
        for key in FIELD_KEYS:field_payload[key].append(all_fields[key])
        print(case['case_id'], 'fixed_nominal', npost['certificate_rad'],
              'common',common['certificate_rad'],'adaptive_vertices',max(apost),
              'common_power',common['p_quantized_mW'].sum(),flush=True)
    payload = {key:np.array(value) for key,value in basis_payload.items()}
    payload.update(case_id=np.array([c['case_id'] for c in DESIGNS]),design=np.array([c['design'] for c in DESIGNS]),
                   source_names=np.array(['EIC','PIC']),source_lower_mW=LOWER,source_upper_mW=UPPER,
                   source_nominal_mW=NOMINAL,source_vertices_mW=VERTICES,
                   pmax_mW=PMAX,total_budget_mW=BUDGET,bits=BITS,eta=ETA,eps_rad=EPS)
    np.savez_compressed(out/'workload_robust_inputs.npz', **payload)
    np.savez_compressed(out/'workload_robust_physical_fields.npz',case_id=payload['case_id'],
        load_labels=np.array(['EIC_unit','PIC_unit']+[f'heater_{i}' for i in range(9)]+['nominal_direct']),
        **{k:np.array(v) for k,v in field_payload.items()})
    write_csv(out/'workload_robust_summary.csv',rows)
    write_csv(out/'workload_robust_vertices.csv',detail_rows)
    write_csv(out/'workload_robust_sources.csv',source_rows)
    write_csv(out/'workload_robust_load_map.csv',ratio_rows)
    write_csv(out/'workload_robust_passive_tolerance.csv',tolerance_rows)
    Gcu=payload['G_rad_per_mW'][0]; Gpassive=payload['G_rad_per_mW'][3]
    cross_ratio=float(brentq(lambda ratio: np.ptp(Gpassive@np.array([1.,ratio]))-
        np.ptp(Gcu@np.array([1.,ratio])), .1, 1.))
    mechanism = dict(passive_AlN_vs_Cu_crossover_PIC_EIC_ratio=cross_ratio,
        crossover_assumption='Constant-property linear source model, equal total geometric setup; outside the primary workload box',
        EIC_passive_AlN_to_Cu_differential_sensitivity_ratio=float(np.ptp(Gpassive[:,0])/np.ptp(Gcu[:,0])),
        PIC_passive_AlN_to_Cu_differential_sensitivity_ratio=float(np.ptp(Gpassive[:,1])/np.ptp(Gcu[:,1])),
        passive_AlN_zero_power_target_tolerance=passive_target_tolerance(Gpassive),
        common_command_certificate_improvement_percent={cid:float(100*(1-max(payload['common_corner_certificates_rad'][j])/
            max(payload['fixed_nominal_corner_certificates_rad'][j]))) for j,cid in enumerate(payload['case_id'])})
    (out/'workload_robust_mechanism.json').write_text(json.dumps(mechanism,indent=2)+'\n')
    verification = dict(source_unit_power_max_error_W_per_m=max_normalization,
        energy_max_relative_error=max_energy,elastic_equilibrium_max_relative_error=max_equilibrium,
        source_thermal_superposition_max_relative_error=max_source_superposition,
        nominal_source_phase_reassembly_max_error_rad=max_nominal_source_reconstruction,
        adversarial_corner_certificate_violation_rad=max_exact_control_corner_violation,
        sampled_interior_certificate_violation_rad=max_interior_violation,
        source_box_support_vs_vertices_max_error_rad=max_support_identity_error,
        adversarial_checks=5*2*4*36*2,random_interior_checks=5*2*1000,
        nominal_reference_comparisons=reference_rows,
        all_passed=bool(max(max_normalization,max_energy,max_equilibrium,max_source_superposition,
            max_nominal_source_reconstruction,max_exact_control_corner_violation,max_interior_violation,max_support_identity_error,
            *[max(v[k] for k in v if k!='case_id') for v in reference_rows])<1e-8))
    if not verification['all_passed']:raise AssertionError(verification)
    (out/'workload_robust_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    metadata = dict(model='Steady linear 2D thermoelastic/EIM model; constant source shapes and material properties',
        workload_envelope='Independent deterministic sensitivity box, not measured workload statistics',
        source_order=['EIC','PIC'],source_lower_mW=LOWER.tolist(),source_upper_mW=UPPER.tolist(),
        source_nominal_mW=NOMINAL.tolist(),source_vertices_mW=VERTICES.tolist(),
        designs=DESIGNS,mesh=[120,60],optical_length_m=.001,
        power_units='1 W/m line power equals 1 mW over the assigned 1 mm propagation length',
        error_model='Independent entrywise response intervals eta=.02 and sensed phase intervals eps=.002 rad',
        policies='Nominal-tuned fixed, common source-box-robust fixed, and separate steady per-vertex commands; no transient feedback',
        optical_metric='Normalized equal-amplitude lossless nine-input coherent-combiner output |mean exp(i phi)|²',
        optical_bound='For C < pi, a certified phase range implies efficiency >= cos²(C/2); no amplitude, propagation or coupling loss',
        physical_cases=5,loads_per_case=12,total_physical_right_hand_sides=60,
        pair_constraints_per_vertex=72,box_vertices=4,elapsed_seconds=time.time()-start,
        provenance_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'code/core.py',ROOT/'code/modal_coefficients.json']})
    (out/'workload_robust_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if make_plots:plot(out,payload,rows)
    return payload,rows


def plot(out,data,rows):
    plt.rcParams.update({'font.size':12.,'axes.labelsize':12.,'axes.titlesize':12.,'legend.fontsize':10.5,
                         'xtick.labelsize':10.5,'ytick.labelsize':11.,'pdf.fonttype':42})
    fig,axes = plt.subplots(2,2,figsize=(9.4,7.1),layout='constrained')
    colors = ['#486a89','#00877c','#be7034','#7052a3','#bd4260']
    labels = [d['label'] for d in DESIGNS]
    x = np.arange(5); width=.25
    for shift,policy,label,color in [(-1,'nominal_fixed','Nominal fixed','#7b97b4'),
                                    (0,'common_robust_fixed','Robust fixed','#00877c'),
                                    (1,'adaptive_vertices_only','Vertex-specific','#dcaf69')]:
        y=[]
        for j,cid in enumerate(data['case_id']):
            if policy=='adaptive_vertices_only':value=max(data['adaptive_corner_certificates_rad'][j])
            else:value=next(r['load_box_certificate_rad'] for r in rows if r['case_id']==cid and r['policy']==policy)
            y.append(value)
        axes[0,0].bar(x+shift*width,y,width,label=label,color=color)
    axes[0,0].set_xticks(x,labels,rotation=20,ha='right')
    axes[0,0].set_ylabel('Worst tested certificate (rad)')
    axes[0,0].set_title('(a) Fixed versus separate vertex commands')
    axes[0,0].legend(frameon=False,loc='upper right')
    for j,(G,H) in enumerate(zip(data['G_rad_per_mW'],data['H_rad_per_mW'])):
        certs = data['common_corner_certificates_rad'][j]
        axes[0,1].plot(range(4),certs,'o-',label=labels[j],color=colors[j])
    axes[0,1].set_xticks(range(4),['200/20','200/30','300/20','300/30'])
    axes[0,1].set_xlabel('EIC/PIC power (mW over 1 mm)')
    axes[0,1].set_ylabel('Common fixed certificate (rad)')
    axes[0,1].set_title('(b) Limiting workload corner')
    # The material colours in panels (b)-(d) share the legend in panel (d).
    ratios = np.linspace(.02,1.,197)
    # Independent differential slopes expose whether source columns are collinear.
    for j,G in enumerate(data['G_rad_per_mW']):
        phase = np.array([G@np.array([250.,250.*r]) for r in ratios])
        axes[1,0].plot(ratios,np.ptp(phase,axis=1),label=labels[j],color=colors[j])
    axes[1,0].axvspan(20/300,30/200,color='#dddddd',alpha=.45)
    cross=json.loads((out/'workload_robust_mechanism.json').read_text())['passive_AlN_vs_Cu_crossover_PIC_EIC_ratio']
    axes[1,0].axvline(cross,color='#333333',linestyle=':',alpha=.7)
    axes[1,0].set_xlabel('PIC/EIC load ratio (EIC = 250 mW)')
    axes[1,0].set_ylabel('Passive phase range (rad)')
    axes[1,0].set_title('(c) Passive ranking reverses with load ratio')
    for j,cid in enumerate(data['case_id']):
        selected = next(r for r in rows if r['case_id']==cid and r['policy']=='common_robust_fixed')
        axes[1,1].scatter(selected['sum_power_mW'],selected['certified_combiner_efficiency_lower_bound'],color=colors[j],s=44,label=labels[j])
    axes[1,1].set_xlabel('Common fixed heater power (mW)')
    axes[1,1].set_ylabel('Certified ideal combining efficiency')
    axes[1,1].set_title('(d) Optical guarantee versus power')
    axes[1,1].margins(x=.27,y=.22)
    axes[1,1].legend(frameon=False,loc='center left')
    for ax in axes.ravel():
        ax.grid(alpha=.2);ax.set_axisbelow(True)
    temporary_png=out/'workload_robust_comparison.tmp.png'
    fig.savefig(temporary_png,dpi=220)
    temporary_png.replace(out/'workload_robust_comparison.png')
    temporary=out/'workload_robust_comparison.tmp.pdf'
    fig.savefig(temporary)
    if not temporary.read_bytes().rstrip().endswith(b'%%EOF'):raise IOError('PDF incomplete')
    temporary.replace(out/'workload_robust_comparison.pdf')
    plt.close(fig)
    # Compact source-mechanism figure for readers who want to separate the
    # material/source interaction from the controller comparison.
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),layout='constrained')
    sx=np.arange(5)
    sensitivities=np.ptp(data['G_rad_per_mW'],axis=1)*1000
    axes[0].bar(sx-.18,sensitivities[:,0],.36,color='#486a89',label='EIC source')
    axes[0].bar(sx+.18,sensitivities[:,1],.36,color='#be7034',label='PIC source')
    axes[0].set_xticks(sx,labels,rotation=25,ha='right')
    axes[0].set_ylabel('Differential sensitivity (mrad/mW)')
    axes[0].set_title('(a) Source-specific differential response')
    axes[0].legend(frameon=False,loc='upper right')
    for j in [0,3]:
        G=data['G_rad_per_mW'][j]
        phase=np.array([G@np.array([250.,250.*r]) for r in ratios])
        axes[1].plot(ratios,np.ptp(phase,axis=1),color=colors[j],label=labels[j],linewidth=2)
    axes[1].axvspan(20/300,30/200,color='#dddddd',alpha=.6)
    axes[1].axvline(cross,color='#555555',linestyle=':',linewidth=1.5)
    crossing_phase=float(np.ptp(data['G_rad_per_mW'][0]@np.array([250.,250.*cross])))
    axes[1].scatter([cross],[crossing_phase],color='#333333',s=25,zorder=4)
    axes[1].annotate(f'Crossing = {cross:.3f}',(cross,crossing_phase),xytext=(.39,.27),
                     arrowprops={'arrowstyle':'->','color':'#555555'},fontsize=11)
    axes[1].set_xlabel('PIC/EIC ratio (EIC = 250 mW)')
    axes[1].set_ylabel('Passive phase range (rad)')
    axes[1].set_title('(b) Nominal passive gain has a load limit')
    axes[1].legend(frameon=False,loc='upper left')
    for ax in axes:
        ax.grid(alpha=.2);ax.set_axisbelow(True)
    tp=out/'workload_source_composition.tmp.png';fig.savefig(tp,dpi=220);tp.replace(out/'workload_source_composition.png')
    tp=out/'workload_source_composition.tmp.pdf';fig.savefig(tp)
    if not tp.read_bytes().rstrip().endswith(b'%%EOF'):raise IOError('PDF incomplete')
    tp.replace(out/'workload_source_composition.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    run(args.out,make_plots=not args.no_plots)
