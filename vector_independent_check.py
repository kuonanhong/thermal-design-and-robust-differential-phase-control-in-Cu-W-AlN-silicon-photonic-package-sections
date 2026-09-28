#!/usr/bin/env python3
"""Independent audit of frozen full-vector optical projections and controls.

No production solver is imported. No Maxwell eigenproblem is repeated. The
script independently rebuilds optical perturbations, coefficient chain rules,
ordered-pair linear programs, interval certificates, and exported summaries.
Run from any directory: python code/vector_independent_check.py --root ROOT
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def project(raw, index, prefix, c):
    """Independent scalar contraction; raw heater axes are (heater, lane)."""
    index_change = c['dn_eff_dT_K'] * raw[prefix+'_temperature_K'][index]
    for axis in 'xyz':
        index_change = index_change + c['dn_eff_d_elastic_strain_'+axis] * raw[prefix+'_elastic_strain_'+axis][index]
    for axis in 'xy':
        index_change = index_change + c['dn_eff_d_total_strain_'+axis] * raw[prefix+'_total_strain_'+axis][index]
    phase = 2*np.pi*c['path_length_m']/c['wavelength_m'] * index_change
    return phase.T if prefix == 'heater' else phase


def constraints(phi, H, pmax=20., budget=100., bits=8, eta=.02, eps=.002):
    q = pmax/(2**bits-1)
    A, b = [], []
    for a in range(len(phi)):
        for k in range(len(phi)):
            if a == k:
                continue
            difference = H[a]-H[k]
            width = np.abs(H[a])+np.abs(H[k])
            margin = 2*eps + q/2*np.sum(np.abs(difference)+eta*width)
            A.append(difference+eta*width)
            b.append(phi[k]-phi[a]-margin)
    return np.asarray(A), np.asarray(b), budget-H.shape[1]*q/2


def certificate(phi, H, command, quantization=False):
    q, eta, eps = 20/255, .02, .002
    best, pair = -np.inf, None
    for a in range(len(phi)):
        for k in range(len(phi)):
            if a == k:
                continue
            difference = H[a]-H[k]
            width = np.abs(H[a])+np.abs(H[k])
            value = phi[a]-phi[k] + difference@command + 2*eps + eta*width@command
            if quantization:
                value += q/2*np.sum(np.abs(difference)+eta*width)
            if value > best:
                best, pair = float(value), (a, k)
    return best, pair


def solve_lp(phi, H, target=None):
    A, b, budget = constraints(phi, H)
    m = H.shape[1]
    opts = {'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9}
    if target is not None:
        result = linprog(np.ones(m), A_ub=np.vstack([A, np.ones(m)]),
                         b_ub=np.r_[b+target,budget], bounds=[(0.,20.)]*m,
                         method='highs', options=opts)
        return result, None
    result = linprog(np.r_[np.zeros(m),1.],
                     A_ub=np.vstack([np.column_stack([A,-np.ones(len(A))]), np.r_[np.ones(m),0.]]),
                     b_ub=np.r_[b,budget], bounds=[(0.,20.)]*m+[(0.,None)], method='highs', options=opts)
    if not result.success:
        raise AssertionError('Independent minimax LP failed.')
    secondary = linprog(np.ones(m), A_ub=np.vstack([A,np.ones(m)]),
                        b_ub=np.r_[b+result.fun+1e-8,budget], bounds=[(0.,20.)]*m,
                        method='highs', options=opts)
    if not secondary.success:
        raise AssertionError('Independent secondary power LP failed.')
    return result, secondary


def run(root):
    root = Path(root).resolve()
    out = root/'results/vector'
    cpath = out/'vector_coefficients.json'
    c = json.loads(cpath.read_text(encoding='utf8'))
    values, passed = {}, {}

    def check(name, error, tolerance):
        error = float(error)
        values[name] = error
        passed[name] = bool(np.isfinite(error) and error <= tolerance)

    source_paths = [root/'results/physical_inputs.npz', root/'results/placement_physical_inputs.npz']
    sources = [np.load(path, allow_pickle=False) for path in source_paths]
    source = {str(name):(raw,i) for raw in sources for i,name in enumerate(raw['case_id'])}
    dpath = out/'vector_reprojected_inputs.npz'
    d = np.load(dpath, allow_pickle=False)
    rows = csv_rows(out/'vector_reprojected_cases.csv')
    index = {str(name):j for j,name in enumerate(d['case_id'])}
    check('reported_case_count', abs(len(index)-22), 0.)
    check('all_case_sources_present', sum(name not in source for name in index), 0.)
    coeffs = {}
    for mesh, step in [(10.,1e-4),(5.,1e-4),(2.5,1e-4),(2.5,5e-5)]:
        cc = json.loads((out/f'vector_coefficients_{mesh:g}nm_{step:g}.json').read_text(encoding='utf8'))
        coeffs[(mesh,step)] = cc
        suffix = f'{mesh:g}nm_{step:g}'
        thermal = cc['dn_eff_d_n_si']*cc['dn_si_dT_K'] + cc['dn_eff_d_n_clad']*cc['dn_clad_dT_K']
        check('thermooptic_chain_rule_'+suffix, abs(thermal-cc['dn_eff_dT_K']), 1e-14)
        anisotropic = np.array([cc['dn_eff_d_n_'+axis] for axis in 'xyz'])
        constitutive = np.full((3,3),cc['p12'])
        np.fill_diagonal(constitutive,cc['p11'])
        pe = -.5*cc['n_si']**3 * (anisotropic@constitutive)
        check('tensor_photoelastic_chain_rule_'+suffix, np.max(np.abs(pe-np.array([cc['dn_eff_d_elastic_strain_'+axis] for axis in 'xyz']))), 1e-12)
        check('width_affine_shape_chain_rule_'+suffix, abs(cc['dn_eff_d_width_per_m']*cc['width_m']-cc['dn_eff_d_total_strain_x']), 1e-12)
        check('height_affine_shape_chain_rule_'+suffix, abs(cc['dn_eff_d_height_per_m']*cc['height_m']-cc['dn_eff_d_total_strain_y']), 1e-12)
        check('isotropic_index_tensor_derivative_agreement_'+suffix, abs(anisotropic.sum()-cc['dn_eff_d_n_si']), 1e-6)
        check('reported_index_chain_rule_error_'+suffix, abs(abs(anisotropic.sum()-cc['dn_eff_d_n_si'])-cc['index_chain_rule_error']), 1e-14)
        check('plane_strain_longitudinal_path_coefficient_'+suffix, abs(cc['longitudinal_path_strain_coefficient']), 1e-15)
        check('symmetric_xy_derivative_near_zero_'+suffix, abs(cc['dn_eff_d_eps_xy']), 1e-6)
        check('reported_mode_tracking_overlap_'+suffix, max(0.,.99-cc['minimum_mode_overlap']), 0.)
        check('guided_index_range_'+suffix, max(cc['n_clad']-cc['neff'],cc['neff']-cc['n_si'],0.), 0.)
    check('selected_coefficients_equal_2p5nm_export', max(abs(float(c[k])-float(coeffs[(2.5,1e-4)][k])) for k in c if isinstance(c[k],(float,int))), 0.)

    phi_error, H_error = 0., 0.
    for name,j in index.items():
        raw,i = source[name]
        phi_error = max(phi_error,float(np.max(np.abs(project(raw,i,'background',c)-d['phi_rad'][j]))))
        H_error = max(H_error,float(np.max(np.abs(project(raw,i,'heater',c)-d['H_rad_per_mW'][j]))))
    check('independent_background_reprojection_rad', phi_error, 1e-11)
    check('independent_heater_reprojection_rad_per_mW', H_error, 1e-12)
    for left,right in [('AlN_R1e-08_k170_n120','AlN_L320_R880'),('graded_W_AlN_R1e-08_k170_n120','graded_W_AlN_L320_R880')]:
        check('duplicate_layout_phase_'+left,np.max(np.abs(d['phi_rad'][index[left]]-d['phi_rad'][index[right]])),1e-10)
        check('duplicate_layout_response_'+left,np.max(np.abs(d['H_rad_per_mW'][index[left]]-d['H_rad_per_mW'][index[right]])),1e-12)
    groups = {}
    for row in csv_rows(out/'vector_control_commands.csv'):
        groups.setdefault(row['case_id'],[]).append(row)
    extrema = {name:0. for name in ['summary_csv','command_array','power_bounds','total_budget','reserved_budget','nearest_quantization',
        'LP_primary_optimum','LP_secondary_power','post_exceeds_pre','exact_corner_attainment','sampled_error_box_violation']}
    rng = np.random.default_rng(20260925)
    def command_check(phi,H,p,pq):
        q = 20/255
        extrema['power_bounds'] = max(extrema['power_bounds'],float(np.max(-p)),float(np.max(p-20)),float(np.max(-pq)),float(np.max(pq-20)))
        extrema['total_budget'] = max(extrema['total_budget'],float(pq.sum()-100))
        extrema['reserved_budget'] = max(extrema['reserved_budget'],float(p.sum()-(100-H.shape[1]*q/2)))
        extrema['nearest_quantization'] = max(extrema['nearest_quantization'],float(np.max(np.abs(pq-np.rint(p/q)*q))))
        pre,_ = certificate(phi,H,p,True)
        post,(a,b) = certificate(phi,H,pq,False)
        extrema['post_exceeds_pre'] = max(extrema['post_exceeds_pre'],post-pre)
        e = np.zeros(len(phi));e[a]=.002;e[b]=-.002
        dh = np.zeros_like(H);dh[a]=.02*np.abs(H[a]);dh[b]=-.02*np.abs(H[b])
        extrema['exact_corner_attainment'] = max(extrema['exact_corner_attainment'],abs(np.ptp(phi+e+(H+dh)@pq)-post))
        for _ in range(16):
            phase = phi+rng.uniform(-.002,.002,len(phi))+(H+.02*np.abs(H)*rng.uniform(-1,1,H.shape))@pq
            extrema['sampled_error_box_violation'] = max(extrema['sampled_error_box_violation'],float(np.ptp(phase)-post))
        return pre,post

    for row in rows:
        cid = row['case_id'];j=index[cid];raw,source_index=source[cid]
        phi,H=d['phi_rad'][j],d['H_rad_per_mW'][j]
        cmd=sorted(groups[cid],key=lambda v:int(v['heater']))
        p=np.array([float(v['continuous_mW']) for v in cmd]);pq=np.array([float(v['quantized_mW']) for v in cmd])
        extrema['command_array']=max(extrema['command_array'],float(np.max(np.abs(pq-d['p_quantized_mW'][j]))))
        pre,post=command_check(phi,H,p,pq)
        primary,secondary=solve_lp(phi,H)
        extrema['LP_primary_optimum']=max(extrema['LP_primary_optimum'],abs(pre-primary.fun-1e-8))
        extrema['LP_secondary_power']=max(extrema['LP_secondary_power'],abs(p.sum()-secondary.fun))
        expected=dict(mean_phi_rad=np.mean(phi),passive_spread_rad=np.ptp(phi),H_diagonal_mean_rad_per_mW=np.diag(H).mean(),
                      robust_Cpost_rad=post,robust_power_mW=pq.sum(),mean_temperature_rise_K=raw['background_temperature_K'][source_index].mean())
        extrema['summary_csv']=max(extrema['summary_csv'],max(abs(float(row[k])-v) for k,v in expected.items()))
    target_rows=csv_rows(out/'vector_target025.csv');target_groups={}
    for row in csv_rows(out/'vector_target025_commands.csv'):
        key=(float(row['mesh_nm']),row['case_id']);target_groups.setdefault(key,[]).append(row)
    target_error,target_power_error,target_status_error,target_bound_violation=0.,0.,0.,0.
    target_feasible=0;target_selected=[]
    for row in target_rows:
        mesh=float(row['mesh_nm']);cid=row['case_id'];cc=coeffs[(mesh,1e-4)];raw,i=source[cid]
        phi,H=project(raw,i,'background',cc),project(raw,i,'heater',cc)
        zero=np.ptp(phi)+.004
        target_error=max(target_error,abs(np.ptp(phi)-float(row['passive_spread_rad'])),abs(zero-float(row['zero_power_Cpost_rad'])))
        target_status_error=max(target_status_error,float((zero<=.25)!=(row['zero_power_meets_0p25']=='True')))
        solved,_=solve_lp(phi,H,target=.25)
        target_status_error=max(target_status_error,float(solved.success!=(row['minimum_power_feasible']=='True')))
        if solved.success:
            cmd=sorted(target_groups[(mesh,cid)],key=lambda v:int(v['heater']))
            p=np.array([float(v['continuous_mW']) for v in cmd]);pq=np.array([float(v['quantized_mW']) for v in cmd])
            pre,post=command_check(phi,H,p,pq)
            target_power_error=max(target_power_error,abs(solved.fun-p.sum()))
            target_error=max(target_error,abs(pq.sum()-float(row['quantized_power_mW'])),abs(pre-float(row['Cpre_rad'])),abs(post-float(row['Cpost_rad'])))
            target_bound_violation=max(target_bound_violation,pre-.25,post-.25)
            target_feasible+=1
        if cid=='AlN_L320_R600':
            target_selected.append(dict(mesh_nm=mesh,zero_power_certificate_rad=float(zero),
                                        quantized_power_mW=float(row['quantized_power_mW']),certificate_rad=float(row['Cpost_rad'])))
    for name,error in extrema.items():
        check(name,error,1e-7 if name=='LP_secondary_power' else 3e-8 if name=='LP_primary_optimum' else 1e-9)
    check('target_all_mesh_rows_reconstruction',target_error,1e-10)
    check('target_LP_power_optimum_error_mW',target_power_error,1e-7)
    check('target_LP_and_zero_power_feasibility_agreement',target_status_error,0.)
    check('target_certificate_violation_rad',target_bound_violation,1e-9)
    check('selected_layout_zero_power_failure_all_meshes',float(not all(v['zero_power_certificate_rad']>.25 for v in target_selected)),0.)

    # Independently inspect the exported nominal mode fields and analytic-slab
    # dispersion. Exported eigensolver residuals are inspected, not recomputed.
    fields=np.load(out/'vector_nominal_fields.npz',allow_pickle=False)
    E=fields['E'];area=np.diff(fields['y_um'])[:,None]*np.diff(fields['x_um'])[None,:]
    ex_fraction=float(np.sum(area*np.abs(E[0])**2)/np.sum(area[None]*np.abs(E)**2))
    check('electric_field_polarization_reconstruction',abs(ex_fraction-float(fields['Ex_fraction'])),1e-12)
    check('nominal_field_effective_index_matches_coefficients',abs(float(fields['neff'])-c['neff']),1e-12)
    check('nominal_field_polarization_matches_coefficients',abs(ex_fraction-c['Ex_fraction']),1e-12)
    check('nominal_quasi_TE_fraction',max(0.,.5-ex_fraction),0.)
    benches=csv_rows(out/'vector_benchmarks.csv');benchmark_error=0.;dispersion_error=0.
    for row in benches:
        exact=float(row['exact_neff']);numerical=float(row['numerical_neff'])
        benchmark_error=max(benchmark_error,abs(abs(numerical-exact)-float(row['abs_error'])))
        if row['test']=='slab':
            u=np.pi*c['height_m']/c['wavelength_m']*np.sqrt(c['n_si']**2-exact**2)
            w=np.pi*c['height_m']/c['wavelength_m']*np.sqrt(exact**2-c['n_clad']**2)
            ratio=1. if row['polarization']=='TE' else c['n_si']**2/c['n_clad']**2
            dispersion_error=max(dispersion_error,abs(u*np.tan(u)-ratio*w))
        else:
            benchmark_error=max(benchmark_error,abs(exact-c['n_si']))
    check('benchmark_error_reconstruction',benchmark_error,1e-13)
    check('independent_analytic_slab_dispersion_residual',dispersion_error,1e-11)
    for pol in ['TE','TM']:
        errors=[float(row['abs_error']) for row in benches if row['test']=='slab' and row['polarization']==pol]
        check('slab_error_decreases_'+pol,max(0.,float(np.max(np.diff(errors)))),0.)
    mesh=csv_rows(out/'vector_mesh_convergence.csv');padding=csv_rows(out/'vector_padding_convergence.csv')
    check('transverse_mesh_grows_both_axes',float(not all(int(b['Nx'])>int(a['Nx']) and int(b['Ny'])>int(a['Ny']) for a,b in zip(mesh,mesh[1:]))),0.)
    mesh_delta=abs(float(mesh[-1]['neff'])-float(mesh[-2]['neff']))
    summary=json.loads((out/'vector_summary.json').read_text(encoding='utf8'))
    check('mesh_summary_absolute_change',abs(mesh_delta-summary['mesh_2p5_to_1p25_absolute_neff_change']),1e-13)
    check('mesh_summary_relative_change',abs(abs(float(mesh[-1]['neff'])/float(mesh[-2]['neff'])-1)-summary['mesh_2p5_to_1p25_relative_neff_change']),1e-13)
    check('padding_summary_change',abs(abs(float(padding[-1]['neff'])-float(padding[-2]['neff']))-summary['padding_1p2_to_1p6_absolute_neff_change']),1e-13)
    for field,label in [('derivative_5_to_2p5_relative_changes',((5.,1e-4),(2.5,1e-4))),
                        ('derivative_step_halving_relative_changes',((2.5,1e-4),(2.5,5e-5)))]:
        error=max(abs(abs(coeffs[label[1]][key]/coeffs[label[0]][key]-1)-value) for key,value in summary[field].items())
        check(field+'_reconstruction',error,1e-13)
    cu=d['phi_rad'][index['Cu_R0e+00_k170_n120']];aln=d['phi_rad'][index['AlN_L320_R600']]
    reduction=float(100*(1-np.ptp(aln)/np.ptp(cu)))
    check('passive_reduction_percent_reconstruction',abs(reduction-summary['AlN_320_600_reduction_percent']),1e-10)
    check('best_passive_layout_identity',float(min(rows,key=lambda row:float(row['passive_spread_rad']))['case_id']!=summary['best_passive']['case_id']),0.)
    check('best_controlled_layout_identity',float(min(rows,key=lambda row:float(row['robust_Cpost_rad']))['case_id']!=summary['best_controlled']['case_id']),0.)
    check('reported_maximum_eigenpair_residual',summary['maximum_eigenpair_residual'],1e-8)
    eim=json.loads((root/'code/modal_coefficients.json').read_text(encoding='utf8'))
    raw,i=source['AlN_L320_R600'];phi_eim=project(raw,i,'background',eim)
    comparison=dict(full_vector_neff=c['neff'],EIM_neff=eim['neff'],
                    selected_AlN_passive_reduction_percent=reduction,
                    EIM_zero_power_certificate_rad=float(np.ptp(phi_eim)+.004),
                    vector_zero_power_certificate_rad=float(np.ptp(aln)+.004),
                    selected_layout_target_by_optical_mesh=target_selected,
                    reported_eigenproblems=int(summary['eigenproblems']),
                    reported_maximum_eigenpair_residual=summary['maximum_eigenpair_residual'])
    check('EIM_zero_power_conclusion_changes_under_vector_model',float(not (comparison['EIM_zero_power_certificate_rad']<=.25<comparison['vector_zero_power_certificate_rad'])),0.)
    counts=dict(reported_projection_cases=22,unique_physical_layouts=20,minimax_LPs_independently_resolved=22,
                target_LPs_independently_resolved=len(target_rows),target_feasible_commands_checked=target_feasible,
                exact_adversarial_corners_checked=22+target_feasible,random_error_draws_checked=16*(22+target_feasible))
    scope=['No production projection, control, or Maxwell solver was imported.',
           'No costly eigenproblem was rerun; eigenpair residuals and mode tracking are inspected from exports, not independently recomputed.',
           'The waveguide calculation is 2D full vector; package thermomechanics remain a 2D plane-strain extruded model.',
           'Photoelastic constants remain nominal assumptions; cladding photoelasticity and full-package optical scattering are excluded.',
           'Continuous LP optimality and the attained fixed-command interval certificates are checked; global integer-DAC power optimality is not claimed.',
           'The 2% calibration box does not represent the difference between EIM and full-vector models.',
           'The calculations are numerical verification, not experimental validation or a publication guarantee.']
    output=dict(all_passed=all(passed.values()),checks=passed,metrics=values,counts=counts,comparison=comparison,scope=scope,
                source_hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [*source_paths,cpath,dpath,Path(__file__)]})
    destination=root/'results/vector_independent_verification.json'
    destination.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(all_passed=output['all_passed'],failed_checks=[name for name,status in passed.items() if not status],counts=counts,comparison=comparison,output=str(destination)),indent=2))
    if not output['all_passed']:
        raise AssertionError('Independent vector-result verification failed.')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
