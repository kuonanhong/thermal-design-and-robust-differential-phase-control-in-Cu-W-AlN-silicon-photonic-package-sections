#!/usr/bin/env python3
"""Independently verify the exported workload-box optical-control study.

This module deliberately imports no project solver. It reconstructs the LP
using every ordered channel pair, checks interval-certificate attainment by
explicit adversarial perturbations, and checks the finite-box vertex argument
at independently sampled interior loads. These are numerical/model checks,
not experimental validation or an optimality certificate for a discrete DAC.

Run from any directory: python code/workload_independent_check.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def ordered_lp(phases, H, pmax, budget, bits, eta, eps, include_dac_margin=True,
               return_solver_result=False):
    """Rebuild the continuous robust LP with all n(n-1) ordered pairs."""
    phases = np.atleast_2d(phases)
    n, m = H.shape
    q = pmax / (2**bits - 1) if include_dac_margin else 0.
    A, rhs = [], []
    for phi in phases:
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                difference = H[a] - H[b]
                width = np.abs(H[a]) + np.abs(H[b])
                reserve = 2 * eps + q / 2 * np.sum(np.abs(difference) + eta * width)
                A.append(np.r_[difference + eta * width, -1.])
                rhs.append(phi[b] - phi[a] - reserve)
    A.append(np.r_[np.ones(m), 0.])
    rhs.append(budget - m * q / 2)
    A, rhs = np.asarray(A), np.asarray(rhs)
    result = linprog(np.r_[np.zeros(m), 1.], A_ub=A, b_ub=rhs,
                     bounds=[(0., pmax)] * m + [(0., None)], method='highs',
                     options={'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9})
    if not result.success:
        raise AssertionError('Independent LP failed: ' + result.message)
    # Independently reconstruct the same declared lexicographic objective.
    fixed_rhs = rhs[:-1] + result.fun + 1e-8
    power = linprog(np.ones(m), A_ub=np.vstack([A[:-1, :-1], np.ones(m)]),
                    b_ub=np.r_[fixed_rhs, budget - m * q / 2],
                    bounds=[(0., pmax)] * m, method='highs',
                    options={'primal_feasibility_tolerance': 1e-9,
                             'dual_feasibility_tolerance': 1e-9})
    if not power.success:
        raise AssertionError('Independent secondary LP failed: ' + power.message)
    output = (result.fun, power.fun, A, rhs)
    return (*output, result) if return_solver_result else output


def verify_lp_dual(result, A, rhs, pmax):
    """Dual residuals and a lower bound for the exported floating-point LP.

    Nonnegative pair weights sum to one; the nonnegative budget multiplier
    adds lambda*(sum(p)-budget). Minimizing that affine expression over the
    heater box gives a lower bound on the minimax optimum. Fraction arithmetic
    treats every assembled floating-point coefficient as its exact binary
    rational value, including normalization, then rounds the bound downward.
    This certifies that numerical LP, not the physical model or a continuum
    discretization error. No project LP builder is imported.
    """
    m = A.shape[1]-1
    if not (np.all(A[:-1, -1] == -1.) and A[-1, -1] == 0.
            and np.all(A[-1, :-1] == 1.)):
        raise AssertionError('Unexpected LP rows for the dual construction.')
    objective = np.r_[np.zeros(m), 1.]
    y = np.asarray(result.ineqlin.marginals)
    lower = np.asarray(result.lower.marginals)
    upper = np.asarray(result.upper.marginals)
    x = np.asarray(result.x)
    stationarity = objective-A.T@y-lower-upper
    dual_value = float(rhs@y + pmax*np.sum(upper[:m]))
    primal_violation = max(0., float(np.max(A@x-rhs)), float(np.max(-x)),
                           float(np.max(x[:m]-pmax)))
    sign_violation = max(0., float(np.max(y)), float(np.max(-lower)),
                         float(np.max(upper)), abs(float(upper[-1])))
    complementarity = max(float(np.max(np.abs(y*(rhs-A@x)))),
                          float(np.max(np.abs(lower*x))),
                          float(np.max(np.abs(upper[:m]*(pmax-x[:m])))))
    # Discard numerically inadmissible multiplier signs before forming the bound.
    weights = [Fraction.from_float(float(max(0., -v))) for v in y[:-1]]
    scale = sum(weights, Fraction(0))
    if scale <= 0:
        raise AssertionError('No nonzero admissible pair multipliers.')
    weights = [v/scale for v in weights]
    multiplier = Fraction.from_float(float(max(0., -y[-1])))/scale
    exact = -sum((w*Fraction.from_float(float(b)) for w,b in zip(weights,rhs[:-1])), Fraction(0))
    exact -= multiplier*Fraction.from_float(float(rhs[-1]))
    for j in range(m):
        slope = multiplier + sum((w*Fraction.from_float(float(v))
                                  for w,v in zip(weights,A[:-1,j])), Fraction(0))
        exact += Fraction.from_float(float(pmax))*min(Fraction(0), slope)
    downward = float(exact)
    if Fraction.from_float(downward) > exact:
        downward = float(np.nextafter(downward, -np.inf))
    return dict(primal_objective_rad=float(result.fun), dual_objective_rad=dual_value,
                absolute_primal_dual_gap_rad=abs(float(result.fun)-dual_value),
                primal_feasibility_violation=primal_violation,
                dual_sign_violation=sign_violation,
                stationarity_max_absolute_residual=float(np.max(np.abs(stationarity))),
                complementarity_max_absolute_residual=complementarity,
                rational_lower_bound_rad=downward,
                rational_bound_numerator=str(exact.numerator),
                rational_bound_denominator=str(exact.denominator),
                primal_minus_rational_lower_bound_rad=float(result.fun)-downward,
                nonzero_pair_multipliers=sum(w > 0 for w in weights),
                budget_multiplier=float(multiplier),
                excludes_0p25_rad=bool(exact > Fraction(1,4)),
                scope='Exact-rational dual lower bound for assembled binary floating-point LP data; not a physical-model or continuum-error certificate.')


def exact_certificate(phi, H, command, eta, eps):
    """Fixed-command exact entrywise-box phase-width certificate."""
    actual = phi + H @ command
    best = (-np.inf, None)
    for a in range(len(phi)):
        for b in range(len(phi)):
            if a == b:
                continue
            value = actual[a] - actual[b] + 2 * eps
            value += eta * ((np.abs(H[a]) + np.abs(H[b])) @ command)
            if value > best[0]:
                best = (float(value), (a, b))
    return best


def pre_certificate(phases, H, command, pmax, bits, eta, eps):
    q = pmax / (2**bits - 1)
    best = -np.inf
    for phi in np.atleast_2d(phases):
        value = phi + H @ command
        for a in range(len(phi)):
            for b in range(len(phi)):
                if a == b:
                    continue
                difference = H[a] - H[b]
                width = np.abs(H[a]) + np.abs(H[b])
                v = value[a] - value[b] + 2 * eps + eta * width @ command
                v += q / 2 * np.sum(np.abs(difference) + eta * width)
                best = max(best, float(v))
    return best


def run(root):
    root = Path(root).resolve()
    result_dir = root / 'results'
    path = result_dir / 'workload_robust_inputs.npz'
    d = np.load(path, allow_pickle=False)
    checks, metrics, descriptions = {}, {}, {}

    def check(name, value, tolerance, description=''):
        value = float(value)
        metrics[name] = value
        checks[name] = bool(np.isfinite(value) and value <= tolerance)
        if description:
            descriptions[name] = description

    pmax = float(d['pmax_mW'])
    budget = float(d['total_budget_mW'])
    bits = int(d['bits'])
    eta = float(d['eta'])
    eps = float(d['eps_rad'])
    q = pmax / (2**bits - 1)
    G, H_all = d['G_rad_per_mW'], d['H_rad_per_mW']
    lower, upper = d['source_lower_mW'], d['source_upper_mW']
    nominal, vertices = d['source_nominal_mW'], d['source_vertices_mW']
    expected_vertices = np.asarray(list(itertools.product(*zip(lower, upper))))
    check('source_vertex_enumeration_error_mW', np.max(np.abs(vertices - expected_vertices)), 0.)
    check('declared_source_halfwidth_relative_error', np.max(np.abs((upper - lower) / (2 * nominal) - .2)), 1e-14)
    check('source_box_center_error_mW', np.max(np.abs((upper + lower) / 2 - nominal)), 1e-13)
    check('source_names_order', float(list(d['source_names']) != ['EIC', 'PIC']), 0.)
    check('nominal_background_reconstruction_rad', np.max(np.abs(np.einsum('cns,s->cn', G, nominal) - d['nominal_phase_rad'])), 1e-10)
    check('number_of_designs', abs(len(G) - 5), 0.)
    check('finite_source_and_heater_responses', float(not (np.isfinite(G).all() and np.isfinite(H_all).all())), 0.)

    # Independently reconstruct the optical projection from all 60 exported
    # thermomechanical load solutions, including an independently assembled
    # nominal background. No project projection function is imported.
    raw_path = result_dir / 'workload_robust_physical_fields.npz'
    raw = np.load(raw_path, allow_pickle=False)
    coeff_path = root / 'code/modal_coefficients.json'
    c = json.loads(coeff_path.read_text(encoding='utf8'))
    factor = 2 * np.pi * c['path_length_m'] / c['wavelength_m']
    dn = c['dn_eff_dT_K'] * raw['temperature_K']
    dn += sum(c['dn_eff_d_elastic_strain_' + axis] * raw['elastic_strain_' + axis] for axis in 'xyz')
    dn += sum(c['dn_eff_d_total_strain_' + axis] * raw['total_strain_' + axis] for axis in 'xy')
    projected = factor * dn
    check('source_optical_reprojection_rad_per_mW', np.max(np.abs(projected[:, :2].transpose(0, 2, 1) - G)), 1e-12)
    check('heater_optical_reprojection_rad_per_mW', np.max(np.abs(projected[:, 2:11].transpose(0, 2, 1) - H_all)), 1e-12)
    check('nominal_direct_optical_superposition_rad', np.max(np.abs(projected[:, :2].transpose(0, 2, 1) @ nominal - projected[:, 11])), 1e-10)
    check('source_temperature_export_error_K_per_mW', np.max(np.abs(raw['temperature_K'][:, :2].transpose(0, 2, 1) - d['source_temperature_K_per_mW'])), 1e-13)
    check('heater_temperature_export_error_K_per_mW', np.max(np.abs(raw['temperature_K'][:, 2:11].transpose(0, 2, 1) - d['heater_temperature_K_per_mW'])), 1e-13)
    check('line_power_unit_at_one_mm', abs(c['path_length_m'] * 1000 - 1), 1e-14)
    for field in ['temperature_K', 'total_strain_x', 'total_strain_y', 'total_strain_z',
                  'elastic_strain_x', 'elastic_strain_y', 'elastic_strain_z',
                  'stress_x_Pa', 'stress_y_Pa', 'stress_z_Pa']:
        recomposed = np.einsum('csn,s->cn', raw[field][:, :2], nominal)
        norm = max(float(np.max(np.abs(raw[field][:, 11]))), 1e-15)
        check('nominal_direct_' + field + '_relative_superposition', np.max(np.abs(recomposed - raw[field][:, 11])) / norm, 1e-9)
    for axis in 'xyz':
        elastic = raw['total_strain_' + axis] - c['alpha_si_per_K'] * raw['temperature_K']
        check('stress_free_thermal_subtraction_' + axis, np.max(np.abs(elastic - raw['elastic_strain_' + axis])), 1e-13)
    check('plane_strain_ezz', np.max(np.abs(raw['total_strain_z'])), 1e-15)

    existing = {}
    for file in ['control_inputs.npz', 'placement_inputs.npz']:
        original = np.load(result_dir / file, allow_pickle=False)
        for index, name in enumerate(original['case_id']):
            existing[str(name)] = (original['phi_rad'][index], original['H_rad_per_mW'][index])
    background_error, response_error, matched = 0., 0., 0
    original_names = dict(Cu='Cu_R0e+00_k170_n120', W='graded_W_R0e+00_k170_n120',
                          AlN_original='AlN_R1e-08_k170_n120', AlN_passive='AlN_L320_R600',
                          WAlN_control='graded_W_AlN_L180_R880')
    for index, name in enumerate(d['case_id']):
        # Case IDs are expected to preserve physical input identity. Explicit
        # source_case_id is accepted if the presentation uses shorter labels.
        source_name = str(d['source_case_id'][index]) if 'source_case_id' in d.files else original_names.get(str(name), str(name))
        if source_name in existing:
            previous_phi, previous_H = existing[source_name]
            background_error = max(background_error, float(np.max(np.abs(G[index] @ nominal - previous_phi))))
            response_error = max(response_error, float(np.max(np.abs(H_all[index] - previous_H))))
            matched += 1
    check('legacy_background_agreement_rad', background_error, 1e-9)
    check('legacy_heater_agreement_rad_per_mW', response_error, 1e-10)
    check('legacy_case_match_count', abs(matched - len(G)), 0.)
    metrics['legacy_cases_matched'] = matched

    rng = np.random.default_rng(20260925)
    maxima = {name: 0. for name in [
        'continuous_bound_violation', 'quantized_bound_violation', 'total_budget_violation',
        'reserved_continuous_budget_violation', 'DAC_grid_error', 'nearest_rounding_error',
        'pre_export_error', 'LP_optimum_error', 'LP_command_feasibility', 'lexicographic_power_error',
        'post_minus_pre', 'certificate_attainment_error', 'interior_minus_vertex',
        'interior_superposition_error', 'direct_combiner_minus_lower_bound',
        'perturbed_combiner_minus_lower_bound', 'common_shift_combiner_error',
        'phase_width_exceeds_certificate', 'common_optimum_below_vertex_optimum',
        'corner_certificate_export_error',
    ]}
    all_rows, lp_count, corner_count, interior_count, perturbation_count = [], 0, 0, 0, 0
    for case in range(len(G)):
        H = H_all[case]
        phases = vertices @ G[case].T
        phi_nominal = G[case] @ nominal
        exact_by_mode = {}
        policies = [('nominal', [phi_nominal], None), ('common', phases, None)]
        policies += [('adaptive', [phases[v]], v) for v in range(len(vertices))]
        optimal = {}
        for mode, optimization_phases, vertex_index in policies:
            suffix = (case,) if vertex_index is None else (case, vertex_index)
            p = d['continuous_commands_' + mode + '_mW'][suffix]
            pq = d['commands_' + mode + '_mW'][suffix]
            declared_pre = float(d[mode + '_pre_rad'][suffix])
            n, m = H.shape
            maxima['continuous_bound_violation'] = max(maxima['continuous_bound_violation'], float(np.max(-p)), float(np.max(p - pmax)))
            maxima['quantized_bound_violation'] = max(maxima['quantized_bound_violation'], float(np.max(-pq)), float(np.max(pq - pmax)))
            maxima['total_budget_violation'] = max(maxima['total_budget_violation'], float(pq.sum() - budget))
            maxima['reserved_continuous_budget_violation'] = max(maxima['reserved_continuous_budget_violation'], float(p.sum() - (budget - m * q / 2)))
            maxima['DAC_grid_error'] = max(maxima['DAC_grid_error'], float(np.max(np.abs(pq / q - np.rint(pq / q)))))
            maxima['nearest_rounding_error'] = max(maxima['nearest_rounding_error'], float(np.max(np.abs(pq - np.clip(np.rint(p / q) * q, 0., pmax)))))
            reconstructed_pre = pre_certificate(optimization_phases, H, p, pmax, bits, eta, eps)
            optimum, lex_power, A, b = ordered_lp(optimization_phases, H, pmax, budget, bits, eta, eps)
            # Exported pre may be the primary optimum or the achieved
            # lexicographic certificate (differing by <= 1e-8 rad).
            maxima['pre_export_error'] = max(maxima['pre_export_error'], abs(reconstructed_pre - declared_pre))
            maxima['LP_optimum_error'] = max(maxima['LP_optimum_error'], abs(optimum - declared_pre))
            maxima['LP_command_feasibility'] = max(maxima['LP_command_feasibility'], float(np.max(A @ np.r_[p, optimum + 1e-8] - b)))
            maxima['lexicographic_power_error'] = max(maxima['lexicographic_power_error'], abs(p.sum() - lex_power))
            optimal[(mode, vertex_index)] = optimum
            lp_count += 1
            evaluation_indices = list(range(len(vertices))) if mode != 'adaptive' else [vertex_index]
            certificates = []
            for v in evaluation_indices:
                phi = phases[v]
                cert, (a, b_lane) = exact_certificate(phi, H, pq, eta, eps)
                certificates.append(cert)
                export_key = {'nominal':'fixed_nominal_corner_certificates_rad',
                              'common':'common_corner_certificates_rad',
                              'adaptive':'adaptive_corner_certificates_rad'}[mode]
                maxima['corner_certificate_export_error'] = max(maxima['corner_certificate_export_error'], abs(cert - float(d[export_key][case, v])))
                actual_phase = phi + H @ pq
                direct_efficiency = float(abs(np.mean(np.exp(1j * actual_phase)))**2)
                bound_efficiency = float(np.cos(cert / 2)**2) if cert < np.pi else 0.
                maxima['direct_combiner_minus_lower_bound'] = max(maxima['direct_combiner_minus_lower_bound'], bound_efficiency - direct_efficiency)
                maxima['common_shift_combiner_error'] = max(maxima['common_shift_combiner_error'], abs(direct_efficiency - float(abs(np.mean(np.exp(1j * (actual_phase + 19.37))))**2)))
                # Explicitly realize the ordered pair's maximizing independent
                # calibration and additive observation error box corner.
                dH = np.zeros_like(H)
                dH[a] = eta * np.abs(H[a])
                dH[b_lane] = -eta * np.abs(H[b_lane])
                e = np.zeros(n)
                e[a], e[b_lane] = eps, -eps
                adversarial = phi + (H + dH) @ pq + e
                attained = float(np.ptp(adversarial))
                maxima['certificate_attainment_error'] = max(maxima['certificate_attainment_error'], abs(attained - cert))
                adversarial_efficiency = float(abs(np.mean(np.exp(1j * adversarial)))**2)
                maxima['perturbed_combiner_minus_lower_bound'] = max(maxima['perturbed_combiner_minus_lower_bound'], bound_efficiency - adversarial_efficiency)
                for _ in range(12):
                    dh_draw = rng.uniform(-1., 1., H.shape) * eta * np.abs(H)
                    e_draw = rng.uniform(-eps, eps, n)
                    perturbed = phi + (H + dh_draw) @ pq + e_draw
                    maxima['phase_width_exceeds_certificate'] = max(maxima['phase_width_exceeds_certificate'], float(np.ptp(perturbed)) - cert)
                    maxima['perturbed_combiner_minus_lower_bound'] = max(maxima['perturbed_combiner_minus_lower_bound'], bound_efficiency - float(abs(np.mean(np.exp(1j * perturbed)))**2))
                    perturbation_count += 1
                row = dict(case_id=str(d['case_id'][case]), policy=mode, vertex=v,
                           certificate_rad=cert, nominal_pair_rad=float(np.ptp(actual_phase)),
                           ideal_MZI_penalty_dB=float(-10 * np.log10(np.cos(cert / 2)**2)) if cert < np.pi else None,
                           ideal_combiner_efficiency=direct_efficiency,
                           ideal_combiner_efficiency_lower_bound=bound_efficiency,
                           quantized_power_mW=float(pq.sum()))
                all_rows.append(row)
                corner_count += 1
            exact_by_mode[(mode, vertex_index)] = max(certificates)
            if mode == 'common' or mode == 'adaptive':
                maxima['post_minus_pre'] = max(maxima['post_minus_pre'], max(certificates) - reconstructed_pre)
            else:
                nominal_cert, _ = exact_certificate(phi_nominal, H, pq, eta, eps)
                maxima['post_minus_pre'] = max(maxima['post_minus_pre'], nominal_cert - reconstructed_pre)
            if mode != 'adaptive':
                vertex_bound = max(certificates)
                for _ in range(64):
                    fraction = rng.uniform(0., 1., len(nominal))
                    workload = lower + fraction * (upper - lower)
                    weights = np.asarray([np.prod([fraction[j] if vertex[j] == upper[j] else 1 - fraction[j] for j in range(len(nominal))]) for vertex in vertices])
                    direct = G[case] @ workload
                    convex = weights @ phases
                    maxima['interior_superposition_error'] = max(maxima['interior_superposition_error'], float(np.max(np.abs(direct - convex))))
                    certificate, _ = exact_certificate(direct, H, pq, eta, eps)
                    maxima['interior_minus_vertex'] = max(maxima['interior_minus_vertex'], certificate - vertex_bound)
                    interior_count += 1
        for v in range(len(vertices)):
            maxima['common_optimum_below_vertex_optimum'] = max(maxima['common_optimum_below_vertex_optimum'], optimal[('adaptive', v)] - optimal[('common', None)])

    for name, value in maxima.items():
        tolerance = 3e-8 if name in ['pre_export_error', 'LP_optimum_error'] else 1e-6 if name == 'lexicographic_power_error' else 1e-9
        check(name, value, tolerance)

    # Check every exported numeric CSV claim, including passive baselines and
    # the out-of-box load-map sweep, without importing the production formula.
    def read_csv(name):
        with (result_dir / name).open(newline='', encoding='utf8') as stream:
            return list(csv.DictReader(stream))

    indices = {str(name): j for j, name in enumerate(d['case_id'])}
    policies = {'nominal_fixed':'nominal', 'common_robust_fixed':'common', 'adaptive_vertex':'adaptive'}
    vertex_rows = read_csv('workload_robust_vertices.csv')
    csv_errors = dict(certificate=0., phase=0., efficiency=0., efficiency_bound=0., penalty=0., power=0., source_power=0.)
    for row in vertex_rows:
        j, v = indices[row['case_id']], int(row['vertex'])
        H, source = H_all[j], G[j]
        if row['policy'] == 'passive':
            command = np.zeros(H.shape[1])
        else:
            mode = policies[row['policy']]
            command = d['commands_' + mode + '_mW'][j]
            if mode == 'adaptive':
                command = command[v]
        phase = source @ vertices[v] + H @ command
        cert, _ = exact_certificate(source @ vertices[v], H, command, eta, eps)
        lb = float(np.cos(cert / 2)**2) if cert < np.pi else 0.
        csv_errors['certificate'] = max(csv_errors['certificate'], abs(cert - float(row['certificate_rad'])))
        csv_errors['phase'] = max(csv_errors['phase'], abs(np.ptp(phase) - float(row['phase_range_rad'])))
        csv_errors['efficiency'] = max(csv_errors['efficiency'], abs(abs(np.mean(np.exp(1j * phase)))**2 - float(row['nominal_combiner_efficiency'])))
        csv_errors['efficiency_bound'] = max(csv_errors['efficiency_bound'], abs(lb - float(row['certified_combiner_efficiency_lower_bound'])))
        csv_errors['penalty'] = max(csv_errors['penalty'], abs(-10*np.log10(lb) - float(row['certified_phase_only_penalty_dB'])))
        csv_errors['power'] = max(csv_errors['power'], abs(command.sum() - float(row['sum_power_mW'])))
        csv_errors['source_power'] = max(csv_errors['source_power'], abs(vertices[v, 0] - float(row['EIC_mW'])), abs(vertices[v, 1] - float(row['PIC_mW'])))
    for name, value in csv_errors.items():
        check('vertex_csv_' + name, value, 1e-10)
    summary_rows = read_csv('workload_robust_summary.csv')
    summary_error, adaptive_box_misstatement = 0., 0.
    for row in summary_rows:
        j = indices[row['case_id']]
        is_adaptive = row['policy'] == 'adaptive_vertices_only'
        detail_policy = 'adaptive_vertex' if is_adaptive else row['policy']
        details = [v for v in vertex_rows if v['case_id'] == row['case_id'] and v['policy'] == detail_policy]
        certificates = [float(v['certificate_rad']) for v in details]
        expected = dict(worst_corner_nominal_phase_range_rad=max(float(v['phase_range_rad']) for v in details),
                        sum_power_mW=max(float(v['sum_power_mW']) for v in details),
                        minimum_nominal_vertex_combiner_efficiency=min(float(v['nominal_combiner_efficiency']) for v in details),
                        certified_combiner_efficiency_lower_bound=min(float(v['certified_combiner_efficiency_lower_bound']) for v in details),
                        certified_phase_only_penalty_dB=max(float(v['certified_phase_only_penalty_dB']) for v in details))
        if is_adaptive:
            adaptive_box_misstatement = max(adaptive_box_misstatement, float(row['load_box_certificate_rad'] != ''), float(row['nominal_load_phase_range_rad'] != ''))
            expected['pre_round_objective_rad'] = float(np.max(d['adaptive_pre_rad'][j]))
        else:
            expected['load_box_certificate_rad'] = max(certificates)
            if row['policy'] == 'passive':
                command = np.zeros(H_all[j].shape[1])
            else:
                mode = policies[row['policy']]
                command = d['commands_' + mode + '_mW'][j]
                expected['pre_round_objective_rad'] = float(d[mode + '_pre_rad'][j])
            expected['nominal_load_phase_range_rad'] = float(np.ptp(G[j] @ nominal + H_all[j] @ command))
        for key, value in expected.items():
            summary_error = max(summary_error, abs(float(row[key]) - value))
        worst = int(row['worst_vertex'])
        summary_error = max(summary_error, max(certificates) - float(next(v['certificate_rad'] for v in details if int(v['vertex']) == worst)))
    check('summary_csv_aggregation', summary_error, 1e-10)
    check('adaptive_not_mislabeled_box_policy', adaptive_box_misstatement, 0.)
    load_map = read_csv('workload_robust_load_map.csv')
    load_map_error = 0.
    for row in load_map:
        j = indices[row['case_id']]
        phi = G[j] @ np.array([float(row['EIC_mW']), float(row['PIC_mW'])])
        load_map_error = max(load_map_error, abs(np.ptp(phi) - float(row['passive_phase_range_rad'])))
        for policy in ['nominal', 'common']:
            cert, _ = exact_certificate(phi, H_all[j], d['commands_' + policy + '_mW'][j], eta, eps)
            load_map_error = max(load_map_error, abs(cert - float(row[policy + '_fixed_certificate_rad'])))
    check('load_map_all_certificates_rad', load_map_error, 1e-10)
    source_rows = read_csv('workload_robust_sources.csv')
    source_error = 0.
    for row in source_rows:
        j, s = indices[row['case_id']], ['EIC', 'PIC'].index(row['source'])
        source_error = max(source_error, abs(np.mean(G[j, :, s]) - float(row['mean_phase_sensitivity_rad_per_mW'])),
                           abs(np.ptp(G[j, :, s]) - float(row['phase_spread_sensitivity_rad_per_mW'])),
                           abs(np.mean(d['source_temperature_K_per_mW'][j, :, s]) - float(row['mean_temperature_sensitivity_K_per_mW'])))
    check('source_sensitivity_csv_error', source_error, 1e-12)
    support_error = 0.
    for j in range(len(G)):
        for command in [np.zeros(H_all[j].shape[1]), d['commands_nominal_mW'][j], d['commands_common_mW'][j], *d['commands_adaptive_mW'][j]]:
            corner_value = max(exact_certificate(G[j] @ load, H_all[j], command, eta, eps)[0] for load in vertices)
            ordered_support = []
            for a in range(G.shape[1]):
                for b_lane in range(G.shape[1]):
                    if a == b_lane:
                        continue
                    source_difference = G[j, a] - G[j, b_lane]
                    heater_difference = H_all[j, a] - H_all[j, b_lane]
                    calibration_width = np.abs(H_all[j, a]) + np.abs(H_all[j, b_lane])
                    ordered_support.append(source_difference @ nominal + np.abs(source_difference) @ ((upper-lower)/2)
                                           + heater_difference @ command + eta*calibration_width @ command + 2*eps)
            support_error = max(support_error, abs(corner_value - max(ordered_support)))
    check('support_function_equals_vertex_enumeration_rad', support_error, 1e-10)
    tolerance_rows = read_csv('workload_robust_passive_tolerance.csv')
    tolerance_error, nominal_status_error, threshold_error = 0., 0., 0.
    for row in tolerance_rows:
        j = indices[row['case_id']]
        target = float(row['target_rad'])
        widths = []
        for a in range(G.shape[1]):
            for b_lane in range(a+1, G.shape[1]):
                difference = G[j, a] - G[j, b_lane]
                rate = np.abs(difference) @ nominal
                if rate > 1e-15:
                    widths.append((target - 2*eps - abs(difference @ nominal))/rate)
        raw_delta = min(widths)
        nominal_passes = np.ptp(G[j] @ nominal) + 2*eps <= target
        delta = min(raw_delta, 1.) if nominal_passes else 0.
        tolerance_error = max(tolerance_error, abs(raw_delta - float(row['raw_algebraic_halfwidth'])), abs(delta - float(row['fractional_halfwidth'])))
        nominal_status_error = max(nominal_status_error, float(nominal_passes != (row['nominal_passes'] == 'True')))
        if nominal_passes and delta < 1:
            def passive_box_at(halfwidth):
                loads = itertools.product(*zip(nominal*(1-halfwidth), nominal*(1+halfwidth)))
                return max(np.ptp(G[j] @ np.asarray(load)) + 2*eps for load in loads)
            threshold_error = max(threshold_error, abs(passive_box_at(delta)-target),
                                  passive_box_at(delta-1e-6)-target, target-passive_box_at(delta+1e-6))
    check('passive_target_fractional_halfwidth_error', tolerance_error, 1e-11)
    check('passive_target_nominal_status_error', nominal_status_error, 0.)
    check('passive_target_exact_threshold_rad', threshold_error, 1e-10)
    mechanism = json.loads((result_dir/'workload_robust_mechanism.json').read_text(encoding='utf8'))
    cu, aln = G[indices['Cu']], G[indices['AlN_passive']]
    ratio = float(mechanism['passive_AlN_vs_Cu_crossover_PIC_EIC_ratio'])
    phi_cu, phi_aln = cu @ np.array([1., ratio]), aln @ np.array([1., ratio])
    coefficient = (aln[np.argmax(phi_aln)]-aln[np.argmin(phi_aln)]) - (cu[np.argmax(phi_cu)]-cu[np.argmin(phi_cu)])
    exact_local_ratio = -coefficient[0]/coefficient[1]
    check('selected_crossover_piecewise_linear_root_error', abs(exact_local_ratio-ratio), 1e-10)
    check('selected_crossover_phase_width_difference', abs(np.ptp(phi_aln)-np.ptp(phi_cu)), 1e-12)
    metrics['selected_crossover_ratio_independent'] = float(exact_local_ratio)
    lower_sign = np.ptp(aln @ np.array([1.,ratio-1e-5])) - np.ptp(cu @ np.array([1.,ratio-1e-5]))
    upper_sign = np.ptp(aln @ np.array([1.,ratio+1e-5])) - np.ptp(cu @ np.array([1.,ratio+1e-5]))
    check('selected_crossover_changes_sign', float(lower_sign*upper_sign >= 0), 0.)
    check('selected_crossover_outside_primary_box', float(upper[1]/lower[0] >= ratio), 0.)
    mechanism_error = 0.
    for s, source in enumerate(['EIC','PIC']):
        mechanism_error = max(mechanism_error, abs(np.ptp(aln[:,s])/np.ptp(cu[:,s]) - float(mechanism[source+'_passive_AlN_to_Cu_differential_sensitivity_ratio'])))
    for j, case_id in enumerate(d['case_id']):
        improvement = 100*(1-np.max(d['common_corner_certificates_rad'][j])/np.max(d['fixed_nominal_corner_certificates_rad'][j]))
        mechanism_error = max(mechanism_error, abs(improvement-float(mechanism['common_command_certificate_improvement_percent'][str(case_id)])))
    check('mechanism_exported_sensitivity_and_improvement', mechanism_error, 1e-10)
    relaxed_error, lower_bound_violation = 0., 0.
    relaxed_values, dual_certificates = {}, {}
    for j, case_id in enumerate(d['case_id']):
        phases = vertices @ G[j].T
        relaxation, _, relaxed_A, relaxed_rhs, solved = ordered_lp(
            phases, H_all[j], pmax, budget, bits, eta, eps,
            include_dac_margin=False, return_solver_result=True)
        dual = verify_lp_dual(solved, relaxed_A, relaxed_rhs, pmax)
        dual_certificates[str(case_id)] = dual
        for field in ['primal_feasibility_violation', 'dual_sign_violation',
                      'stationarity_max_absolute_residual', 'complementarity_max_absolute_residual',
                      'absolute_primal_dual_gap_rad']:
            check('relaxed_dual_'+str(case_id)+'_'+field, dual[field], 1e-8)
        check('relaxed_rational_bound_'+str(case_id)+'_above_primal',
              -dual['primal_minus_rational_lower_bound_rad'], 1e-9)
        check('relaxed_rational_bound_'+str(case_id)+'_gap',
              abs(dual['primal_minus_rational_lower_bound_rad']), 1e-8)
        relaxed_values[str(case_id)] = float(relaxation)
        relaxed_error = max(relaxed_error, abs(relaxation-float(d['continuous_relaxation_lower_bound_rad'][j])))
        for command in [np.zeros(H_all[j].shape[1]), d['commands_nominal_mW'][j], d['commands_common_mW'][j], *d['commands_adaptive_mW'][j]]:
            command_box_certificate = max(exact_certificate(phi, H_all[j], command, eta, eps)[0] for phi in phases)
            lower_bound_violation = max(lower_bound_violation, relaxation-command_box_certificate)
        for row in summary_rows:
            if row['case_id'] == case_id:
                if row['policy'] == 'adaptive_vertices_only':
                    if row['continuous_relaxation_lower_bound_rad'] != '':
                        raise AssertionError('Fixed-command lower bound cannot be labeled as an adaptive policy bound.')
                else:
                    relaxed_error = max(relaxed_error, abs(relaxation-float(row['continuous_relaxation_lower_bound_rad'])))
    check('relaxed_fixed_command_LP_optimum_error_rad', relaxed_error, 1e-10)
    check('relaxed_optimum_lower_than_all_tested_fixed_commands', lower_bound_violation, 1e-10)
    metrics['continuous_relaxation_lower_bounds_rad'] = relaxed_values
    metrics['relaxed_LP_dual_certificates'] = dual_certificates
    metrics['designs_with_0p25_rad_fixed_command_infeasibility_proven'] = [
        name for name, value in dual_certificates.items() if value['excludes_0p25_rad']]
    metrics['independent_relaxed_lower_bound_LPs'] = len(G)
    metrics.update(vertex_csv_rows_checked=len(vertex_rows), summary_csv_rows_checked=len(summary_rows), load_map_csv_rows_checked=len(load_map), source_csv_rows_checked=len(source_rows), passive_target_rows_checked=len(tolerance_rows), physical_loads_reprojected=int(projected.shape[0] * projected.shape[1]))
    metrics.update(independent_LPs=lp_count, evaluated_command_vertex_pairs=corner_count,
                   random_interior_loads=interior_count,
                   random_sensor_and_calibration_draws=perturbation_count,
                   explicit_adversarial_corners=corner_count,
                   deterministic_random_seed=20260925)
    output = dict(all_passed=all(checks.values()), checks=checks, metrics=metrics,
                  descriptions=descriptions, reconstructed_corner_metrics=all_rows,
                  scope=('Independent exported-data and optimization verification. '
                         'Continuous LP optimality is checked; no global DAC-integer optimality '
                         'or adaptive policy over interior workloads is claimed. The optical '
                         'efficiency metric assumes equal amplitudes and ideal lossless combining. '
                         'This is not experimental/full-vector model validation.'),
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [path, raw_path, coeff_path, Path(__file__)]})
    outpath = result_dir / 'workload_independent_verification.json'
    outpath.write_text(json.dumps(output, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(all_passed=output['all_passed'], metrics=metrics,
                          failed_checks=[k for k, passed in checks.items() if not passed],
                          output=str(outpath)), indent=2))
    if not output['all_passed']:
        raise AssertionError('Independent workload checks failed.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
