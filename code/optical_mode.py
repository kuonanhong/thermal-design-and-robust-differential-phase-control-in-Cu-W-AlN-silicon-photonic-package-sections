#!/usr/bin/env python3
"""Reduced quasi-TE optical response, exact slab checks, and reproducible data.

This is a two-step effective-index method (EIM), NOT a full-vector 2D Maxwell
mode solver. Coordinates: x width/[100], y thickness/[010], z propagation/[001].
The package displacement is invariant in z, hence total epsilon_zz = 0; the
optical phase has no longitudinal path-length contribution. The affine local
macrostrain assumption sets delta w=w*epsilon_xx and delta h=h*epsilon_yy.

Bulk thermo-optic coefficients describe stress-free material. Photoelasticity
therefore acts on elastic strain epsilon_el=epsilon_total-alpha*deltaT I.
The dominant Ex approximation retains delta B_xx=p11*e_xx+p12*(e_yy+e_zz);
other vector-field components, shear, cladding photoelasticity and anisotropic
mechanical submodel are outside this reduced model. Nominal p coefficients are
literature-inspired assumptions, not a wavelength-specific measured calibration.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARAMETERS = dict(n_si=3.476, n_clad=1.444, width_m=450e-9, height_m=220e-9,
 wavelength_m=1550e-9, path_length_m=1e-3, dn_si_dT_K=1.86e-4,
 dn_clad_dT_K=1.0e-5, alpha_si_per_K=2.6e-6,
 p11=-.094, p12=.017, p44=-.051)

def slab_mode(n_core:float, n_clad:float, thickness_m:float,
              wavelength_m:float, polarization:str='TE') -> dict:
    """Fundamental even mode of a symmetric isotropic three-layer slab.

    u tan(u) = q sqrt(V^2-u^2), q=1 (TE), n_core^2/n_clad^2 (TM).
    u=k_transverse*d/2, V=k0*d/2*sqrt(n_core^2-n_clad^2).
    """
    if not n_core > n_clad > 0: raise ValueError('Need n_core > n_clad > 0')
    if polarization not in ('TE','TM'): raise ValueError('Choose TE or TM')
    k0=2*np.pi/wavelength_m
    V=k0*thickness_m*.5*np.sqrt(n_core*n_core-n_clad*n_clad)
    q=1. if polarization=='TE' else n_core*n_core/(n_clad*n_clad)
    fn=lambda u:u*np.tan(u)-q*np.sqrt(max(0.,V*V-u*u))
    u=brentq(fn,1e-14,min(V,np.pi/2)-1e-14,xtol=1e-14,rtol=1e-14)
    neff=np.sqrt(n_core*n_core-(2*u/(k0*thickness_m))**2)
    return dict(neff=float(neff),u=float(u),V=float(V),q=float(q),
                normalized_dispersion_residual=float(abs(fn(u))/max(1.,V)))

def effective_index(p:dict|None=None, order:str='vertical_first')->float:
    p=PARAMETERS if p is None else p
    if order=='vertical_first':
        nv=slab_mode(p['n_si'],p['n_clad'],p['height_m'],p['wavelength_m'],'TE')['neff']
        return slab_mode(nv,p['n_clad'],p['width_m'],p['wavelength_m'],'TM')['neff']
    if order=='horizontal_first':
        nh=slab_mode(p['n_si'],p['n_clad'],p['width_m'],p['wavelength_m'],'TM')['neff']
        return slab_mode(nh,p['n_clad'],p['height_m'],p['wavelength_m'],'TE')['neff']
    raise ValueError(order)

def modal_coefficients(p:dict|None=None, relative_step:float=1e-5, order:str='vertical_first')->dict:
    p=dict(PARAMETERS if p is None else p)
    sens={}
    for k in ('n_si','n_clad','width_m','height_m'):
        dp=relative_step*p[k];pp=dict(p);pm=dict(p);pp[k]+=dp;pm[k]-=dp
        sens[k]=(effective_index(pp,order=order)-effective_index(pm,order=order))/(2*dp)
    pe=-.5*p['n_si']**3*sens['n_si']
    result=dict(model='quasi-TE two-step effective-index approximation; dominant Ex photoelasticity',
       coordinates='x=[100] width; y=[010] thickness; z=[001] propagation',
       **p,neff=effective_index(p,order=order),EIM_order=order,
       dn_eff_d_n_si=sens['n_si'],dn_eff_d_n_clad=sens['n_clad'],
       dn_eff_d_width_per_m=sens['width_m'],dn_eff_d_height_per_m=sens['height_m'],
       dn_eff_dT_K=sens['n_si']*p['dn_si_dT_K']+sens['n_clad']*p['dn_clad_dT_K'],
       dn_eff_d_elastic_strain_x=pe*p['p11'],
       dn_eff_d_elastic_strain_y=pe*p['p12'],dn_eff_d_elastic_strain_z=pe*p['p12'],
       dn_eff_d_total_strain_x=sens['width_m']*p['width_m'],
       dn_eff_d_total_strain_y=sens['height_m']*p['height_m'],
       longitudinal_path_strain_coefficient=0.,
       derivative_relative_step=relative_step,
       nominal_photoelastic_note='p values are legacy nominal assumptions; wavelength/crystal calibration is not established',
       validation_scope='Analytical/numerical slab verification and derivative checks, not full-vector rectangular-waveguide validation')
    return result

def phase_components(deltaT,epsilon_total_x,epsilon_total_y,coefficients=None,pe_scale=1.):
    c=modal_coefficients() if coefficients is None else coefficients
    dt=np.asarray(deltaT);ex=np.asarray(epsilon_total_x);ey=np.asarray(epsilon_total_y)
    eex=ex-c['alpha_si_per_K']*dt;eey=ey-c['alpha_si_per_K']*dt
    eez=-c['alpha_si_per_K']*dt
    thermal=c['dn_eff_dT_K']*dt
    photo=pe_scale*(c['dn_eff_d_elastic_strain_x']*eex+c['dn_eff_d_elastic_strain_y']*eey+c['dn_eff_d_elastic_strain_z']*eez)
    boundary=c['dn_eff_d_total_strain_x']*ex+c['dn_eff_d_total_strain_y']*ey
    factor=2*np.pi*c['path_length_m']/c['wavelength_m']
    return dict(thermal_rad=factor*thermal,photoelastic_rad=factor*photo,
       cross_section_rad=factor*boundary,total_rad=factor*(thermal+photo+boundary),
       elastic_strain_x=eex,elastic_strain_y=eey,elastic_strain_z=eez)

def slab_fd(n_core,n_clad,thickness_m,wavelength_m,polarization,spacing_m,
            domain_m=4e-6):
    """Independent cell-centred self-adjoint finite-difference slab eigenproblem.

    TE: E''+k0^2*n^2 E=beta^2 E.
    TM: (n^-2 H')'+k0^2 H=beta^2*n^-2 H.
    Dirichlet truncation uses zero ghost cell fields, >1.8 um from core.
    TM face n^-2 uses harmonic transmissibility; generalized diagonal mass is
    removed with B^-1/2. Geometries are exactly mesh-aligned in supplied checks.
    """
    N=int(round(domain_m/spacing_m));dx=domain_m/N
    x=(np.arange(N)+.5)*dx-domain_m*.5
    n=np.where(abs(x)<thickness_m/2-1e-16,n_core,n_clad)
    k0=2*np.pi/wavelength_m
    if polarization=='TE':
        diagonal=-2./dx**2+k0*k0*n*n;off=np.full(N-1,1./dx**2)
    else:
        c=1/n**2;faces=2*c[:-1]*c[1:]/(c[:-1]+c[1:]);diagflux=np.r_[c[0],faces]+np.r_[faces,c[-1]]
        diagonal=(-diagflux/dx**2+k0*k0)/c
        off=faces/dx**2/np.sqrt(c[:-1]*c[1:])
    val=eigh_tridiagonal(diagonal,off,eigvals_only=True,select='i',select_range=(N-1,N-1))[0]
    return float(np.sqrt(val)/k0)

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(out:Path):
    out.mkdir(parents=True,exist_ok=True);c=modal_coefficients();p=PARAMETERS
    (out/'modal_coefficients.json').write_text(json.dumps(c,indent=2)+'\n')
    Path(__file__).with_name('modal_coefficients.json').write_text(json.dumps(c,indent=2)+'\n')
    alt=modal_coefficients(order='horizontal_first')
    (out/'modal_coefficients_horizontal_first.json').write_text(json.dumps(alt,indent=2)+'\n')
    Path(__file__).with_name('modal_coefficients_horizontal_first.json').write_text(json.dumps(alt,indent=2)+'\n')
    rows=[]
    for pol in ('TE','TM'):
      analytic=slab_mode(p['n_si'],p['n_clad'],p['height_m'],p['wavelength_m'],pol)
      for dx in (10e-9,5e-9,2.5e-9,1.25e-9):
        fd=slab_fd(p['n_si'],p['n_clad'],p['height_m'],p['wavelength_m'],pol,dx)
        rows.append(dict(polarization=pol,spacing_nm=dx*1e9,n_exact=analytic['neff'],n_fd=fd,absolute_error=abs(fd-analytic['neff']),relative_error=abs(fd/analytic['neff']-1)))
    write_csv(out/'modal_slab_verification.csv',rows)
    deriv=[]
    for ds in (1e-3,1e-4,1e-5,1e-6):
      cc=modal_coefficients(relative_step=ds)
      deriv.append(dict(relative_step=ds,**{k:v for k,v in cc.items() if k.startswith('dn_eff')}))
    write_csv(out/'modal_derivative_convergence.csv',deriv)
    widths=np.arange(350.,601.,25.);heights=np.arange(180.,281.,10.)
    sweep=[]
    for w in widths:
      for h in heights:
        pp={**p,'width_m':w*1e-9,'height_m':h*1e-9};cc=modal_coefficients(pp)
        sweep.append(dict(width_nm=w,height_nm=h,neff=cc['neff'],dn_eff_dT_K=cc['dn_eff_dT_K'],width_sensitivity_per_nm=cc['dn_eff_d_width_per_m']*1e-9,height_sensitivity_per_nm=cc['dn_eff_d_height_per_m']*1e-9))
    write_csv(out/'modal_dimension_sweep.csv',sweep)
    # A nonlinear direct EIM recalculation checks first-order Taylor expansion.
    rng=np.random.default_rng(260925);lin=[]
    for amplitude in (1e-5,1e-4,1e-3):
      for j in range(100):
        delta=rng.uniform(-amplitude,amplitude,4);pp=dict(p)
        for k,d in zip(('n_si','n_clad','width_m','height_m'),delta):pp[k]=p[k]*(1+d)
        exact=effective_index(pp)-c['neff'];linear=sum(c['dn_eff_d_'+k if k in ('n_si','n_clad') else 'dn_eff_d_'+k.replace('_m','')+'_per_m']*p[k]*d for k,d in zip(('n_si','n_clad','width_m','height_m'),delta))
        lin.append(dict(relative_perturbation_bound=amplitude,sample=j,exact_delta_neff=exact,linear_delta_neff=linear,absolute_remainder=abs(exact-linear)))
    write_csv(out/'modal_linearization_check.csv',lin)
    # Independent stress-optic form verifies tensor signs and plane-strain zz.
    rng_check=np.random.default_rng(417);tensor_errors=[]
    E=130e9;nu=.28;lam=E*nu/((1+nu)*(1-2*nu));mu=E/(2*(1+nu))
    C1=p['n_si']**3/(2*E)*(p['p11']-2*nu*p['p12'])
    C2=p['n_si']**3/(2*E)*(-nu*p['p11']+(1-nu)*p['p12'])
    for _ in range(100):
        dt=float(rng_check.uniform(0,40));ex,ey=rng_check.uniform(-1e-3,1e-3,2)
        ee=np.array([ex,ey,0.])-p['alpha_si_per_K']*dt
        sigma=2*mu*ee+lam*ee.sum()
        stress_form=c['dn_eff_d_n_si']*(-C1*sigma[0]-C2*(sigma[1]+sigma[2]))
        strain_form=phase_components(dt,ex,ey,c)['photoelastic_rad']*p['wavelength_m']/(2*np.pi*p['path_length_m'])
        tensor_errors.append(abs(stress_form-strain_form))
    summary=dict(coefficients=c,photoelastic_tensor_crosscheck_max_abs_neff_error=max(tensor_errors),slab_verification_max_finest_relative_error=max(r['relative_error'] for r in rows if r['spacing_nm']==1.25),
      alternate_EIM_order_neff=effective_index(order='horizontal_first'),
      alternate_EIM_order_note='Method-order spread is a model-form warning, not an error bound or full-vector validation.',
      slab_analytical_max_residual=max(slab_mode(p['n_si'],p['n_clad'],p['height_m'],p['wavelength_m'],pol)['normalized_dispersion_residual'] for pol in ('TE','TM')),
      linearization_max_remainder_by_bound={str(a):max(r['absolute_remainder'] for r in lin if r['relative_perturbation_bound']==a) for a in (1e-5,1e-4,1e-3)},
      all_slab_mesh_errors_decrease=all(all(b['absolute_error']<a['absolute_error'] for a,b in zip([r for r in rows if r['polarization']==pol][:-1],[r for r in rows if r['polarization']==pol][1:])) for pol in ('TE','TM')))
    (out/'modal_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10})
    fig,axs=plt.subplots(1,3,figsize=(10,3.8),layout='constrained')
    for pol in ('TE','TM'):
      rr=[r for r in rows if r['polarization']==pol]
      axs[0].loglog([r['spacing_nm'] for r in rr],[r['absolute_error'] for r in rr],'o-',label=pol)
    axs[0].set(xlabel='Slab grid spacing (nm)',ylabel='|n FD − n exact|',title='(a) Slab check');axs[0].legend()
    axs[0].set_xticks([1.25,2.5,5.,10.],['1.25','2.5','5','10']);axs[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for h in (180.,220.,260.):
      rr=[r for r in sweep if r['height_nm']==h]
      axs[1].plot([r['width_nm'] for r in rr],[r['neff'] for r in rr],label=f'h={h:.0f} nm')
    axs[1].set(xlabel='Width (nm)',ylabel='EIM effective index',title='(b) EIM index');axs[1].legend(fontsize=10)
    rr=[r for r in sweep if r['height_nm']==220.]
    axs[2].plot([r['width_nm'] for r in rr],[r['dn_eff_dT_K']*1e4 for r in rr],color='#B34229')
    axs[2].set(xlabel='Width (nm)',ylabel='Thermal response (10⁻⁴ K⁻¹)',title='(c) Thermal response')
    for ax in axs:ax.grid(alpha=.2)
    # Atomically replace final figures so simultaneous document builds never see a partial PDF.
    tmp_pdf=out/'modal_verification.tmp.pdf';tmp_png=out/'modal_verification.tmp.png'
    fig.savefig(tmp_pdf);fig.savefig(tmp_png,dpi=200);plt.close(fig)
    if tmp_pdf.stat().st_size<1000:raise RuntimeError('Unexpectedly short modal PDF')
    tmp_pdf.replace(out/'modal_verification.pdf');tmp_png.replace(out/'modal_verification.png')
    assert summary['photoelastic_tensor_crosscheck_max_abs_neff_error']<1e-15
    assert summary['all_slab_mesh_errors_decrease']
    assert summary['slab_verification_max_finest_relative_error']<1e-4
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'modal');args=ap.parse_args();run(args.out)
