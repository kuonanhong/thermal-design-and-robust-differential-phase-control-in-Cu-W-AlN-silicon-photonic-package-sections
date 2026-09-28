#!/usr/bin/env python3
"""Independent 2-D full-vector WGMODES check of the reduced optical map.

SI inputs are converted to micrometres only at the Maxwell solver boundary.
The vendored, unmodified MIT source is used; see vendor/wgmodes/PROVENANCE.md.
This upgrades local optical validation, not the 2-D package model to 3-D.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,csv,hashlib,time,importlib,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent/'vendor/wgmodes'))
from modesolver import wgmodes
from optical_mode import PARAMETERS,slab_mode,modal_coefficients
from physical import FIELD_KEYS,optical_from_fields
from control_analysis import optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MOD=importlib.import_module('modesolver.wgmodes')
ORIGINAL_EIGS=MOD.eigs
SOLVE_RESIDUALS=[]
def checked_eigs(A,*args,**kwargs):
    kwargs['v0']=np.random.default_rng(9272026).normal(size=A.shape[0])
    val,vec=ORIGINAL_EIGS(A,*args,**kwargs)
    residual=max(np.linalg.norm(A@v-l*v)/np.linalg.norm(A@v) for l,v in zip(val,vec.T))
    SOLVE_RESIDUALS.append(float(residual))
    return val,vec
MOD.eigs=checked_eigs

def axis(core_um,step_um,pad_um,nominal_core_um):
    """Exactly aligned interfaces; fixed core-cell counts under perturbations."""
    n=int(round(nominal_core_um/step_um));inner=np.linspace(-core_um/2,core_um/2,n+1)
    distance=[0.];d=step_um
    while distance[-1]<pad_um:
        distance.append(min(pad_um,distance[-1]+d));d=min(d*1.10**(step_um/.01),.05*step_um/.01)
    ds=np.array(distance[1:]);return np.r_[-core_um/2-ds[::-1],inner,core_um/2+ds]

def solve(p=None,step_nm=5.,pad_um=1.2,core_n=None,epsxy=0.,reference=None,nmodes=2):
    p=dict(PARAMETERS if p is None else p);w=p['width_m']*1e6;h=p['height_m']*1e6
    x=axis(w,step_nm/1000,pad_um,.45);y=axis(h,step_nm/1000,pad_um,.22)
    xc=(x[1:]+x[:-1])/2;yc=(y[1:]+y[:-1])/2;X,Y=np.meshgrid(xc,yc)
    core=(abs(X)<w/2)&(abs(Y)<h/2);ns=np.repeat(p['n_si'],3) if core_n is None else np.array(core_n)
    eps=[np.where(core,n*n,p['n_clad']**2) for n in ns];xy=np.where(core,epsxy,0.)
    fields=wgmodes(p['wavelength_m']*1e6,2.36,nmodes,np.diff(x),np.diff(y),'0000',
        epsxx=eps[0],epsyy=eps[1],epszz=eps[2],epsxy=xy,epsyx=xy,solver='superlu')
    area=np.diff(y)[:,None]*np.diff(x)[None,:]
    ef=np.stack(fields[1:4],axis=0);norm=(area[None,:,:,None]*abs(ef)**2).sum(axis=(0,1,2))
    frac=(area[:,:,None]*abs(fields[1])**2).sum(axis=(0,1))/norm
    if reference is None:
        candidates=np.flatnonzero(frac>.5)
        if not len(candidates):raise RuntimeError('No quasi-TE mode found')
        idx=int(candidates[np.argmax(np.real(fields[0][candidates]))]);overlap=1.
    else:
        ref=reference['E'];scores=np.abs((area[None,:,:,None]*np.conj(ref[:,:,:,None])*ef).sum(axis=(0,1,2)))**2
        scores/=norm*np.sum(area[None,:,:]*abs(ref)**2);idx=int(np.argmax(scores));overlap=float(scores[idx])
        if frac[idx]<.5 or overlap<.99:raise RuntimeError('Mode tracking failed')
    if abs(np.imag(fields[0][idx]))>1e-8:raise RuntimeError('Unexpected lossy eigenvalue')
    return dict(neff=float(np.real(fields[0][idx])),Ex_fraction=float(frac[idx]),overlap=overlap,
        residual=SOLVE_RESIDUALS[-1],x_um=x,y_um=y,E=ef[:,:,:,idx],
        H=np.stack(fields[4:7],axis=0)[:,:,:,idx],shape=list(core.shape))

def derivatives(step_nm=5.,pad_um=1.2,relative_step=1e-4):
    p=dict(PARAMETERS);base=solve(p,step_nm,pad_um);sens={};overlaps=[]
    for key in ['n_si','n_clad','width_m','height_m']:
        delta=p[key]*relative_step;pp=dict(p);pm=dict(p);pp[key]+=delta;pm[key]-=delta
        a=solve(pp,step_nm,pad_um,reference=base);b=solve(pm,step_nm,pad_um,reference=base)
        sens[key]=(a['neff']-b['neff'])/(2*delta);overlaps.extend([a['overlap'],b['overlap']])
    for j,key in enumerate(['n_x','n_y','n_z']):
        delta=p['n_si']*relative_step;ns=np.repeat(p['n_si'],3);np_=ns.copy();nm=ns.copy();np_[j]+=delta;nm[j]-=delta
        a=solve(p,step_nm,pad_um,core_n=np_,reference=base);b=solve(p,step_nm,pad_um,core_n=nm,reference=base)
        sens[key]=(a['neff']-b['neff'])/(2*delta);overlaps.extend([a['overlap'],b['overlap']])
    delta=p['n_si']**2*relative_step
    a=solve(p,step_nm,pad_um,epsxy=delta,reference=base);b=solve(p,step_nm,pad_um,epsxy=-delta,reference=base)
    sens['eps_xy']=(a['neff']-b['neff'])/(2*delta)
    pe_matrix=np.full((3,3),p['p12']);np.fill_diagonal(pe_matrix,p['p11'])
    pecoeff=-.5*p['n_si']**3*np.array([sens['n_x'],sens['n_y'],sens['n_z']])@pe_matrix
    c=dict(modal_coefficients());c.update(model='full-vector 2D transverse-H finite difference, WGMODES',neff=base['neff'],
        dn_eff_d_n_si=sens['n_si'],dn_eff_d_n_clad=sens['n_clad'],
        dn_eff_d_n_x=sens['n_x'],dn_eff_d_n_y=sens['n_y'],dn_eff_d_n_z=sens['n_z'],dn_eff_d_eps_xy=sens['eps_xy'],
        dn_eff_d_width_per_m=sens['width_m'],dn_eff_d_height_per_m=sens['height_m'],
        dn_eff_dT_K=sens['n_si']*p['dn_si_dT_K']+sens['n_clad']*p['dn_clad_dT_K'],
        dn_eff_d_total_strain_x=sens['width_m']*p['width_m'],dn_eff_d_total_strain_y=sens['height_m']*p['height_m'],
        dn_eff_d_elastic_strain_x=pecoeff[0],dn_eff_d_elastic_strain_y=pecoeff[1],dn_eff_d_elastic_strain_z=pecoeff[2],
        Ex_fraction=base['Ex_fraction'],mesh_core_step_nm=step_nm,padding_um=pad_um,
        derivative_relative_step=relative_step,minimum_mode_overlap=min(overlaps),
        index_chain_rule_error=abs(sum(sens[k] for k in ['n_x','n_y','n_z'])-sens['n_si']),
        validation_scope='2D waveguide Maxwell eigenmode verification only; not full 3D package validation',
        EIM_order='not applicable')
    return c,base

def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)

def benchmarks(out):
    rows=[]
    # Uniform all-symmetric Hx, antisymmetric Hy has exact constant Hx solution.
    for step in [.05,.025]:
        n=3.476;eps=np.full((20,20),n*n)
        r=wgmodes(1.55,3.45,1,step,step,'SSSS',eps=eps,solver='superlu')
        rows.append(dict(test='uniform',polarization='Hx',step_nm=step*1000,exact_neff=n,
                         numerical_neff=float(np.real(np.atleast_1d(r[0])[0])),abs_error=abs(np.atleast_1d(r[0])[0]-n),residual=SOLVE_RESIDUALS[-1]))
    for pol,bnd in [('TE','00AA'),('TM','00SS')]:
        exact=slab_mode(3.476,1.444,220e-9,1550e-9,pol)['neff']
        for step in [10.,5.,2.5,1.25]:
            x=np.linspace(-.1,.1,5);y=axis(.22,step/1000,1.8,.22);yc=(y[1:]+y[:-1])/2
            eps=np.tile(np.where(abs(yc)<.11,3.476**2,1.444**2)[:,None],(1,4))
            rr=wgmodes(1.55,exact,1,np.diff(x),np.diff(y),bnd,eps=eps,solver='superlu')
            value=float(np.real(np.atleast_1d(rr[0])[0]));rows.append(dict(test='slab',polarization=pol,step_nm=step,exact_neff=exact,numerical_neff=value,abs_error=abs(value-exact),residual=SOLVE_RESIDUALS[-1]))
    csvwrite(out/'vector_benchmarks.csv',rows);return rows

def reproject(c,out):
    sources=[np.load(ROOT/'results/physical_inputs.npz'),np.load(ROOT/'results/placement_physical_inputs.npz')]
    rows=[];phis=[];hs=[];ids=[];pqs=[];commands=[]
    for source in sources:
        for i in range(len(source['case_id'])):
            design=str(source['design'][i]);R=float(source['Rb_m2K_W'][i])
            if source is sources[0] and not(source['nx'][i]==120 and source['k_AlN_W_mK'][i]==170 and (R==1e-8 if 'AlN' in design else R==0)):continue
            bg={k:source['background_'+k][i] for k in FIELD_KEYS};hh={k:source['heater_'+k][i] for k in FIELD_KEYS}
            phi,*_=optical_from_fields(bg,c);H,*_=optical_from_fields(hh,c);H=H.T
            pmax=20.;bits=8;reserve=9*pmax/(2**bits-1)/2
            r=optimize(phi,H,pmax,100-reserve,robust=True,bits=bits,eta=.02,eps_rad=.002)
            cid=str(source['case_id'][i]);ids.append(cid);phis.append(phi);hs.append(H);pqs.append(r['p_quantized_mW'])
            rows.append(dict(case_id=cid,design=design,mean_phi_rad=float(phi.mean()),passive_spread_rad=float(np.ptp(phi)),
                H_diagonal_mean_rad_per_mW=float(np.diag(H).mean()),robust_Cpost_rad=r['posterior_certificate_rad'],
                robust_power_mW=r['quantized_sum_mW'],mean_temperature_rise_K=float(bg['temperature_K'].mean())))
            for j,(p,pq) in enumerate(zip(r['p_mW'],r['p_quantized_mW'])):commands.append(dict(case_id=cid,heater=j,continuous_mW=p,quantized_mW=pq))
    csvwrite(out/'vector_reprojected_cases.csv',rows);csvwrite(out/'vector_control_commands.csv',commands)
    np.savez_compressed(out/'vector_reprojected_inputs.npz',case_id=ids,phi_rad=phis,H_rad_per_mW=hs,p_quantized_mW=pqs)
    return rows

def run(out):
    start=time.time();out.mkdir(parents=True,exist_ok=True);bench=benchmarks(out);mesh=[]
    for step in [10.,5.,2.5,1.25]:
        r=solve(step_nm=step,pad_um=1.2);mesh.append(dict(step_nm=step,padding_um=1.2,neff=r['neff'],Ex_fraction=r['Ex_fraction'],Nx=r['shape'][1],Ny=r['shape'][0],residual=r['residual']))
        print('mesh',mesh[-1],flush=True)
    csvwrite(out/'vector_mesh_convergence.csv',mesh);pads=[]
    for pad in [.4,.8,1.2,1.6]:
        r=solve(step_nm=5.,pad_um=pad);pads.append(dict(padding_um=pad,step_nm=5.,neff=r['neff'],Ex_fraction=r['Ex_fraction']))
    csvwrite(out/'vector_padding_convergence.csv',pads)
    cr=[];base=None
    for step,dr in [(10.,1e-4),(5.,1e-4),(2.5,1e-4),(2.5,5e-5)]:
        print('derivatives',step,dr,flush=True);c,base=derivatives(step_nm=step,pad_um=1.2,relative_step=dr);cr.append(c)
        (out/f'vector_coefficients_{step:g}nm_{dr:g}.json').write_text(json.dumps(c,indent=2)+'\n')
    c=cr[2];(out/'vector_coefficients.json').write_text(json.dumps(c,indent=2)+'\n')
    csvwrite(out/'vector_derivative_convergence.csv',cr)
    np.savez_compressed(out/'vector_nominal_fields.npz',**{k:v for k,v in base.items() if k not in ['shape']})
    rows=reproject(c,out);cu=next(r for r in rows if r['case_id'].startswith('Cu_'));best=min(rows,key=lambda r:r['passive_spread_rad']);aln=next(r for r in rows if r['case_id']=='AlN_L320_R600')
    derivkeys=['dn_eff_dT_K','dn_eff_d_total_strain_x','dn_eff_d_total_strain_y','dn_eff_d_elastic_strain_x','dn_eff_d_elastic_strain_y','dn_eff_d_elastic_strain_z']
    summary=dict(solver='WGMODES 3.0.2',commit='6f54c5e82a8de9d7c78a1c86eef2d658b45bc443',
        scope='2D full-vector local optical verification; package remains 2D plane strain and extruded',
        coefficients=c,uniform_max_error=max(r['abs_error'] for r in bench if r['test']=='uniform'),
        finest_slab_max_error=max(r['abs_error'] for r in bench if r['test']=='slab' and r['step_nm']==1.25),
        mesh_2p5_to_1p25_absolute_neff_change=abs(mesh[-1]['neff']-mesh[-2]['neff']),
        mesh_2p5_to_1p25_relative_neff_change=abs(mesh[-1]['neff']/mesh[-2]['neff']-1),
        padding_1p2_to_1p6_absolute_neff_change=abs(pads[-1]['neff']-pads[-2]['neff']),
        derivative_5_to_2p5_relative_changes={k:abs(cr[2][k]/cr[1][k]-1) for k in derivkeys},
        derivative_step_halving_relative_changes={k:abs(cr[3][k]/cr[2][k]-1) for k in derivkeys},
        maximum_eigenpair_residual=max(SOLVE_RESIDUALS),Cu=cu,best_passive=best,AlN_320_600=aln,
        AlN_320_600_reduction_percent=100*(1-aln['passive_spread_rad']/cu['passive_spread_rad']),
        best_controlled=min(rows,key=lambda r:r['robust_Cpost_rad']),
        elapsed_seconds=time.time()-start,eigenproblems=len(SOLVE_RESIDUALS),
        caveats=['Finite mesh and computational window; no 3D package or experimental validation.',
                'Core diagonal tensor photoelasticity included; symmetric-core xy coefficient tested zero to numerical precision; cladding photoelasticity excluded.',
                'Legacy photoelastic coefficients remain assumptions; gain uncertainty 2% does not encompass optical-model form error.',
                'Both background phase and heater response reprojected consistently; 22 reported cases contain two repeated layouts.'])
    summary['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'results/physical_inputs.npz',ROOT/'results/placement_physical_inputs.npz']}
    (out/'vector_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(out,mesh,bench,base,rows)
    print(json.dumps(summary,indent=2))

def plot(out,mesh,bench,base,rows):
    plt.rcParams.update({'font.size':11,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10})
    fig,axs=plt.subplots(2,2,figsize=(10,7.3),layout='constrained')
    x=base['x_um'];y=base['y_um'];E=base['E'];F=np.sum(abs(E)**2,axis=0);F/=F.max()
    im=axs[0,0].pcolormesh(x,y,F,cmap='magma',shading='flat',norm=matplotlib.colors.LogNorm(vmin=1e-4,vmax=1));axs[0,0].set(xlim=(-.65,.65),ylim=(-.45,.45),xlabel='x (µm)',ylabel='y (µm)',title='(a) Full-vector quasi-TE |E|²')
    axs[0,0].plot([-.225,.225,.225,-.225,-.225],[-.11,-.11,.11,.11,-.11],color='cyan',lw=1);fig.colorbar(im,ax=axs[0,0],label='Normalized intensity (log scale)')
    axs[0,1].plot([r['step_nm'] for r in mesh],[r['neff'] for r in mesh],'o-',label='Full vector')
    axs[0,1].axhline(modal_coefficients()['neff'],c='#c05a22',ls='--',label='EIM (vertical first)')
    axs[0,1].set(xlabel='Core mesh spacing (nm)',ylabel='Effective index',title='(b) Optical model and mesh');axs[0,1].legend()
    for pol in ['TE','TM']:
        rr=[r for r in bench if r['test']=='slab' and r['polarization']==pol];axs[1,0].loglog([r['step_nm'] for r in rr],[r['abs_error'] for r in rr],'o-',label=pol)
    axs[1,0].set_xticks([1.25,2.5,5.,10.],['1.25','2.5','5','10']);axs[1,0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axs[1,0].set(xlabel='Core mesh spacing (nm)',ylabel='|n numeric − n analytic|',title='(c) Exact-slab verification');axs[1,0].legend()
    ids=['Cu_R0e+00_k170_n120','graded_W_R0e+00_k170_n120','AlN_R1e-08_k170_n120','AlN_L320_R600']
    vv=[next(r for r in rows if r['case_id']==i) for i in ids];xx=np.arange(4)
    axs[1,1].bar(xx-.18,[r['passive_spread_rad'] for r in vv],.36,label='Passive phase spread')
    axs[1,1].bar(xx+.18,[r['robust_Cpost_rad'] for r in vv],.36,label='Robust Cpost')
    axs[1,1].set(xticks=xx,xticklabels=['Cu','W','AlN\n320/880','AlN\n320/600'],ylabel='Phase difference (rad)',title='(d) Vector-reprojected comparison');axs[1,1].legend(fontsize=9)
    for ax in axs.flat:ax.grid(alpha=.18)
    for ext in ['png','pdf']:
        tmp=out/f'vector_verification.tmp.{ext}';fig.savefig(tmp,dpi=220);tmp.replace(out/f'vector_verification.{ext}')
    plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/vector');args=ap.parse_args();run(args.out)
