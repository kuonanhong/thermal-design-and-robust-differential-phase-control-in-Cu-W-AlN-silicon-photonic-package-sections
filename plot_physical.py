#!/usr/bin/env python3
"""Publication figures for the executed conservative thermomechanical model."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch,Rectangle
from core import geometry,CHANNEL_X,CHANNEL_Y,TSV_X

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12.5,'axes.titlesize':12.5,'axes.labelsize':12.5,'xtick.labelsize':11.8,'ytick.labelsize':11.8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
COLORS=['#1f67a8','#8956a3','#158c85','#d97629']
LABELS={'Cu':'Cu','graded_W':'Graded W','AlN':'AlN','graded_W_AlN':'Graded W + AlN'}

def save(fig,name):
 tmp=OUT/(name+'.tmp.pdf');fig.savefig(tmp,format='pdf',bbox_inches='tight');assert tmp.stat().st_size>100;tmp.replace(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight');plt.close(fig)

def main():
 d=np.load(ROOT/'results/control_inputs.npz');rows=list(csv.DictReader((ROOT/'results/physical_summary.csv').open()));od=list(csv.DictReader((ROOT/'results/optical_physical_summary.csv').open()))
 def idx(design,R=0.,nx=120):
  return np.where((d['design']==design)&np.isclose(d['Rb_m2K_W'],R,atol=1e-14)&(d['nx']==nx)&(d['k_AlN_W_mK']==170))[0][0]
 ids=[idx('Cu'),idx('graded_W'),idx('AlN',1e-8),idx('graded_W_AlN',1e-8)]
 g=geometry(120,60,design='graded_W_AlN');names=['encapsulant','substrate','Si','oxide','Cu','W','Cu/W','AlN'];pal=['#f0f2f4','#a4afb6','#c5deef','#ecdcb5','#dc9235','#7c8798','#b1959b','#63b3a5'];a=np.zeros_like(g['q'])
 for j,n in enumerate(names):a[g['labels']==n]=j
 fig,ax=plt.subplots(figsize=(8.2,5.2));ax.imshow(a,origin='lower',extent=[0,1200,0,600],cmap=ListedColormap(pal),vmin=0,vmax=len(names)-1,interpolation='nearest',aspect='equal')
 ax.plot(CHANNEL_X*1e6,np.full(9,350),'o',color='#b51f36',ms=4,label='Waveguide sampling points')
 for x in CHANNEL_X*1e6:ax.add_patch(Rectangle((x-10,380),20,20,facecolor='#262626',edgecolor='white',linewidth=.3))
 ax.annotate('Gaussian line heaters\n(1 W/m basis loads)',xy=(690,390),xytext=(925,470),arrowprops=dict(arrowstyle='->'),fontsize=9.5)
 ax.annotate('EIC: 250 W/m',xy=(540,470),ha='center',fontsize=10.5)
 ax.annotate('PIC: 25 W/m',xy=(540,330),ha='center',fontsize=9.5)
 ax.set(xlabel='x (µm)',ylabel='y (µm)',title='Invariant-z package cross-section; optical length 1 mm')
 ax.text(.99,.99,'z propagation: out of page\nPlane strain: εzz = 0',transform=ax.transAxes,ha='right',va='top',fontsize=9.5,bbox=dict(fc='white',ec='none',alpha=.85))
 handles=[Patch(facecolor=c,label=n) for n,c in zip(names,pal)];handles.append(plt.Line2D([],[],marker='o',color='#b51f36',linestyle='',label='Waveguide samples'))
 ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=5,frameon=False,fontsize=9.5);fig.tight_layout();save(fig,'physical_geometry')
 # Main passive result: spatial deviations and actuator efficiency, not rawmeanphase=error.
 fig,axs=plt.subplots(1,3,figsize=(10,3.8))
 for k,i in enumerate(ids):
  label=LABELS[str(d['design'][i])]
  axs[0].plot(CHANNEL_X*1e6,d['temperature_background_K'][i]+25,'o-',ms=3,color=COLORS[k],label=label)
  axs[1].plot(CHANNEL_X*1e6,d['phi_rad'][i]-d['phi_rad'][i].mean(),'o-',ms=3,color=COLORS[k])
  axs[2].plot(np.arange(1,10),np.diag(d['H_rad_per_mW'][i]),'o-',ms=3,color=COLORS[k])
 axs[0].set(xlabel='Channel x (µm)',ylabel='Temperature (°C)',title='(a) Channel temperature')
 axs[1].set(xlabel='Channel x (µm)',ylabel='Phase relative to mean (rad)',title='(b) Differential phase')
 axs[2].set(xlabel='Heater index',ylabel='Self-response (rad/mW)',title='(c) Heater response',xticks=[1,3,5,7,9])
 axs[0].legend(frameon=False,fontsize=10.5);fig.tight_layout();save(fig,'physical_passive_tradeoff')
 # Contact resistance, no log zero fake: plotR0as separate horizontal reference.
 fig,axs=plt.subplots(1,3,figsize=(10,3.8))
 for design,color in [('AlN',COLORS[2]),('graded_W_AlN',COLORS[3])]:
  sel=(d['design']==design)&(d['nx']==120)&(d['kind']=='interface')&(d['Rb_m2K_W']>0)
  R=d['Rb_m2K_W'][sel];ph=d['phi_rad'][sel];H=d['H_rad_per_mW'][sel];temp=d['temperature_background_K'][sel]
  axs[0].semilogx(R,temp.mean(axis=1)+25,'o-',color=color,label=LABELS[design]);axs[1].semilogx(R,np.ptp(ph,axis=1),'o-',color=color);axs[2].semilogx(R,np.diagonal(H,axis1=1,axis2=2).mean(axis=1),'o-',color=color)
 cu=ids[0]
 for ax,y in zip(axs,[d['temperature_background_K'][cu].mean()+25,np.ptp(d['phi_rad'][cu]),np.diag(d['H_rad_per_mW'][cu]).mean()]):
  ax.axhline(y,color=COLORS[0],ls='--',label='Cu reference');ax.set_xlabel(r'$R_\mathrm{K}$ (m$^2$ K/W)');ax.grid(alpha=.15)
 axs[0].set(ylabel='Channel mean (°C)',title='(a) Mean temperature');axs[1].set(ylabel='max φ − min φ (rad)',title='(b) Phase spread');axs[2].set(ylabel=r'Mean $H_{ii}$ (rad/mW)',title='(c) Heater efficiency');axs[0].legend(frameon=False,fontsize=10.5);fig.tight_layout();save(fig,'physical_contact_sweep')
 # Same-rangephysicalheaterH matrices.
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.45));hs=[d['H_rad_per_mW'][ids[j]] for j in [0,3]];vmin=min(z.min() for z in hs);vmax=max(z.max() for z in hs)
 for ax,H,lab in zip(axs,hs,['Cu','Graded W + AlN']):
  im=ax.imshow(H,origin='lower',cmap='viridis',vmin=vmin,vmax=vmax,extent=[.5,9.5,.5,9.5]);ax.set(xlabel='Heater index',ylabel='Observation channel',title=lab,xticks=[1,3,5,7,9],yticks=[1,3,5,7,9])
 axs[1].set_ylabel('');fig.colorbar(im,ax=axs,shrink=.82,label='H (rad/mW)',pad=.06);save(fig,'physical_heater_matrix')
 # Mesh curves quantify phase andresponseconvergence; avoidmaximumstressclaim.
 fig,axs=plt.subplots(1,3,figsize=(10,3.8))
 for k,design in enumerate(LABELS):
  js=[idx(design,1e-8 if 'AlN' in design else 0.,nx) for nx in [60,120,180,240]]
  nx=d['nx'][js]
  axs[0].plot(1200/nx,d['temperature_background_K'][js].mean(axis=1)+25,'o-',color=COLORS[k],label=LABELS[design]);axs[1].plot(1200/nx,np.ptp(d['phi_rad'][js],axis=1),'o-',color=COLORS[k]);axs[2].plot(1200/nx,np.diagonal(d['H_rad_per_mW'][js],axis1=1,axis2=2).mean(axis=1),'o-',color=COLORS[k])
 for ax in axs:ax.set_xlabel('Cell size (µm)');ax.invert_xaxis();ax.grid(alpha=.15)
 axs[0].set(ylabel='Channel mean (°C)',title='(a) Channel temperature');axs[1].set(ylabel='Phase spread (rad)',title='(b) Phase spread');axs[2].set(ylabel=r'Mean $H_{ii}$ (rad/mW)',title='(c) Heater response');axs[0].legend(frameon=False,fontsize=10.5);fig.tight_layout();save(fig,'physical_mesh_convergence')
 # Physical fields samecolorlimits.
 cases=[str(d['case_id'][ids[j]]) for j in [0,3]];ff=[np.load(ROOT/'results'/f'fields_{z}.npz') for z in cases]
 fig,axs=plt.subplots(2,2,figsize=(9.4,6.2),layout='constrained');tempmin=25;tempmax=max(z['temperature_C'].max() for z in ff);smax=max(np.quantile(z['stress_von_mises_Pa']/1e6,.99) for z in ff)
 for j,(f,lab) in enumerate(zip(ff,['Cu','Graded W + AlN'])):
  it=axs[0,j].imshow(f['temperature_C'],origin='lower',extent=[0,1200,0,600],cmap='inferno',vmin=tempmin,vmax=tempmax,aspect='equal');axs[0,j].set_title(lab)
  iss=axs[1,j].imshow(f['stress_von_mises_Pa']/1e6,origin='lower',extent=[0,1200,0,600],cmap='magma',vmin=0,vmax=smax,aspect='equal')
  for ax in axs[:,j]:ax.set(xlabel='x (µm)',ylabel='y (µm)');ax.plot(CHANNEL_X*1e6,np.full(9,350),'c.',ms=2)
 fig.colorbar(it,ax=axs[0,:],shrink=.85,label='Temperature (°C)',pad=.02);fig.colorbar(iss,ax=axs[1,:],shrink=.85,label='von Mises (MPa); clipped at p99',pad=.02);save(fig,'physical_fields')

if __name__=='__main__':main()
