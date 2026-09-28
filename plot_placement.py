#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12.5,'axes.titlesize':12.5,'axes.labelsize':12.5,'xtick.labelsize':11.8,'ytick.labelsize':11.8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def main():
 d=np.load(ROOT/'results/placement_inputs.npz');base=np.load(ROOT/'results/control_inputs.npz');cu=np.where((base['design']=='Cu')&(base['nx']==120))[0][0]
 fig,axs=plt.subplots(1,2,figsize=(9.6,4.4));ax=axs[0]
 for design,color,label in [('AlN','#158c85','AlN'),('graded_W_AlN','#d97629','Graded W + AlN')]:
  sel=d['design']==design;ax.scatter(d['temperature_background_K'][sel].mean(axis=1)+25,np.ptp(d['phi_rad'][sel],axis=1),color=color,s=35,label=label)
 ax.scatter(base['temperature_background_K'][cu].mean()+25,np.ptp(base['phi_rad'][cu]),marker='s',color='#1f67a8',s=45,label='Cu reference')
 for cid,text,offset in [('AlN_L320_R880','Original (320, 880)',(18,16)),('AlN_L460_R740','Lowest mean T\n(460, 740)',(10,20)),('AlN_L320_R600','Lowest phase spread\n(320, 600)',(48,-15))]:
  i=np.where(d['case_id']==cid)[0][0];x=d['temperature_background_K'][i].mean()+25;y=np.ptp(d['phi_rad'][i]);ax.annotate(text,xy=(x,y),xytext=(31.6,.32) if cid=='AlN_L320_R600' else (30.25,1.0) if cid=='AlN_L460_R740' else offset,textcoords='data' if cid in ['AlN_L320_R600','AlN_L460_R740'] else 'offset points',fontsize=10.5,arrowprops=dict(arrowstyle='->',lw=.6))
 ax.set(xlabel='Mean channel temperature (°C)',ylabel='Passive phase spread (rad)',title='(a) Temperature and phase objectives');ax.legend(frameon=False,fontsize=10.5,loc='upper right');ax.grid(alpha=.15)
 x=np.linspace(420,780,9);ax=axs[1];ph=base['phi_rad'][cu];ax.plot(x,ph-ph.mean(),'o-',ms=3,color='#1f67a8',label='Cu')
 for cid,label,color in [('AlN_L320_R880','Original: (320, 880)','#158c85'),('AlN_L460_R740','Min. mean T: (460, 740)','#8956a3'),('AlN_L320_R600','Min. spread: (320, 600)','#d97629')]:
  i=np.where(d['case_id']==cid)[0][0];ph=d['phi_rad'][i];ax.plot(x,ph-ph.mean(),'o-',ms=3,color=color,label=label)
 ax.set(xlabel='Channel x (µm)',ylabel='Phase relative to mean (rad)',title='(b) Differential phase profiles');ax.legend(frameon=False,fontsize=10.5,loc='lower right');ax.grid(alpha=.15)
 fig.tight_layout();tmp=ROOT/'figures/physical_placement.tmp.pdf';fig.savefig(tmp,format='pdf',bbox_inches='tight');assert tmp.stat().st_size>100;tmp.replace(ROOT/'figures/physical_placement.pdf');fig.savefig(ROOT/'figures/physical_placement.png',dpi=240,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
