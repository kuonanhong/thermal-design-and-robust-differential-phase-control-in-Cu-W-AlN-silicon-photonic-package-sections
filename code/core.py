#!/usr/bin/env python3
"""Package-scale linear thermomechanics; SI units; x-y section, invariant z.

Derived from the supplied reproducible cross-section code and extended for
finite AlN thermal boundary resistance and factorized multi-load operators.
No optical assumptions or ideal trim are embedded in this module.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, splu
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf
MATERIALS = {
 'Si': dict(k=130., E=130e9, nu=.28, alpha=2.6e-6, rho_e=1e-3),
 'Cu': dict(k=390., E=110e9, nu=.34, alpha=16.5e-6, rho_e=1.72e-8),
 'W': dict(k=170., E=400e9, nu=.28, alpha=4.5e-6, rho_e=5.6e-8),
 'AlN': dict(k=170., E=310e9, nu=.24, alpha=4.5e-6, rho_e=1e12),
 'oxide': dict(k=1.4, E=70e9, nu=.17, alpha=.5e-6, rho_e=1e12),
 'substrate': dict(k=20., E=25e9, nu=.25, alpha=12e-6, rho_e=1e12),
 'encapsulant': dict(k=.7, E=5e9, nu=.30, alpha=20e-6, rho_e=1e12),
}
CONFIG = dict(width_m=1.2e-3, height_m=.60e-3, ambient_C=25.,
 h_bottom_W_m2K=100000., h_top_W_m2K=500., EIC_power_W_per_m=250.,
 PIC_power_W_per_m=25., TSV_diameter_m=40e-6, TSV_length_m=200e-6,
 thermal_reference_note='Stress-free at the 25 C ambient; isotropic constant material properties')
DESIGNS = ['Cu', 'graded_W', 'AlN', 'graded_W_AlN']
TSV_X = np.array([100,240,380,520,660,800,940,1080])*1e-6

def geometry(nx=120,ny=60,package='3D',design='graded_W_AlN',scales=None,defect=-1):
 scales=scales or {}; W,H=CONFIG['width_m'],CONFIG['height_m']
 dx,dy=W/nx,H/ny; x=(np.arange(nx)+.5)*dx;y=(np.arange(ny)+.5)*dy
 X,Y=np.meshgrid(x,y);labels=np.full((ny,nx),'encapsulant',dtype='U16')
 labels[Y<100e-6]='substrate';labels[(Y>=100e-6)&(Y<300e-6)]='Si'
 labels[(Y>=300e-6)&(Y<320e-6)]='oxide'
 if package=='2.5D':
  pic=(X>=240e-6)&(X<840e-6)&(Y>=320e-6)&(Y<400e-6)
  eic=(X>=880e-6)&(X<1160e-6)&(Y>=320e-6)&(Y<460e-6)
 else:
  pic=(X>=240e-6)&(X<840e-6)&(Y>=320e-6)&(Y<400e-6)
  eic=(X>=340e-6)&(X<740e-6)&(Y>=420e-6)&(Y<520e-6)
  labels[(X>=240e-6)&(X<840e-6)&(Y>=400e-6)&(Y<420e-6)]='oxide'
 labels[pic|eic]='Si';props={key:np.zeros((ny,nx)) for key in ['k','E','nu','alpha']}
 for name,m in MATERIALS.items():
  for key in props:props[key][labels==name]=m[key]
 fws=[];via_masks=[]
 for i,c in enumerate(TSV_X):
  mask=(abs(X-c)<20e-6-1e-15)&(Y>=100e-6)&(Y<300e-6);via_masks.append(mask)
  dist=abs(c-600e-6)
  fw=((1. if dist<100e-6 else .5 if dist<250e-6 else 0.)*scales.get('W_strength',1.)) if 'graded_W' in design else 0.
  fws.append(fw);labels[mask]='W' if fw==1 else 'Cu/W' if fw else 'Cu'
  for key in props:
   props[key][mask]=(1-fw)*MATERIALS['Cu'][key]+fw*MATERIALS['W'][key]
  if i==defect:props['k'][mask]*=.1
 if 'AlN' in design:
  guard_centers=np.asarray(scales.get('guard_centers_um',[320.,880.]))*1e-6
  guard_width=scales.get('guard_width_m',40e-6)
  guard=np.zeros((ny,nx),dtype=bool)
  for gc in guard_centers:guard|=(X>=gc-guard_width/2)&(X<gc+guard_width/2)
  guard&=(Y>=220e-6)&(Y<320e-6)
  if any(np.any(guard & vm) for vm in via_masks):
   raise ValueError('AlN guard overlaps a TSV; choose a physically separate gap.')
  labels[guard]='AlN'
  for key in props:props[key][guard]=MATERIALS['AlN'][key]
  props['k'][guard]*=scales.get('AlN_k',1.)
 props['k']*=scales.get('k',1.);props['alpha']*=scales.get('alpha',1.)
 q=np.zeros((ny,nx))
 q[eic]=CONFIG['EIC_power_W_per_m']*scales.get('power',1.)/(eic.sum()*dx*dy)
 q[pic]=CONFIG['PIC_power_W_per_m']*scales.get('power',1.)/(pic.sum()*dx*dy)
 return dict(nx=nx,ny=ny,dx=dx,dy=dy,x=x,y=y,X=X,Y=Y,labels=labels,q=q,pic=pic,eic=eic,
             fw=fws,package=package,design=design,scales=scales,**props)

def thermal(g):
 nx,ny,dx,dy=g['nx'],g['ny'],g['dx'],g['dy'];k=g['k'];idx=np.arange(nx*ny).reshape(ny,nx)
 rows=[];cols=[];vals=[];diag=np.zeros(nx*ny)
 for a,b,cond in [(idx[:,:-1],idx[:,1:],2*k[:,:-1]*k[:,1:]/(k[:,:-1]+k[:,1:])*dy/dx),
                  (idx[:-1,:],idx[1:,:],2*k[:-1,:]*k[1:,:]/(k[:-1,:]+k[1:,:])*dx/dy)]:
  a,b,cond=a.ravel(),b.ravel(),cond.ravel();rows.extend([a,b]);cols.extend([b,a]);vals.extend([-cond,-cond]);np.add.at(diag,a,cond);np.add.at(diag,b,cond)
 hscale=g.get('scales',{}).get('h',1.);hb=CONFIG['h_bottom_W_m2K']*hscale;ht=CONFIG['h_top_W_m2K']
 gb=dx/(dy/(2*k[0,:])+1/hb);gt=dx/(dy/(2*k[-1,:])+1/ht)
 diag[idx[0,:]]+=gb;diag[idx[-1,:]]+=gt
 rows.append(idx.ravel());cols.append(idx.ravel());vals.append(diag)
 A=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(nx*ny,nx*ny)).tocsr()
 b=g['q'].ravel()*dx*dy;dT=spsolve(A,b).reshape(ny,nx)
 qin=float(b.sum());qbot=float(np.sum(gb*dT[0,:]));qtop=float(np.sum(gt*dT[-1,:]))
 return dT,dict(power_in_W_per_m=qin,bottom_out_W_per_m=qbot,top_out_W_per_m=qtop,
                energy_relative_error=abs(qin-qbot-qtop)/max(qin,1e-30))

def constitutive(E,nu):
 c=E/((1+nu)*(1-2*nu));D=np.zeros((len(E),3,3))
 D[:,0,0]=D[:,1,1]=c*(1-nu);D[:,0,1]=D[:,1,0]=c*nu;D[:,2,2]=c*(1-2*nu)/2
 return D

def B_matrix(xi,eta,dx,dy):
 dNdx=np.array([-(1-eta),1-eta,1+eta,-(1+eta)])/(2*dx)
 dNdy=np.array([-(1-xi),-(1+xi),1+xi,1-xi])/(2*dy)
 B=np.zeros((3,8));B[0,0::2]=dNdx;B[1,1::2]=dNdy;B[2,0::2]=dNdy;B[2,1::2]=dNdx
 return B

def elasticity(g,dT,patch=False):
 nx,ny,dx,dy=g['nx'],g['ny'],g['dx'],g['dy'];nn=(nx+1)*(ny+1)
 grid=np.arange(nn).reshape(ny+1,nx+1)
 en=np.stack([grid[:-1,:-1],grid[:-1,1:],grid[1:,1:],grid[1:,:-1]],axis=-1).reshape(-1,4)
 ed=(2*en[:,:,None]+np.arange(2)).reshape(-1,8)
 E,nu,alpha=g['E'].ravel(),g['nu'].ravel(),g['alpha'].ravel();D=constitutive(E,nu)
 beta=E*alpha/(1-2*nu)*dT.ravel();th=np.column_stack([beta,beta,np.zeros_like(beta)])
 Ke=np.zeros((nx*ny,8,8));fe=np.zeros((nx*ny,8));det=dx*dy/4
 for xi in [-1/np.sqrt(3),1/np.sqrt(3)]:
  for eta in [-1/np.sqrt(3),1/np.sqrt(3)]:
   B=B_matrix(xi,eta,dx,dy);Ke+=np.einsum('ai,eab,bj->eij',B,D,B)*det;fe+=th@B*det
 K=coo_matrix((Ke.ravel(),(np.repeat(ed,8,axis=1).ravel(),np.tile(ed,(1,8)).ravel())),shape=(2*nn,2*nn)).tocsr()
 f=np.bincount(ed.ravel(),weights=fe.ravel(),minlength=2*nn)
 if patch:
  X,Y=np.meshgrid(np.arange(nx+1)*dx,np.arange(ny+1)*dy)
  exact=np.stack([1e-4*X+2e-4*Y,-.5e-4*X+3e-4*Y],axis=-1).ravel()
  bn=np.unique(np.r_[grid[0,:],grid[-1,:],grid[:,0],grid[:,-1]]);fixed=(2*bn[:,None]+np.arange(2)).ravel()
  u=np.zeros(2*nn);u[fixed]=exact[fixed]
 else:
  fixed=np.r_[2*grid[0,:]+1,0];u=np.zeros(2*nn)
 free=np.setdiff1d(np.arange(2*nn),fixed)
 u[free]=spsolve(K[free][:,free],f[free]-K[free][:,fixed]@u[fixed])
 B=B_matrix(0.,0.,dx,dy);strain=u[ed]@B.T;stress=np.einsum('eij,ej->ei',D,strain)-th
 # epsilon_z = 0; sigma_z = lambda*(eps_x+eps_y)-E*alpha/(1-2nu)*dT.
 lam=E*nu/((1+nu)*(1-2*nu));sz=lam*(strain[:,0]+strain[:,1])-beta
 sx,sy,txy=stress.T;vm=np.sqrt(.5*((sx-sy)**2+(sy-sz)**2+(sz-sx)**2)+3*txy**2)
 nodalu=u.reshape(ny+1,nx+1,2);uy=nodalu[:,:,1]
 # Package-top displacement range is support-dependent, not a free 3D warpage.
 top_range=float(np.ptp(uy[-1,:]))
 result=dict(strain=strain.reshape(ny,nx,3),stress=stress.reshape(ny,nx,3),sigma_z=sz.reshape(ny,nx),
             von_mises=vm.reshape(ny,nx),u=nodalu,top_displacement_range_m=top_range,
             equilibrium_relative_residual=float(np.linalg.norm((K@u-f)[free])/max(np.linalg.norm(f[free]),1e-20)))
 if patch:result['patch_displacement_relative_error']=float(np.max(abs(u-exact))/np.max(abs(exact)))
 return result

class ThermalOperator:
 """Cell-centred conservative conduction, with AlN-face Kapitza resistance.

 R_K has SI units m² K/W and is applied once to faces whose labels contain
 exactly one AlN cell. All other interfaces are ideal. Mechanical interfaces
 remain perfectly bonded; contact resistance does not imply mechanical slip.
 """
 def __init__(self,g,R_K=0.,h_bottom=None,h_top=None):
  self.g=g; nx,ny,dx,dy=g['nx'],g['ny'],g['dx'],g['dy']; k=g['k']
  idx=np.arange(nx*ny).reshape(ny,nx); lab=g['labels']; rows=[];cols=[];vals=[];diag=np.zeros(nx*ny)
  interface_count=0
  for a,b,ka,kb,la,lb,dist,area in [
   (idx[:,:-1],idx[:,1:],k[:,:-1],k[:,1:],lab[:,:-1],lab[:,1:],dx,dy),
   (idx[:-1,:],idx[1:,:],k[:-1,:],k[1:,:],lab[:-1,:],lab[1:,:],dy,dx)]:
   inter=(la=='AlN') != (lb=='AlN'); interface_count+=int(inter.sum())
   cond=area/(dist/(2*ka)+dist/(2*kb)+R_K*inter)
   a,b,cond=a.ravel(),b.ravel(),cond.ravel(); rows.extend([a,b]); cols.extend([b,a]); vals.extend([-cond,-cond]);np.add.at(diag,a,cond);np.add.at(diag,b,cond)
  hb=CONFIG['h_bottom_W_m2K'] if h_bottom is None else h_bottom
  ht=CONFIG['h_top_W_m2K'] if h_top is None else h_top
  self.gb=dx/(dy/(2*k[0,:])+1/hb) if hb>0 else np.zeros(nx)
  self.gt=dx/(dy/(2*k[-1,:])+1/ht) if ht>0 else np.zeros(nx)
  diag[idx[0,:]]+=self.gb;diag[idx[-1,:]]+=self.gt
  rows.append(idx.ravel());cols.append(idx.ravel());vals.append(diag)
  self.A=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(nx*ny,nx*ny)).tocsc()
  self.factor=splu(self.A); self.interface_faces=interface_count
 def solve(self,q):
  g=self.g; q=np.asarray(q); multi=q.ndim==3
  rhs=(q.reshape(q.shape[0],-1).T if multi else q.ravel())*g['dx']*g['dy']
  t=self.factor.solve(rhs)
  fields=t.T.reshape(q.shape) if multi else t.reshape(q.shape)
  return fields
 def energy(self,q,t):
  pin=float(np.sum(q)*self.g['dx']*self.g['dy']); pout=float(self.gb@t[0,:]+self.gt@t[-1,:])
  return dict(power_in_W_per_m=pin,power_out_W_per_m=pout,energy_relative_error=abs(pin-pout)/max(abs(pin),1e-30))

class ElasticOperator:
 """Cached Q4 plane-strain stiffness with exact linear thermal loads."""
 def __init__(self,g):
  self.g=g; nx,ny,dx,dy=g['nx'],g['ny'],g['dx'],g['dy'];nn=(nx+1)*(ny+1)
  grid=np.arange(nn).reshape(ny+1,nx+1)
  en=np.stack([grid[:-1,:-1],grid[:-1,1:],grid[1:,1:],grid[1:,:-1]],axis=-1).reshape(-1,4)
  ed=(2*en[:,:,None]+np.arange(2)).reshape(-1,8)
  E,nu,alpha=g['E'].ravel(),g['nu'].ravel(),g['alpha'].ravel();D=constitutive(E,nu)
  beta_per_K=E*alpha/(1-2*nu); th=np.column_stack([beta_per_K,beta_per_K,np.zeros_like(beta_per_K)])
  Ke=np.zeros((nx*ny,8,8));fe_per_K=np.zeros((nx*ny,8));det=dx*dy/4
  for xi in [-1/np.sqrt(3),1/np.sqrt(3)]:
   for eta in [-1/np.sqrt(3),1/np.sqrt(3)]:
    B=B_matrix(xi,eta,dx,dy);Ke+=np.einsum('ai,eab,bj->eij',B,D,B)*det;fe_per_K+=th@B*det
  self.K=coo_matrix((Ke.ravel(),(np.repeat(ed,8,axis=1).ravel(),np.tile(ed,(1,8)).ravel())),shape=(2*nn,2*nn)).tocsr()
  self.fixed=np.r_[2*grid[0,:]+1,0];self.free=np.setdiff1d(np.arange(2*nn),self.fixed)
  self.factor=splu(self.K[self.free][:,self.free].tocsc())
  self.ed=ed;self.D=D;self.beta_per_K=beta_per_K;self.fe_per_K=fe_per_K;self.E=E;self.nu=nu;self.nn=nn
 def solve(self,dT):
  g=self.g;nx,ny=g['nx'],g['ny'];ed=self.ed
  fe=self.fe_per_K*dT.ravel()[:,None]
  f=np.bincount(ed.ravel(),weights=fe.ravel(),minlength=2*self.nn)
  u=np.zeros(2*self.nn);u[self.free]=self.factor.solve(f[self.free])
  strain=u[ed]@B_matrix(0.,0.,g['dx'],g['dy']).T
  beta=self.beta_per_K*dT.ravel();th=np.column_stack([beta,beta,np.zeros_like(beta)])
  stress=np.einsum('eij,ej->ei',self.D,strain)-th
  lam=self.E*self.nu/((1+self.nu)*(1-2*self.nu));sz=lam*(strain[:,0]+strain[:,1])-beta
  sx,sy,txy=stress.T;vm=np.sqrt(.5*((sx-sy)**2+(sy-sz)**2+(sz-sx)**2)+3*txy*txy)
  result=dict(strain=strain.reshape(ny,nx,3),stress=stress.reshape(ny,nx,3),sigma_z=sz.reshape(ny,nx),von_mises=vm.reshape(ny,nx),u=u.reshape(ny+1,nx+1,2),
   equilibrium_relative_residual=float(np.linalg.norm((self.K@u-f)[self.free])/max(np.linalg.norm(f[self.free]),1e-20)))
  return result

CHANNEL_X=np.linspace(420e-6,780e-6,9)
CHANNEL_Y=350e-6
HEATER_SIGMA_X=12e-6
HEATER_BOTTOM=380e-6
HEATER_TOP=400e-6

def heater_sources(g):
 """Nine Gaussian x, uniform vertical heaters, each 1 W/m in invariant z.

 Exact cell integrals avoid grid-dependent injected power. Sources are confined
 to y=380..400 µm, inside the PIC. These are equivalent volumetric line heaters,
 not resolved metal conductors; DC resistance and optical absorption excluded.
 """
 nx,ny,dx,dy=g['nx'],g['ny'],g['dx'],g['dy']
 xe=np.arange(nx+1)*dx;ye=np.arange(ny+1)*dy
 fy=np.maximum(0,np.minimum(ye[1:],HEATER_TOP)-np.maximum(ye[:-1],HEATER_BOTTOM))/(HEATER_TOP-HEATER_BOTTOM)
 result=[]
 for x0 in CHANNEL_X:
  fx=.5*np.diff(erf((xe-x0)/(np.sqrt(2)*HEATER_SIGMA_X)))
  a=np.outer(fy,fx);a/=a.sum();result.append(a/(dx*dy))
 return np.array(result)

def sample_fields(g,t,m):
 pts=np.column_stack([np.full(len(CHANNEL_X),CHANNEL_Y),CHANNEL_X])
 interp=lambda a:RegularGridInterpolator((g['y'],g['x']),a,bounds_error=True)(pts)
 dt=interp(t);ex=interp(m['strain'][:,:,0]);ey=interp(m['strain'][:,:,1]);alpha=MATERIALS['Si']['alpha']
 return dict(temperature_K=dt,total_strain_x=ex,total_strain_y=ey,total_strain_z=np.zeros_like(ex),
  elastic_strain_x=ex-alpha*dt,elastic_strain_y=ey-alpha*dt,elastic_strain_z=-alpha*dt,
  stress_x_Pa=interp(m['stress'][:,:,0]),stress_y_Pa=interp(m['stress'][:,:,1]),stress_z_Pa=interp(m['sigma_z']))
