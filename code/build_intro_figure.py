#!/usr/bin/env python3
"""Build bilingual editable TikZ co-design schematics from exported nominal data.

The two plotted phase profiles are read from control_inputs.npz. The package
section uses the dimensions declared in core.py. This is a conceptual overview
of the numerical study, not an image of a measured device. Run from any folder.
Requires NumPy, XeLaTeX, TikZ and the bundled Noto fonts (ZH only).
"""
from pathlib import Path
import argparse
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TEXT = {
 'EN': {
  'title': 'Why cooling, optical phase and heater control must be co-designed',
  'a': '(a) Sources and layout',
  'b': '(b) Cooling and optical phase',
  'c': '(c) Robust fixed command',
  'geom': r'$1.2\times0.60$ mm section; $z$ is out of plane',
  'legend': r'Cu/W vias\quad AlN shunts\quad channel samples',
  'source': 'Independent source powers',
  'source2': r'EIC: $200$--$300$ mW\\PIC: $20$--$30$ mW',
  'source3': r'Nominal: $250/25$ mW over $L=1$ mm',
  'source4': 'A deterministic sensitivity box',
  'design': r'Move equal-volume AlN shunts.\\Recompute $G$ and $H$ for each layout.',
  'axis': r'Channel position $x$ ($\mu$m)',
  'axisy': r'Phase minus mean (rad)',
  'cu': 'Cu', 'aln':'Original AlN',
  'mean':'Mean temperature', 'spread':'Phase range',
  'less':'cooler', 'more':'less uniform',
  'phi':r'$\boldsymbol\phi=G\boldsymbol\ell+H\boldsymbol p$',
  'g':r'$G$: two source responses',
  'h':r'$H$: nine physical heater responses',
  'conflict':r'Lower mean temperature alone\\does not ensure smaller phase spread.',
  'box':'Four load vertices',
  'fixed':r'One shared\\command $\boldsymbol p_q$\\(held fixed)',
  'limits':r'$0\leq p_{q,j}\leq20$ mW\\$\sum_jp_{q,j}\leq100$ mW',
  'dac':'8-bit endpoint-inclusive DAC',
  'uncert':r'Calibration: $\pm2\%$ per $H$ entry\\Phase sensing: $\pm0.002$ rad per channel',
  'cert':r'$C_{\mathcal B}(\boldsymbol p_q)=\max_v C(\boldsymbol\ell_v,\boldsymbol p_q)$',
  'cert2':'Exact for the declared linear model',
  'target':'A nominal pass need not cover the box',
  'target2':r'AlN $320/600\,\mu$m, $\boldsymbol p=0$:\\$0.2333\ \longrightarrow\ 0.2791$ rad',
  'scopehead':'Declared scope',
  'scope':r'Steady 2D package; fixed source shapes.\\No transient feedback or experimental validation.',
  'checkhead':'Independent checks',
  'check':r'Energy and mechanics checks.\\Optical and LP reconstruction.\\Worst-case error attainment.',
  'modelhead':'Optical model sensitivity',
  'model':r'Local full-vector Maxwell check of EIM.\\Zero-power qualification can change.',
 },
 'ZH': {
  'title':'為何降溫、光學相位與加熱控制必須共同設計',
  'a':'(a) 熱源與封裝配置', 'b':'(b) 光學設計的衝突', 'c':'(c) 涵蓋整個負載盒的命令',
  'geom':r'$1.2\times0.60$ mm 截面；$z$ 垂直紙面',
  'legend':r'Cu/W 導通孔\quad AlN 護壁\quad 通道取樣點',
  'source':'獨立變動的熱源功率',
  'source2':r'EIC：$200$--$300$ mW\\PIC：$20$--$30$ mW',
  'source3':r'名目 $250/25$ mW；光路長 $L=1$ mm',
  'source4':'確定性的靈敏度盒狀集合',
  'design':r'在固定 AlN 體積下移動護壁；\\重新計算背景相位與加熱器響應。',
  'axis':r'通道位置 $x$ ($\mu$m)', 'axisy':'扣除平均值後的相位 (rad)',
  'cu':'Cu', 'aln':'AlN，原始配置', 'mean':'平均溫度', 'spread':'相位範圍',
  'less':'較冷', 'more':'較不均勻',
  'phi':r'$\boldsymbol\phi=G\boldsymbol\ell+H\boldsymbol p$',
  'g':r'$G$：兩個獨立熱源響應', 'h':r'$H$：九個物理加熱器響應',
  'conflict':r'僅降低平均溫度，\\不保證通道間的相位差變小。',
  'box':'四個負載頂點', 'fixed':r'全盒共用的\\固定量化命令\\$\boldsymbol p_q$',
  'limits':r'$0\leq p_{q,j}\leq20$ mW\\$\sum_jp_{q,j}\leq100$ mW',
  'dac':'8 位元、含端點的 DAC',
  'uncert':r'校準：$H$ 各元素 $\pm2\%$\\感測：每通道 $\pm0.002$ rad',
  'cert':r'$C_{\mathcal B}(\boldsymbol p_q)=\max_v C(\boldsymbol\ell_v,\boldsymbol p_q)$',
  'cert2':'在線性盒模型內，頂點化簡精確成立',
  'target':'名目合格未必能涵蓋整個負載盒',
  'target2':r'AlN $320/600\,\mu$m，$\boldsymbol p=0$：\\$0.2333\ \longrightarrow\ 0.2791$ rad',
  'scopehead':'明列模型範圍',
  'scope':r'穩態二維封裝、線性固定熱源形狀。\\不宣稱暫態回授或實測元件效能。',
  'checkhead':'獨立數值檢查',
  'check':r'守恆、機械、光學投影、\\LP 重建與最壞擾動界之可達性。',
  'modelhead':'光學模型靈敏度',
  'model':r'以局部完整向量 Maxwell 模型檢查 EIM。\\名目零加熱的合格判定可能改變。',
 }
}

def build(locale):
    t = TEXT[locale]
    data = np.load(ROOT/'results/control_inputs.npz',allow_pickle=False)
    curves = []
    for name in ['Cu_R0e+00_k170_n120','AlN_R1e-08_k170_n120']:
        index = list(data['case_id']).index(name)
        phase = data['phi_rad'][index]
        curves.append(phase-phase.mean())
    pre = r'''\documentclass[tikz,border=2pt]{standalone}
\usepackage{amsmath,amssymb,lmodern,fontspec}
\usetikzlibrary{arrows.meta,calc}
'''
    if locale=='ZH':
        pre += r'''\setmainfont[Path=../assets/fonts/,BoldFont=NotoSerifCJKtc-Bold.otf]{NotoSerifCJKtc-Regular.otf}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt
'''
    else:
        pre += r'\setmainfont[BoldFont=NimbusSans-Bold.otf]{NimbusSans-Regular.otf}'+'\n'
    pre += r'''\definecolor{ink}{HTML}{17374A}
\definecolor{teal}{HTML}{087F7E}
\definecolor{blue}{HTML}{39658C}
\definecolor{orange}{HTML}{BC662F}
\definecolor{muted}{HTML}{586B77}
\definecolor{line}{HTML}{C9D6DD}
\definecolor{paper}{HTML}{F6F9FA}
\definecolor{cu}{HTML}{D8973C}
\definecolor{tungsten}{HTML}{8191A0}
\definecolor{aln}{HTML}{69AFA4}
\definecolor{silicon}{HTML}{D0E3EF}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,font=\fontsize{8.3}{10}\selectfont,text=ink,
  every node/.style={inner sep=0pt},
  panel/.style={rounded corners=2pt,fill=paper,draw=line,line width=.5pt},
  arr/.style={-{Latex[length=2mm,width=1.4mm]},draw=muted,line width=.75pt}]
\path[use as bounding box] (0,0) rectangle (16,9.55);
'''
    q=[pre]
    def add(s):q.append(s+'\n')
    def node(x,y,s,style=''):
        add(r'\node['+style+f'] at ({x},{y})'+' {'+s+'};')
    node(8,9.25,t['title'],r'font=\fontsize{12}{14}\selectfont\bfseries')
    for a,b,label in [(0,5.1,'a'),(5.45,10.55,'b'),(10.9,16,'c')]:
        add(fr'\draw[panel] ({a},2.03) rectangle ({b},8.78);')
        node((a+b)/2,8.42,t[label],r'font=\fontsize{8.5}{11}\selectfont\bfseries')
    # Exact scaled section, with the nominal two local shunts. No 3D ring.
    gx,gy,scale=.36,5.69,4.38/1200
    def rect(x1,y1,x2,y2,color):
        add(fr'\fill[{color}] ({gx+x1*scale:.4f},{gy+y1*scale:.4f}) rectangle ({gx+x2*scale:.4f},{gy+y2*scale:.4f});')
    rect(0,0,1200,600,'white')
    rect(0,0,1200,100,'tungsten!75')
    rect(0,100,1200,300,'silicon')
    rect(0,300,1200,320,'cu!35')
    rect(240,320,840,400,'silicon')
    rect(240,400,840,420,'cu!35')
    rect(340,420,740,520,'silicon')
    for x,c in zip([100,240,380,520,660,800,940,1080],['cu','cu','cu!50!tungsten','tungsten','tungsten','cu!50!tungsten','cu','cu']):
        rect(x-20,100,x+20,300,c)
    for x in [320,880]:rect(x-20,220,x+20,320,'aln')
    for x in np.linspace(420,780,9):
        rect(x-7,380,x+7,400,'ink')
        add(fr'\fill[orange] ({gx+x*scale:.4f},{gy+350*scale:.4f}) circle (.022);')
    add(fr'\draw[line,line width=.4pt] ({gx},{gy}) rectangle ({gx+4.38},{gy+2.19});')
    node(gx+540*scale,gy+471*scale,'EIC',r'font=\fontsize{8.5}{10}\selectfont\bfseries')
    node(gx+540*scale,gy+337*scale,'PIC',r'font=\fontsize{7.8}{9}\selectfont\bfseries')
    node(2.55,5.43,t['geom'],r'font=\fontsize{7}{9}\selectfont')
    node(2.55,5.1,t['legend'],r'font=\fontsize{7}{9}\selectfont')
    add(r'\draw[line] (.35,4.87)--(4.75,4.87);')
    node(2.55,4.57,t['source'],r'font=\bfseries')
    node(2.55,4.02,t['source2'],r'align=center,font=\fontsize{9.1}{12}\selectfont')
    node(2.55,3.44,t['source3'],r'font=\fontsize{7.8}{10}\selectfont')
    node(2.55,3.08,t['source4'],r'text=muted,font=\fontsize{7.6}{9}\selectfont')
    node(2.55,2.51,t['design'],r'align=center,text width=4.7cm,font=\fontsize{7.3}{10}\selectfont')
    # Actual centered phase profiles, sampled at the nine declared channels.
    x0,x1,y0,y1=6.40,10.15,5.7,7.72
    pmin,pmax=-.60,.30
    X=lambda j: x0+j*(x1-x0)/8
    Y=lambda p:y0+(p-pmin)/(pmax-pmin)*(y1-y0)
    for v in [-.4,0,.2]:
        add(fr'\draw[line,line width=.35pt] ({x0},{Y(v):.4f})--({x1},{Y(v):.4f});')
        node(x0-.12,Y(v),f'{v:.1f}',r'anchor=east,font=\fontsize{7}{8}\selectfont')
    add(fr'\draw[muted,line width=.4pt] ({x0},{y1})--({x0},{y0})--({x1},{y0});')
    for j,x in [(0,420),(4,600),(8,780)]:
        node(X(j),y0-.15,str(x),r'anchor=north,font=\fontsize{7}{8}\selectfont')
    node(8.18,5.18,t['axis'],r'font=\fontsize{7.6}{9}\selectfont')
    node(5.64,6.72,t['axisy'],r'rotate=90,font=\fontsize{7.5}{9}\selectfont')
    for curve,color in zip(curves,['blue','teal']):
        coords=' '.join(f'({X(j):.4f},{Y(v):.4f})' for j,v in enumerate(curve))
        add(fr'\draw[{color},line width=1pt] plot coordinates {{{coords}}};')
        for j,v in enumerate(curve):
            add(fr'\fill[{color}] ({X(j):.4f},{Y(v):.4f}) circle (.026);')
    for yy,color,lab in [(6.18,'blue','cu'),(5.87,'teal','aln')]:
        add(fr'\draw[{color},line width=1pt] (8.18,{yy})--(8.55,{yy});')
        node(8.65,yy,t[lab],r'anchor=west,font=\fontsize{7.4}{9}\selectfont')
    node(7.48,4.76,t['mean'],r'font=\fontsize{7.6}{9}\selectfont')
    node(9.56,4.76,t['spread'],r'font=\fontsize{7.6}{9}\selectfont')
    node(6.0,4.38,'Cu',r'text=blue,anchor=west,font=\bfseries')
    node(6.0,4.02,'AlN',r'text=teal,anchor=west,font=\bfseries')
    node(7.48,4.38,r'$34.832\,{}^\circ$C')
    node(7.48,4.02,r'$32.210\,{}^\circ$C')
    node(9.56,4.38,r'$0.3945$ rad')
    node(9.56,4.02,r'$0.7699$ rad')
    node(8,3.56,t['phi'],r'font=\fontsize{11}{13}\selectfont')
    node(8,3.18,t['g'],r'font=\fontsize{7.8}{10}\selectfont')
    node(8,2.87,t['h'],r'font=\fontsize{7.8}{10}\selectfont')
    # Two arrows state the causality without implying a physical feedback loop.
    add(r'\draw[arr] (5.13,6.91)--(5.4,6.91);')
    add(r'\draw[arr] (10.58,6.91)--(10.85,6.91);')
    node(8,2.39,t['conflict'],r'align=center,text width=4.8cm,text=orange,font=\fontsize{7.0}{9.2}\selectfont\bfseries')
    # The vertex reduction uses one identical command across all four corners.
    add(r'\fill[teal!6] (11.42,6.71) rectangle (12.99,7.74);')
    add(r'\draw[teal,line width=.7pt] (11.42,6.71) rectangle (12.99,7.74);')
    for x,y in [(11.42,6.71),(11.42,7.74),(12.99,6.71),(12.99,7.74)]:
        add(fr'\fill[teal] ({x},{y}) circle (.047);')
    node(12.2,6.43,t['box'],r'font=\fontsize{7.5}{9}\selectfont')
    node(12.2,7.21,r'$\boldsymbol\ell\in\mathcal B$',r'font=\fontsize{10}{12}\selectfont')
    add(r'\draw[arr] (13.18,7.2)--(13.83,7.2);')
    node(14.78,7.2,t['fixed'],r'align=center,text width=1.95cm,font=\fontsize{7.7}{10}\selectfont\bfseries')
    node(13.45,5.94,t['limits'],r'align=center,font=\fontsize{8}{11}\selectfont')
    node(13.45,5.38,t['dac'],r'font=\fontsize{8.1}{10}\selectfont')
    node(13.45,4.86,t['uncert'],r'align=center,font=\fontsize{7.1}{10}\selectfont')
    add(r'\draw[line] (11.25,4.43)--(15.65,4.43);')
    node(13.45,4.03,t['cert'],r'font=\fontsize{10}{12}\selectfont')
    node(13.45,3.63,t['cert2'],r'font=\fontsize{6.8}{9}\selectfont')
    node(13.45,3.05,t['target'],r'text=orange,font=\fontsize{7.1}{10}\selectfont\bfseries')
    node(13.45,2.54,t['target2'],r'align=center,font=\fontsize{8.2}{11}\selectfont')
    # Lower strip: verification and qualification are distinct from experiment.
    for a,b,head,txt in [(0,5.1,'scopehead','scope'),(5.45,10.55,'checkhead','check'),(10.9,16,'modelhead','model')]:
        add(fr'\draw[panel,fill=white] ({a},.10) rectangle ({b},1.73);')
        add(fr'\fill[teal] ({a+.02},.30) rectangle ({a+.07},1.51);')
        node((a+b)/2,1.31,t[head],r'font=\fontsize{8.8}{11}\selectfont\bfseries')
        node((a+b)/2,.72,t[txt],r'align=center,text width=4.78cm,font=\fontsize{6.6}{9.1}\selectfont')
    add(r'\end{tikzpicture}\end{document}')
    out=ROOT/'figures'/f'intro_codesign_{locale}.tex'
    out.write_text(''.join(q),encoding='utf8')
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--no-compile',action='store_true')
    args=p.parse_args()
    for locale in ['EN','ZH']:
        path=build(locale)
        if not args.no_compile:
            result=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',path.name],
                                  cwd=path.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            if result.returncode:
                raise RuntimeError(result.stdout[-8000:])
            print(path.with_suffix('.pdf'))

if __name__=='__main__':main()
