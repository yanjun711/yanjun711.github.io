"""Render the blog's analytical illustrations as deterministic vector graphics.

Requires Python 3, NumPy and Matplotlib. Optional: --font-dir DIRECTORY with
Lato-Regular.ttf and Lato-Bold.ttf. SVGs contain glyph outlines and no external
resources. These are exact toy calculations, not model evaluation results.
"""
from pathlib import Path
import argparse
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1] / 'public/images/blog'
BLUE, ORANGE, GRAY = '#526b78', '#282828', '#666666'
INK, LIGHT = '#282828', '#e6e6e6'


def configure(font_dir):
    if font_dir:
        for weight in ['Regular', 'Bold']:
            font_manager.fontManager.addfont(str(Path(font_dir) / f'Lato-{weight}.ttf'))
    plt.rcParams.update({
        'font.family': 'Lato' if font_dir else 'DejaVu Sans',
        'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
        'axes.titleweight': 'normal', 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': GRAY, 'ytick.color': GRAY,
        'axes.edgecolor': '#999999', 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'lines.linewidth': 1.4, 'lines.markersize': 4.5,
        'legend.frameon': False, 'legend.fontsize': 9.5,
        'svg.fonttype': 'path', 'svg.hashsalt': 'yanjun-research-notes',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def panels(n, mobile=False, height=2.5):
    fig, axes = plt.subplots(n if mobile else 1, 1 if mobile else n,
                             figsize=(3.6 if mobile else 6.2, n*1.8 if mobile and n>1 else height),
                             layout='constrained', squeeze=False)
    return fig, axes.ravel()


def save(fig, slug, filename):
    target = ROOT/slug/filename
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format='svg', metadata={'Date': None, 'Creator': 'Yanjun Li; analytical illustration'})
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


def quiet_grid(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#ededed', linewidth=.65)


def analytic(kind,mobile):
    if kind=='gspo':
        fig,axes=panels(2,mobile)
        x=np.linspace(.7,1.3,301)
        for a,A,label in zip(axes,[1,-1],['(a) Positive advantage','(b) Negative advantage']):
            a.plot(x,A*x,color=GRAY,ls='--',lw=1,label='Unclipped')
            a.plot(x,np.minimum(A*x,A*np.clip(x,.9,1.1)),color=BLUE,label='Clipped')
            a.axvline(.9 if A<0 else 1.1,color='#c3cbd2',lw=.8)
            a.set(title=label,xlabel='Sequence ratio s',ylabel='Objective contribution',xlim=(.7,1.3),xticks=[.7,.9,1.1,1.3]);a.legend(loc='upper left' if A>0 else 'lower left');quiet_grid(a)
        save(fig,'gspo-sequence-importance-ratios','clipping'+('-mobile' if mobile else '')+'.svg')
    elif kind=='rae':
        fig,axes=panels(1,mobile);a=axes[0];t=np.linspace(0,1,201)
        for ratio,color in [(1,GRAY),(4,BLUE),(16,ORANGE)]:
            alpha=np.sqrt(ratio);a.plot(t,alpha*t/(1+(alpha-1)*t),color=color,ls={1:':',4:'--',16:'-'}[ratio],label=f'Dimension ratio {ratio}')
        a.set(xlabel='Reference time t',ylabel='Shifted time',xlim=(0,1),ylim=(0,1.03),xticks=[0,.5,1],yticks=[0,.5,1]);a.legend(loc='lower right');quiet_grid(a)
        save(fig,'rae-dimension-noise-schedule','schedule'+('-mobile' if mobile else '')+'.svg')


def dinov3(mobile):
    a=np.array([[1,0,2**-.5],[0,1,2**-.5],[2**-.5,2**-.5,1]])
    b=np.array([[1,0,1],[0,1,0],[1,0,1]])
    # The mobile view combines the two identical matrices; no tall three-panel stack.
    mats=[a,b] if mobile else [a,a,b]
    titles=['Reference = rotated','Changed patch 3'] if mobile else ['Reference','Rotated features','Changed patch 3']
    fig,axes=plt.subplots(1,len(mats),figsize=(3.6,2.0) if mobile else (6.2,2.5),layout='constrained')
    for ax,mat,title in zip(axes,mats,titles):
        ax.imshow(mat,vmin=0,vmax=1,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('gram',['#f7f7f7',BLUE]))
        for r,c in itertools.product(range(3),repeat=2):ax.text(c,r,f'{mat[r,c]:.2f}',ha='center',va='center',color='white' if mat[r,c]>.5 else INK,fontsize=8.5 if mobile else 10)
        ax.set(title=title,xticks=[0,1,2],yticks=[0,1,2],xticklabels=['1','2','3'],yticklabels=['1','2','3'],xlabel='Patch')
        ax.set_title(title,fontsize=8.5 if mobile else 10)
        ax.tick_params(length=0,labelsize=8.5 if mobile else 10)
        for spine in ax.spines.values():spine.set_visible(False)
    save(fig,'dinov3-gram-anchoring','gram-matrices'+('-mobile' if mobile else '')+'.svg')


def dynamic(mobile):
    fig,axes=panels(1,mobile,height=2.25);a=axes[0]
    for row,source in enumerate([0,2,4]):
        ready=source+2
        a.plot([source,source+6],[row,row],color='#e8e8e8',lw=10,solid_capstyle='butt')
        a.plot([source,ready],[row,row],color='#b6b6b6',lw=10,solid_capstyle='butt')
        a.plot([ready,ready+2],[row,row],color=BLUE,lw=10,solid_capstyle='butt')
        a.scatter([ready],[row],s=30,facecolors='white',edgecolors=BLUE,zorder=4)
    a.set(xlabel='Control tick (50 ms)',yticks=[0,1,2],yticklabels=['A₀','A₂','A₄'],xticks=[0,2,4,6,8,10],ylim=(2.8,-.6),xlim=(-.2,10.3))
    for i,(color,label) in enumerate([('#b6b6b6','Expired'),(BLUE,'Executed'),('#e8e8e8','Superseded')]):
        a.plot([],[],color=color,lw=5,label=label)
    a.legend(loc='lower left',bbox_to_anchor=(0,1.03),ncol=3,fontsize=8.5,
             columnspacing=.9,handlelength=1.6);quiet_grid(a,'x')
    save(fig,'dynamicvla-action-time-alignment','timeline'+('-mobile' if mobile else '')+'.svg')


def bptt(mobile):
    fig,axes=plt.subplots(figsize=(3.6 if mobile else 6.2,2.3),layout='constrained')
    a=axes;a.axis('off');a.set(xlim=(-.4,3.4),ylim=(-.05,1.9),aspect='equal')
    def arrow(start,end,color):
        a.add_patch(FancyArrowPatch(start,end,arrowstyle='->',mutation_scale=8,
                                   color=color,linewidth=1.1))
    for i in range(4):
        a.add_patch(Circle((i,.9),.18,edgecolor='#999999',facecolor='white',lw=.9))
        a.text(i,.9,'h'+str(i).translate(str.maketrans('0123','₀₁₂₃')),ha='center',va='center',fontsize=12)
        if i:
            arrow((i-1+.18,.95),(i-.18,.95),GRAY)
            arrow((i-.18,.85),(i-1+.18,.85),BLUE)
            a.text(i-.5,1.15,'fθ',ha='center',color=GRAY,fontsize=10)
            a.text(i,1.62,'x'+str(i).translate(str.maketrans('123','₁₂₃')),ha='center',fontsize=12)
            arrow((i,1.48),(i,1.08),GRAY)
            arrow((i+.045,.72),(i+.045,.32),GRAY)
            arrow((i-.045,.32),(i-.045,.72),BLUE)
            a.text(i,.17,'ℓ'+str(i).translate(str.maketrans('123','₁₂₃')),ha='center',fontsize=12)
    a.text(0,.48,'Initial',ha='center',color=GRAY,fontsize=9)
    save(fig,'bptt','unrolled'+('-mobile' if mobile else '')+'.svg')
    fig,axes=panels(1,mobile);a=axes[0];steps=np.arange(41)
    for gain,color,style in [(.8,BLUE,'-'),(1,GRAY,'--'),(1.2,ORANGE,'-')]:
        a.plot(steps,gain**steps,color=color,ls=style,label=f'a = {gain:.1f}')
    a.set_yscale('log');a.set(xlabel='Recurrent steps k',ylabel='Gradient magnitude |a|ᵏ',
        xlim=(0,40),ylim=(1e-4,1e4),xticks=[0,10,20,30,40],yticks=[1e-4,1e-2,1,1e2,1e4])
    a.legend(loc='upper left');quiet_grid(a)
    save(fig,'bptt','gradient-paths'+('-mobile' if mobile else '')+'.svg')


def main():
    renderers={f.__name__:f for f in [dinov3,dynamic,bptt]}
    renderers.update({kind:lambda mobile,k=kind:analytic(k,mobile) for kind in ['gspo','rae']})
    args=argparse.ArgumentParser();args.add_argument('--font-dir')
    args.add_argument('--only',nargs='+',choices=renderers)
    opts=args.parse_args();configure(opts.font_dir)
    for mobile in [False,True]:
        for name in opts.only or renderers:renderers[name](mobile)
    print('Rendered desktop and mobile SVGs for:', ', '.join(opts.only or renderers))


if __name__=='__main__':
    main()
