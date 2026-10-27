"""Render the blog's analytical illustrations as deterministic vector graphics.

Requires Python 3, NumPy and Matplotlib. Optional: --font-dir DIRECTORY with
Lato-Regular.ttf and Lato-Bold.ttf. SVGs contain glyph outlines and no external
resources. These are exact toy calculations, not model evaluation results.
"""
from pathlib import Path
import argparse
import itertools
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1] / 'public/images/blog'
BLUE, ORANGE, GRAY = '#2864a0', '#b45d23', '#687580'
INK, LIGHT = '#30383e', '#dce5ee'


def configure(font_dir):
    if font_dir:
        for weight in ['Regular', 'Bold']:
            font_manager.fontManager.addfont(str(Path(font_dir) / f'Lato-{weight}.ttf'))
    plt.rcParams.update({
        'font.family': 'Lato' if font_dir else 'DejaVu Sans',
        'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
        'axes.titleweight': 'normal', 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': GRAY, 'ytick.color': GRAY,
        'axes.edgecolor': '#a1a9b0', 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'lines.linewidth': 1.65, 'lines.markersize': 4.5,
        'legend.frameon': False, 'legend.fontsize': 9.5,
        'svg.fonttype': 'path', 'svg.hashsalt': 'yanjun-research-notes',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def panels(n, mobile=False, height=3.2):
    fig, axes = plt.subplots(n if mobile else 1, 1 if mobile else n,
                             figsize=(3.6 if mobile else 6.2, n*2.35 if mobile and n>1 else height),
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
    ax.grid(axis=axis, color='#e8ecef', linewidth=.65)


def latent(mobile):
    fig, (a,b)=panels(2,mobile)
    t=np.linspace(0,1,4); z=np.c_[1-t,t]
    w=np.array([[1,0,-1],[0,1,1]])
    q=z@w
    assert np.allclose(q[:,2],2*t-1)
    for c,color in enumerate([BLUE,ORANGE]):
        a.plot(t,z[:,c],'-o',color=color,label=f'Input channel {c+1}')
    a.set(title='(a) Interpolate the time grid',xlabel='Time',ylabel='Latent value',xticks=[0,1/3,2/3,1],xticklabels=['0','⅓','⅔','1'],ylim=(-.1,1.45))
    a.legend(loc='upper center');quiet_grid(a)
    for c,color in enumerate([BLUE,ORANGE,GRAY]):
        b.plot(t,q[:,c],'-o',color=color,label=f'Output {c+1}')
    b.set(title='(b) Apply the channel projection',xlabel='Time',ylabel='Token value',xticks=[0,1/3,2/3,1],xticklabels=['0','⅓','⅔','1'],ylim=(-1.15,1.8))
    b.legend(loc='upper center',ncol=2 if mobile else 1);quiet_grid(b)
    save(fig,'latent-to-4d-alignment','interface'+('-mobile' if mobile else '')+'.svg')


def sam(mobile):
    fig,(a,b)=panels(2,mobile)
    a.plot([0,9],[0,4.5],color=GRAY,lw=1)
    a.scatter([4],[2],c=BLUE,s=38,zorder=4)
    a.scatter([8],[4],facecolors='white',edgecolors=ORANGE,s=65,zorder=4,lw=1.8)
    a.annotate('(4, 2)',(4,2),xytext=(7,-19),textcoords='offset points',color=BLUE,fontsize=10)
    a.annotate('(8, 4)',(8,4),xytext=(-48,12),textcoords='offset points',color=ORANGE,fontsize=10)
    a.scatter([0],[0],color=INK,s=15,zorder=4)
    a.set(title='(a) Two scales, one camera ray',xlabel='Camera depth Z',ylabel='Camera height Y',xlim=(-.4,9.4),ylim=(-.3,5.2),xticks=[0,4,8],yticks=[0,2,4])
    a.text(.05,.92,'Y / Z = ½',transform=a.transAxes,va='top');quiet_grid(a)
    uv=np.array([[25,50],[0,0],[20,0]])
    a_order=[0,1,2,0]
    b.plot(uv[a_order,0],uv[a_order,1],color=BLUE,alpha=.55,lw=1)
    b.scatter(uv[:,0],uv[:,1],s=70,facecolors='white',edgecolors=ORANGE,lw=1.5,label='Doubled scale',zorder=3)
    b.scatter(uv[:,0],uv[:,1],s=16,color=BLUE,label='Original scale',zorder=4)
    b.annotate('(25, 50)',uv[0],xytext=(-58,10),textcoords='offset points',fontsize=10)
    b.set(title='(b) Identical image projections',xlabel='Image u (pixels)',ylabel='Image v (pixels)',xlim=(-8,38),ylim=(-10,82),xticks=[0,25],yticks=[0,25,50])
    b.legend(loc='upper left',fontsize=9);quiet_grid(b)
    save(fig,'sam3d-shape-layout','joint-geometry'+('-mobile' if mobile else '')+'.svg')


def molmo(mobile):
    fig,ax=panels(1,mobile,height=3.15);a=ax[0]
    sims=np.cos([.05,.1,.05,.1]);idx=np.arange(1,5)
    a.axhline(.996,color=GRAY,ls='--',lw=1)
    a.text(.62,.99612,'Update threshold 0.996',fontsize=10,color=GRAY,va='bottom')
    for k,(x,y,old,new) in enumerate(zip(idx,sims,[12,33,7,80],[12,35,7,81])):
        update=y<.996;color=BLUE if update else GRAY
        a.vlines(x,.994,y,color=color,alpha=.5,lw=1)
        a.scatter([x],[y],color=color,s=36,zorder=3)
        label=f'{old} → {new}' if update else f'{old} replayed'
        a.annotate(label,(x,y),xytext=(0,12 if not update else -22),textcoords='offset points',ha='center',fontsize=10,color=color)
    a.set(xlabel='Depth-code position in raster order',ylabel='RGB cosine similarity',xlim=(.5,4.5),ylim=(.994,1.001),xticks=idx,yticks=[.994,.996,.998,1.0])
    a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'));quiet_grid(a)
    save(fig,'molmoact2-selective-depth','replay'+('-mobile' if mobile else '')+'.svg')


def utonia(mobile):
    fig,(a,b)=panels(2,mobile)
    d=np.linspace(0,.02,100)
    for g,color,label in [(.01,BLUE,'g = 0.01 m'),(.1,ORANGE,'g = 0.10 m')]:
        a.plot(d,d/g,color=color,label=label)
    a.scatter([.01],[1],s=30,color=BLUE,zorder=3)
    a.set(title='(a) A chosen spatial unit',xlabel='Metric separation (m)',ylabel='Aligned separation',xlim=(0,.02),ylim=(0,2.35),xticks=[0,.01,.02])
    a.legend(loc='upper left');quiet_grid(a)
    extent=np.geomspace(.01,2,150)
    b.plot(extent,.01/extent,color=GRAY,label='Bounding-box scaling')
    b.plot(extent,np.ones_like(extent),color=BLUE,label='Fixed g = 0.01 m')
    b.set_xscale('log');b.set_yscale('log')
    b.set(title='(b) Add a distant point',xlabel='Cloud x-extent (m)',ylabel='Same local edge after scaling',xticks=[.01,.1,1],xticklabels=['0.01','0.1','1'],yticks=[.01,.1,1],yticklabels=['0.01','0.1','1'],ylim=(.003,4))
    b.legend(loc='lower left',fontsize=9);quiet_grid(b)
    save(fig,'utonia-perceptual-granularity','units'+('-mobile' if mobile else '')+'.svg')


def minimax(mobile):
    fig,axes=panels(2,mobile)
    scores=[[3,1,.2,-1],[.2,3,.3,-1]]
    for g,a in enumerate(axes):
        vals=scores[g];a.axhline(0,color='#ccd3d9',lw=.7)
        for i,v in enumerate(vals):
            chosen=i in [g,3];color=BLUE if i==g else GRAY
            a.bar(i+1,v,width=.52,facecolor=color if chosen else 'white',edgecolor=color if chosen else '#b8c1c8',linewidth=1.1,zorder=2)
        a.set(title=f'({"ab"[g]}) Group {g+1}',xlabel='Context block',ylabel='Maximum index score',xticks=[1,2,3,4],ylim=(-1.9,4.2))
        a.annotate('Selected',(g+1,3),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10,color=BLUE)
        a.annotate('Local',(4,-1),xytext=(0,-18),textcoords='offset points',ha='center',fontsize=10,color=GRAY)
        quiet_grid(a)
    save(fig,'minimax-group-block-selection','selection'+('-mobile' if mobile else '')+'.svg')


def mova(mobile):
    fig,(a,b)=panels(2,mobile)
    j=np.linspace(0,8,241)
    native=np.cos((j-1)/2);aligned=np.cos((j-4)/2)
    for ax,curve,title in [(a,native,'(a) Native token indices'),(b,aligned,'(b) Shared physical time')]:
        ax.axvline(.5,color='#aeb8c0',lw=1,ls='--')
        ax.plot(j/8,curve,color=BLUE)
        ax.scatter([.125 if ax is a else .5],[1],color=BLUE,s=28,zorder=3)
        ax.set(title=title,xlabel='Audio time (s)',ylabel='One-pair rotary score',xlim=(0,1),ylim=(-1.15,1.45),xticks=[0,.5,1],yticks=[-1,0,1])
        ax.text(.98,.94,'Video: 0.5 s',transform=ax.transAxes,ha='right',va='top',fontsize=10,color=GRAY)
        quiet_grid(ax)
    save(fig,'mova-aligned-rope','clocks'+('-mobile' if mobile else '')+'.svg')


def rlsd(mobile):
    fig,axes=panels(2,mobile)
    r=np.geomspace(.25,4,250)
    for i,(a,A) in enumerate(zip(axes,[1,-1])):
        coef=A*(.5+.5*np.clip(r**A,.8,1.2))
        a.axhline(A,color='#aeb8c0',lw=1,ls='--')
        a.plot(r,coef,color=BLUE)
        vals=A*(.5+.5*np.clip(np.array([.5,1,2.])**A,.8,1.2))
        a.scatter([.5,1,2],vals,color=BLUE,s=26,zorder=3)
        a.set_xscale('log',base=2)
        a.set(title='(a) Positive advantage' if A>0 else '(b) Negative advantage',xlabel='Teacher / student evidence ratio',ylabel='Token advantage',xticks=[.25,.5,1,2,4],xticklabels=['¼','½','1','2','4'],ylim=(.82,1.18) if A>0 else (-1.18,-.82),yticks=[.9,1,1.1] if A>0 else [-1.1,-1,-.9])
        quiet_grid(a)
    save(fig,'rlsd-update-sign-magnitude','credit'+('-mobile' if mobile else '')+'.svg')


def fuse(mobile):
    fig,(a,b)=panels(2,mobile)
    vals=[-1,-.5,0,0,0,.5,1];ys=[1,1,1,2,3,1,1]
    a.scatter(vals,ys,s=35,color=BLUE,zorder=3)
    a.set(title='(a) Seven nonempty subsets',xlabel='Disagreement coordinate',ylabel='Subset multiplicity',xticks=[-1,-.5,0,.5,1],yticks=[1,2,3],xlim=(-1.25,1.25),ylim=(.4,3.7))
    a.text(.5,.93,'Shared coordinate = 1',transform=a.transAxes,ha='center',fontsize=10,color=GRAY);quiet_grid(a)
    ps=np.linspace(0,.999,200);cs=[]
    for p in ps:
        cs.append(sum(math.comb(3,r)*(1-p)**r*p**(3-r)/(1-p**3)*(3-r)/(r*2) for r in range(1,4)))
    b.plot(ps,cs,color=BLUE)
    b.scatter([.5],[15/28],s=28,color=BLUE,zorder=3)
    b.annotate('15/28',(.5,15/28),xytext=(8,-17),textcoords='offset points',fontsize=10)
    b.set(title='(b) Strength, not noise direction',xlabel='Layer drop probability p',ylabel='Variance coefficient c₃(p)',xlim=(0,1),ylim=(-.03,1.05),xticks=[0,.5,1],yticks=[0,.5,1]);quiet_grid(b)
    save(fig,'fusereg-random-layer-subsets','subsets'+('-mobile' if mobile else '')+'.svg')


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
    elif kind=='self':
        fig,axes=panels(1,mobile);a=axes[0]
        t=np.arange(4);clean=np.array([1,1.05,1.05,1.05])
        roll=np.ones(4)
        for i in range(1,4):roll[i]=.9*roll[i-1]+.15
        np.testing.assert_allclose(roll,[1,1.05,1.095,1.1355])
        a.axhline(1,color=GRAY,lw=1,ls=':',label='Target')
        a.plot(t,clean,'o-',color=GRAY,label='Clean prefix')
        a.plot(t,roll,'o-',color=BLUE,label='Generated prefix')
        a.set(xlabel='Generated step',ylabel='Scalar state',xticks=t,ylim=(.99,1.19),yticks=[1,1.05,1.1,1.15]);a.legend(loc='upper left');quiet_grid(a)
        save(fig,'self-forcing-generated-prefixes','prefix-drift'+('-mobile' if mobile else '')+'.svg')
    elif kind=='rae':
        fig,axes=panels(1,mobile);a=axes[0];t=np.linspace(0,1,201)
        for ratio,color in [(1,GRAY),(4,BLUE),(16,ORANGE)]:
            alpha=np.sqrt(ratio);a.plot(t,alpha*t/(1+(alpha-1)*t),color=color,ls='--' if ratio==1 else '-',label=f'Dimension ratio {ratio}')
        a.set(xlabel='Reference time t',ylabel='Shifted time',xlim=(0,1),ylim=(0,1.03),xticks=[0,.5,1],yticks=[0,.5,1]);a.legend(loc='lower right');quiet_grid(a)
        save(fig,'rae-dimension-noise-schedule','schedule'+('-mobile' if mobile else '')+'.svg')
    elif kind=='mhc':
        fig,axes=panels(1,mobile);a=axes[0];steps=np.arange(13)
        for curve,color,label in [(1.1**steps,ORANGE,'Unconstrained: ×1.1'),(np.ones_like(steps),GRAY,'Shared mode: ×1'),(.5**steps,BLUE,'Difference mode: ×½')]:a.plot(steps,curve,color=color,label=label)
        a.set_yscale('log');a.set(xlabel='Mixing steps',ylabel='Norm / initial norm',xticks=[0,4,8,12],ylim=(1e-4,30));a.legend(loc='lower left');quiet_grid(a)
        save(fig,'mhc-sinkhorn-residual-mixing','mixing-depth'+('-mobile' if mobile else '')+'.svg')
    elif kind=='long':
        fig,axes=panels(2,mobile)
        weight=1/(1+np.exp(-1/np.sqrt(2)))
        for a,old,new,title in [(axes[0],[.5,.5],[weight,1-weight],'(a) Attention probability'),(axes[1],[.5,.5],[weight,1.],'(b) Returned value')]:
            for i in range(2):a.plot([old[i],new[i]],[i,i],color='#b7c1c8',lw=1.3)
            a.scatter(old,[0,1],color=GRAY,s=25,label='Stale cache',zorder=3)
            a.scatter(new,[0,1],color=BLUE,s=32,label='Re-cached',zorder=3)
            a.set(title=title,xlabel='Probability' if a is axes[0] else 'Output coordinate value',yticks=[0,1],yticklabels=['Token 1','Token 2'] if a is axes[0] else ['Coord. 1','Coord. 2'],xlim=(.2,1.1),ylim=(-.65,1.5))
            a.invert_yaxis();a.legend(loc='upper right',fontsize=9);quiet_grid(a,'x')
        save(fig,'longlive-kv-recache','conditional-cache'+('-mobile' if mobile else '')+'.svg')


def dinov3(mobile):
    a=np.array([[1,0,2**-.5],[0,1,2**-.5],[2**-.5,2**-.5,1]])
    b=np.array([[1,0,1],[0,1,0],[1,0,1]])
    # The mobile view combines the two identical matrices; no tall three-panel stack.
    mats=[a,b] if mobile else [a,a,b]
    titles=['Reference = rotated','Changed patch 3'] if mobile else ['Reference','Rotated features','Changed patch 3']
    fig,axes=plt.subplots(1,len(mats),figsize=(3.6,2.0) if mobile else (6.2,2.5),layout='constrained')
    for ax,mat,title in zip(axes,mats,titles):
        ax.imshow(mat,vmin=0,vmax=1,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('gram',['#f2f4f6','#2864a0']))
        for r,c in itertools.product(range(3),repeat=2):ax.text(c,r,f'{mat[r,c]:.2f}',ha='center',va='center',color='white' if mat[r,c]>.5 else INK,fontsize=8.5 if mobile else 10)
        ax.set(title=title,xticks=[0,1,2],yticks=[0,1,2],xticklabels=['1','2','3'],yticklabels=['1','2','3'],xlabel='Patch')
        ax.set_title(title,fontsize=8.5 if mobile else 10)
        ax.tick_params(length=0,labelsize=8.5 if mobile else 10)
        for spine in ax.spines.values():spine.set_visible(False)
    save(fig,'dinov3-gram-anchoring','gram-matrices'+('-mobile' if mobile else '')+'.svg')


def dynamic(mobile):
    fig,axes=panels(1,mobile,height=3.0);a=axes[0]
    for row,source in enumerate([0,2,4]):
        ready=source+2
        a.plot([source,source+6],[row,row],color='#d8dfe5',lw=10,solid_capstyle='butt')
        a.plot([source,ready],[row,row],color='#adb6bf',lw=10,solid_capstyle='butt')
        a.plot([ready,ready+2],[row,row],color=BLUE,lw=10,solid_capstyle='butt')
        a.scatter([ready],[row],s=30,facecolors='white',edgecolors=BLUE,zorder=4)
    a.set(xlabel='Control tick (50 ms)',yticks=[0,1,2],yticklabels=['A₀','A₂','A₄'],xticks=[0,2,4,6,8,10],ylim=(2.8,-.6),xlim=(-.2,10.3))
    for i,(color,label) in enumerate([('#adb6bf','Expired'),(BLUE,'Executed'),('#d8dfe5','Superseded')]):
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
        a.add_patch(Circle((i,.9),.18,edgecolor='#9aa7b1',facecolor='white',lw=.9))
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
    renderers={f.__name__:f for f in [latent,sam,molmo,utonia,minimax,mova,rlsd,fuse,dinov3,dynamic,bptt]}
    renderers.update({kind:lambda mobile,k=kind:analytic(k,mobile) for kind in ['gspo','self','rae','mhc','long']})
    args=argparse.ArgumentParser();args.add_argument('--font-dir')
    args.add_argument('--only',nargs='+',choices=renderers)
    opts=args.parse_args();configure(opts.font_dir)
    for mobile in [False,True]:
        for name in opts.only or renderers:renderers[name](mobile)
    print('Rendered desktop and mobile SVGs for:', ', '.join(opts.only or renderers))



if __name__=='__main__':
    main()
