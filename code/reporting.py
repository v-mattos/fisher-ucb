"""Build the article's four tables and six figures from validated records."""
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from math import sqrt, log
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy, xlog1py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evidence import read_csv, stats, require

POLICIES=[('UCB1','UCB1','-.','tab:blue'),('KL','KL-UCB','--','tab:orange'),
          ('Fisher practical','Fisher-UCB','-','tab:green')]


def write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def latex(path, headings, rows, alignment=None):
    text=['\\begin{tabular}{'+(alignment or 'l'+'r'*(len(headings)-1))+'}', '\\toprule',
          ' & '.join(headings)+r' \\', '\\midrule']
    text += [r'\midrule' if row is None else ' & '.join(row)+r' \\' for row in rows]
    text+=['\\bottomrule','\\end{tabular}']
    Path(path).write_text('\n'.join(text)+'\n')


def display_number(value, digits=1):
    """Decimal display rounding, excluding sub-micro-unit accumulation noise."""
    value = Decimal(str(round(float(value), 6)))
    return f'{value.quantize(Decimal(10)**-digits, rounding=ROUND_HALF_UP):.{digits}f}'


def confidence_cell(s, bold=False):
    mean = display_number(s['mean'])
    if bold:
        mean = r'\mathbf{' + mean + '}'
    return '$' + mean + r'\,(\pm' + display_number(s['ci95']) + ')$'


def tables(e, destination):
    destination.mkdir(parents=True,exist_ok=True)
    values={};table1=[]
    labels={'UCB1':'UCB1','KL':'KL-UCB','Fisher practical':'Fisher-UCB',
            'Fisher matched':'Fisher-UCB, matched','Fisher certified':'Fisher-UCB, certified'}
    policies=list(labels)
    groups={'Bernoulli':'main_bernoulli','Exponential':'main_exponential',
            'Poisson':'main_poisson','Poisson1000':'poisson_replication'}
    summaries={}
    for column,group in groups.items():
        for policy in policies:
            if column=='Poisson1000' and policy=='Fisher certified':
                data=read_csv(e.root/'poisson_certified/results/poisson_certified_replication_per_seed.csv')
                summaries[column,policy]=stats([float(r['regret']) for r in data])
            else:summaries[column,policy]=e.summary(group,policy)
    best={column:min(policies,key=lambda policy:summaries[column,policy]['mean']) for column in groups}
    for policy in policies:
        row={'algorithm':labels[policy]}
        tex=[labels[policy] + (r' ($\beta=\sqrt2$)' if policy=='Fisher matched' else '')]
        for column,group in [('Bernoulli','main_bernoulli'),('Exponential','main_exponential'),
                             ('Poisson','main_poisson'),('Poisson1000','poisson_replication')]:
            s=summaries[column,policy]
            values[f'table1/{column}/{policy}']=s
            row[column+'_mean']=s['mean'];row[column+'_ci95']=s['ci95'];row[column+'_seeds']=s['seeds']
            tex.append(confidence_cell(s, bold=policy==best[column]))
        table1.append((row,tex))
    write_csv(destination/'table1.csv',[r for r,_ in table1])
    latex(destination/'table1.tex',['Algorithm','Bernoulli','Exponential','Poisson','Poisson ($10^3$ seeds)'],[r for _,r in table1], 'lcccc')
    table2=[]
    for policy in ['UCB1','UCB-V','KL','Fisher practical']:
        row={'algorithm':labels.get(policy,policy)};tex=[row['algorithm']]
        for title,group in [('favourable','binomial_favourable'),('adverse','binomial_adverse')]:
            s=e.summary(group,policy);values[f'table2/{title}/{policy}']=s
            row[title+'_mean']=s['mean'];row[title+'_ci95']=s['ci95']
            tex.append(confidence_cell(s))
        table2.append((row,tex))
    write_csv(destination/'table2.csv',[r for r,_ in table2])
    latex(destination/'table2.tex',['Algorithm',r'Favourable ($\rho<0$)',r'Adverse ($\rho>0$)'],[r for _,r in table2], 'lcc')
    selected=[('near-zero','boundary_nearzero_paper','KL Newton'),
              ('near-zero','boundary_nearzero_paper','Fisher practical'),
              ('near-zero','boundary_nearzero_paper','Fisher matched'),
              ('near-zero','boundary_nearzero_wide','Fisher practical'),
              ('near-zero','boundary_nearzero_wide','Fisher matched'),
              ('near-zero','boundary_nearzero_wide','Fisher certified'),
              ('CTR','boundary_ctr_paper','KL Newton'),('CTR','boundary_ctr_paper','Fisher practical'),
              ('CTR','boundary_ctr_wide','Fisher practical'),('CTR','boundary_ctr_wide','Fisher matched')]
    table3=[];tex=[]
    for instance,group,policy in selected:
        s=e.summary(group,policy)
        base='boundary_nearzero_paper' if instance=='near-zero' else 'boundary_ctr_paper'
        threshold=10*e.summary(base,'KL Newton')['mean']
        flagged=sum(float(r['regret'])>threshold for r in e.observations(group,policy))
        row=dict(instance=instance,group=group,policy=policy,**s,flagged=flagged,threshold=threshold)
        values[f'table3/{group}/{policy}']=row;table3.append(row)
        # Full-precision statistics are provided in the CSV.
        integer_tail = instance=='near-zero' and policy!='KL Newton' or group=='boundary_ctr_wide'
        max_digits = 0 if instance=='near-zero' else 1
        if group=='boundary_ctr_wide':max_digits=0
        tail=['---']*4 if policy=='Fisher certified' else [display_number(s['median']),
            display_number(s['p95'], 0 if integer_tail else 1),
            display_number(s['max'], max_digits),f"{flagged}/{s['seeds']}"]
        config=e.configuration(group,policy)
        beta = r'\sqrt2' if policy=='Fisher matched' else '1.25'
        projection = '[.01,.99]' if config['projection'][0]==.01 else '[.001,.999]'
        label='KL-UCB (Newton)' if policy=='KL Newton' else (
            r'Fisher certified ($Q^\star\approx'+display_number(config['Q'],0)+'$)') if policy=='Fisher certified' else (
            f'Fisher $\\beta={beta}$, ${projection}$')
        if instance=='CTR' and (not tex or tex[-1][0]!='CTR'):tex.append(None)
        tex.append([instance,label,display_number(s['mean'],0 if policy=='Fisher certified' else 1),*tail])
    write_csv(destination/'table3.csv',table3)
    latex(destination/'table3.tex',['Instance','Algorithm','mean','median','p95','max','flagged'],tex,'llrrrrr')
    runtime=read_csv(e.root/'runtime/runtime.csv');write_csv(destination/'table4.csv',runtime)
    methods=list(dict.fromkeys(r['method'] for r in runtime));ks=sorted(set(int(r['K']) for r in runtime))
    lookup={(int(r['K']),r['method']):float(r['seconds_median']) for r in runtime}
    method_labels={'KL bisection':'KL-UCB (bisection)','KL scalar Newton':'KL-UCB (scalar Newton)',
                   'KL vectorized Newton':'KL-UCB (vectorized Newton)',
                   'Fisher practical':'Fisher-UCB (practical)','Fisher matched':'Fisher-UCB (matched)',
                   'Fisher certified':'Fisher-UCB (certified)','UCB-V':'UCB-V'}
    latex(destination/'table4.tex',['Method',*[f'$K={k}$' for k in ks]],
          [[method_labels[method],*[display_number(lookup[k,method],3) for k in ks]] for method in methods])
    values['table4']=runtime
    (destination/'article_values.json').write_text(json.dumps(values,indent=2)+'\n')
    return values


def curves(e, group, policy):
    records=e.records(group,policy)
    grid=records[0]['rounds']
    require(all(np.array_equal(r['rounds'],grid) for r in records), f'Inconsistent display grids: {group}')
    return grid,np.asarray([r['curve'] for r in records])


def figures(e, destination):
    with plt.style.context('default'):
        return _figures(e, destination)


def _figures(e, destination):
    destination.mkdir(parents=True,exist_ok=True);evidence={}
    panels={1:[('Bernoulli','main_bernoulli'),('Exponential','main_exponential')],
            3:[(r'Binomial favorable ($\rho<0$, $p<1/2$)','binomial_favourable'),
               (r'Binomial adverse ($\rho>0$, $p>1/2$)','binomial_adverse')],
            6:[('Bernoulli (K=100)','many_arms_bernoulli'),('Exponential (K=100)','many_arms_exponential')]}
    names={1:'regret_curves.pdf',2:'poisson_curves.pdf',3:'binomial_curves_n100.pdf',
           4:'horizon_scaling.pdf',5:'scaling_runtime_combined.pdf',6:'scaling_regret_k100.pdf'}
    def band(ax,group,policy,label,style='-',color=None,alpha=.15,edge=True):
        x,y=curves(e,group,policy);mean=y.mean(0);sd=y.std(0)
        line,=ax.plot(x,mean,style,label=label,color=color,lw=1.5)
        ax.fill_between(x,np.maximum(mean-sd,0),mean+sd,color=line.get_color(),alpha=alpha,
                        **({} if edge else {'linewidth':0}))
        return dict(group=group,policy=policy,seeds=len(y),rounds=x.tolist(),mean=mean.tolist(),sd=sd.tolist())
    for number,groups in panels.items():
        fig,axes=plt.subplots(1,2,figsize=(11,4));proof=[]
        for ax,(title,group) in zip(axes,groups):
            for policy,label,style,color in POLICIES:proof.append(band(ax,group,policy,label,style,color))
            ax.set(xlabel='round t',ylabel='cumulative regret')
            ax.set_title(title,fontsize=10);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(destination/names[number]);plt.close(fig);evidence[str(number)]=proof
    fig,axes=plt.subplots(1,2,figsize=(12,4));proof=[]
    for policy,label,style,color in POLICIES:proof.append(band(axes[0],'poisson_curves',policy,label,'-',color,alpha=.12,edge=False))
    for group,label in [('poisson_boundary_projected','Projected'),('poisson_boundary_minimal_clip','Minimal clip')]:
        proof.append(band(axes[1],group,'Fisher practical',label,alpha=.12,edge=False))
    for ax,title in zip(axes,['Poisson, small gaps','Poisson, near zero']):
        ax.set(title=title,xlabel='Round',ylabel='Cumulative pseudo-regret');ax.legend(fontsize=8*12/11);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(destination/names[2]);plt.close(fig);evidence['2']=proof
    kl=e.records('horizon','KL');f=e.records('horizon','Fisher practical');hs=f[0]['horizons']
    require(all(np.array_equal(z['horizons'],hs) for z in kl+f), 'Horizon grids differ')
    differences=np.array([a['regret_at_horizons']-b['regret_at_horizons'] for a,b in zip(f,kl)])
    ratios=[]
    for z in f:
        row=[]
        for j,t in enumerate(hs):
            x=float(z['means'][j,2]);n=int(z['counts'][j,2]);budget=log(int(t))/n
            def divergence(q):return float(xlogy(x,x/q)+xlog1py(1-x,-x)-xlog1py(1-x,-q))
            q=brentq(lambda q:divergence(q)-budget,x,np.nextafter(1.,0.),xtol=1e-14)
            width=1.25*sqrt(max(.01,min(.99,x))*(1-max(.01,min(.99,x)))*budget)
            row.append((q-x)/width)
        ratios.append(row)
    ratios=np.asarray(ratios);means=differences.mean(0)
    ci=1.96*differences.std(0,ddof=1)/sqrt(len(f)) if len(f)>1 else np.zeros(len(hs))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].errorbar(hs,means,yerr=ci,marker='o')
    axes[0].set(xscale='log',xlabel='Horizon',ylabel='Fisher practical minus KL regret')
    axes[1].plot(hs,ratios.mean(0),'o-',label='Same-state practical radius ratio')
    reference=sqrt(2)/1.25
    axes[1].axhline(reference,ls='--',color='black',label=rf'$\sqrt{{2}}/\beta\approx{reference:.4f}$')
    axes[1].set(xscale='log',xlabel='Horizon',ylabel='KL radius / practical Fisher radius')
    axes[1].set_ylim(min(reference,ratios.mean(0).min())-.0015,ratios.mean(0).max()+.0005)
    for ax in axes:ax.grid(which='both',ls='--',alpha=.3)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(destination/names[4]);plt.close(fig)
    evidence['4']=dict(horizons=hs.tolist(),paired_mean=means.tolist(),paired_ci95=ci.tolist(),
                       mean_radius_ratio=ratios.mean(0).tolist(),radius_ratios=ratios.tolist(),reference=reference)
    timing=read_csv(e.root/'runtime/runtime.csv');fig,ax=plt.subplots(figsize=(8,4))
    for method in dict.fromkeys(r['method'] for r in timing):
        rows=sorted([r for r in timing if r['method']==method],key=lambda r:int(r['K']))
        ax.plot([int(r['K']) for r in rows],[float(r['seconds_median']) for r in rows],'o-',label=method)
    ax.set(xlabel='Arms',ylabel='Full simulation time (median seconds)',yscale='log');ax.grid(which='both',ls='--',alpha=.3);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(destination/names[5]);plt.close(fig);evidence['5']=timing
    (destination/'plotted_values.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence
