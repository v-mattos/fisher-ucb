"""Publication experiment definitions and numerical policies."""
from types import SimpleNamespace
from math import log, sqrt
import numpy as np
from fisher_ucb import (run_fisher_ucb, run_ucb1, run_kl_ucb,
                       kl_ucb_index_bounded, fisher_gaussian)
from certified_benchmark import FAMILIES, _Arm, local_constants
from fisher_binomial import make_fisher_binomial, make_kl_binomial
from policies import (run_klucb_newton, run_ucbv, run_endpoint, _loop,
                                  klucb_newton_scalar, klucb_newton_vec)
from certification import certificate, computable_bernoulli_quota
from runner import run_job, summarize, write_csv, atomic_json, environment

BETA_CERT = 2.2
GAMMA = .25


def family(name, means=None, projection=None):
    if name == 'gaussian':
        f = dict(means=[0.,.2,.5,.1], K=(-1e6,1e6), domain=(-np.inf,np.inf),
                 fisher=fisher_gaussian, rho=lambda u: 0.,
                 kl=lambda a,b: .5*(a-b)**2, bounded=False, upper=None,
                 sample=lambda rng,m: float(rng.normal(m,1.)))
    elif name == 'binomial':
        f = dict(means=[.8,1.5,2.2,1.2], K=(.05,9.95), domain=(0.,10.),
                 fisher=make_fisher_binomial(10), kl=make_kl_binomial(10),
                 rho=lambda m: (2*m-10)/sqrt(m*10*(10-m)), bounded=True,upper=10.,
                 sample=lambda rng,m: float(rng.binomial(10,m/10)))
    else: f = dict(FAMILIES[name])
    if means is not None: f['means'] = list(means)
    if projection is not None: f['K'] = tuple(projection)
    f['name'] = name
    return f


def config(experiment, fam, T, policy, beta=None, Q=None):
    is_fisher = policy.startswith('Fisher')
    quota_mode = is_fisher and Q is not None
    if policy.startswith('KL'):
        if 'Newton' in policy:
            solver, tolerance, iterations = 'newton_with_bisection_fallback', 1e-10, 60
        elif fam['name'] == 'gaussian':
            solver, tolerance, iterations = 'analytic_gaussian', None, 0
        else:
            solver, tolerance, iterations = 'bisection', 1e-6, 50
    else:
        solver, tolerance, iterations = None, None, 0
    return dict(experiment=experiment, family=fam['name'], means=fam['means'],
                projection=list(fam['K']), T=T, policy=policy, beta=beta,
                Q=Q, gamma=GAMMA, initialization_pulls_per_arm=1,
                warmup=20 if is_fisher and not quota_mode else 1,
                exploration='logarithmic_quota' if quota_mode else
                            'fixed_warmup' if is_fisher else 'single_pull_initialization',
                rng='sequential default_rng(seed)',
                forced_selection='first eligible arm',
                kl_solver=solver, kl_tolerance=tolerance,
                kl_max_iterations=iterations, exact_reference_rewards=True)


def simulate(fam, policy, T, seed, beta=1.25, Q=None, horizons=()):
    arms = [_Arm(fam,m) for m in fam['means']]
    if policy.startswith('Fisher'):
        if policy == 'Fisher certified':
            check = certificate(fam,beta,GAMMA,Q,fam['K'])
            if not check['certified']: raise ValueError(f'Invalid certified configuration: {check}')
        return run_fisher_ucb(arms,T,fam['fisher'],fam['K'],beta=beta,seed=seed,
                              mode='certified' if Q is not None else 'practical',Q=Q,
                              return_forced=True, snapshot_horizons=horizons,
                              return_details=bool(horizons))
    if policy == 'UCB1': return run_ucb1(arms,T,seed=seed)
    if policy == 'UCB-V':
        if not fam['bounded']: raise ValueError('UCB-V requires a known finite reward range')
        return run_ucbv(arms,T,seed=seed,b=fam['upper'])
    if policy == 'Endpoint':
        if fam['name'] != 'bernoulli': raise ValueError('Final Endpoint experiment is Bernoulli')
        return run_endpoint(arms,T,seed=seed)
    if policy == 'KL Newton': return run_klucb_newton(arms,T,seed=seed)
    if fam['name'] == 'gaussian':
        # The positive-mean unbounded bracket is not valid on the Gaussian domain.
        return _loop(arms,T,seed,lambda t,c,m,ss:int(np.argmax(m+np.sqrt(2*log(t+1)/c))))
    return run_kl_ucb(arms,T,fam['kl'],bounded=fam['bounded'],
                      upper_bound=fam['upper'] or 1.,seed=seed,
                      snapshot_horizons=horizons,return_details=bool(horizons))




def batch(args, experiment, fam, policies, seeds=None, T=None, horizons=(), curves=True):
    T = T or args.T; seeds = seeds or args.seeds
    rows, series, per_seed, configurations = [],[],[],[]
    for policy, beta, Q in policies:
        spec=config(experiment,fam,T,policy,beta,Q)
        spec.update(snapshot_horizons=list(horizons),keep_curve=bool(curves))
        configurations.append(spec)
        records=run_job(args.checkpoints,spec,seeds,
                       lambda seed: simulate(fam,policy,T,seed,beta or 1.25,Q,horizons),
                       resume=not args.no_resume,keep_curve=curves,horizons=horizons)
        check=certificate(fam,beta,GAMMA,Q,fam['K']) if beta else None
        label=policy+(f' beta={beta:.12g}' if beta else '')+(f' Q={Q:.12g}' if Q else '')
        row=dict(policy=label,certified=bool(check and check['certified']),
                 beta=beta,Q=Q,projection=str(fam['K']),**summarize(records))
        rows.append(row);series.append((label,records))
        per_seed += [dict(policy=label,seed=i,regret=float(r['final']),
                          forced_pulls=int(r['forced_pulls']),forced_regret=float(r['forced_regret']))
                     for i,r in enumerate(records)]
    kl_mean=next((r['mean'] for r in rows if r['policy'].startswith('KL')),None)
    for row,(_,records) in zip(rows,series):
        row['threshold_exceedance_fraction']=(float(np.mean([r['final']>10*kl_mean for r in records]))
                                             if kl_mean is not None else '')
    write_csv(args.output/'results'/f'{experiment}.csv',rows)
    write_csv(args.output/'results'/f'{experiment}_per_seed.csv',per_seed)
    atomic_json(args.output/'results'/f'{experiment}_metadata.json',
                dict(environment=environment(),seeds=seeds,configurations=configurations))
    return series


def main_benchmark(args):
    for name in ['bernoulli','exponential','poisson']:
        fam=family(name); _,Q=local_constants(fam,GAMMA,BETA_CERT)
        batch(args,f'main_{name}',fam,
                     [('UCB1',None,None),('KL',None,None),
                      ('Fisher practical',1.25,None),('Fisher matched',sqrt(2),None),
                      ('Fisher certified',BETA_CERT,Q)])
    batch(args,'poisson_replication',family('poisson'),
          [('UCB1',None,None),('KL',None,None),('Fisher practical',1.25,None),
           ('Fisher matched',sqrt(2),None)],seeds=args.poisson_seeds,curves=False)


def poisson_certified(args):
    fam=family('poisson');_,Q=local_constants(fam,GAMMA,BETA_CERT)
    output=args.output.parent.parent/'poisson_certified'
    settings=SimpleNamespace(**vars(args))
    settings.output=output;settings.checkpoints=output/'checkpoints'
    batch(settings,'poisson_certified_replication',fam,
          [('Fisher certified',BETA_CERT,Q)],seeds=args.poisson_seeds,curves=False)


def boundary(args):
    for name,means in [('nearzero',[.02,.05,.08,.03]),('ctr',[.005,.010,.020,.008])]:
        fam=family('bernoulli',means); _,Q=local_constants(fam,GAMMA,BETA_CERT)
        batch(args,f'boundary_{name}_paper',fam,
                   [('KL Newton',None,None),('UCB-V',None,None),('Endpoint',None,None),
                    ('Fisher practical',1.25,None),('Fisher matched',sqrt(2),None)],
                   seeds=args.boundary_seeds)
        wide=family('bernoulli',means,(.001,.999))
        batch(args,f'boundary_{name}_wide',wide,
                         [('KL Newton',None,None),('Fisher practical',1.25,None),
                          ('Fisher matched',sqrt(2),None),('Fisher certified',BETA_CERT,Q)],
                         seeds=args.boundary_seeds)


def binomial(args):
    for name,means in [('favourable',[.8,1.5,2.2,1.2]),('adverse',[9.2,8.5,7.8,8.8])]:
        batch(args,f'binomial_{name}',family('binomial',means),
                     [('UCB1',None,None),('UCB-V',None,None),('KL',None,None),
                      ('Fisher practical',1.25,None),('Fisher matched',sqrt(2),None)],
                     seeds=args.binomial_seeds)


def poisson_projection(args):
    batch(args,'poisson_curves',family('poisson'),
                [('UCB1',None,None),('KL',None,None),('Fisher practical',1.25,None)],
                seeds=args.poisson_curve_seeds)
    means=[.10,.15,.20,.08]
    for name,K in [('projected',(.05,1e6)),('minimal_clip',(1e-9,1e9))]:
        batch(args,f'poisson_boundary_{name}',family('poisson',means,K),
                     [('Fisher practical',1.25,None)],T=args.poisson_boundary_T)


def constants(args):
    data={}
    for name in ['bernoulli','exponential','poisson']:
        f=family(name);_,Q=local_constants(f,GAMMA,BETA_CERT)
        data[name]=certificate(f,BETA_CERT,GAMMA,Q,f['K'])
    data['computable_bernoulli']=[computable_bernoulli_quota(K,J) for K,J in
                                [((.01,.99),(.005,.995)),((.2,.5),(.1,.6)),((.25,.45),(.2,.5))]]
    atomic_json(args.output/'results/constants.json',data)


def radius_diagnostic(fam, x, n, t, beta=1.25):
    """Same-state Bernoulli comparison, with explicit local-domain validity."""
    from scipy.optimize import brentq
    if not 0 <= x <= 1 or n < 0 or t < 2:
        raise ValueError('Invalid Bernoulli diagnostic state')
    if n == 0:
        return np.nan, np.nan
    budget=log(t)/n
    if x == 0:
        q=-np.expm1(-budget)
    elif x == 1:
        q=1.
    else:
        q=1. if fam['kl'](x,1.-1e-12)<=budget else brentq(
            lambda q: fam['kl'](x,q)-budget,x,1.-1e-12,xtol=1e-14)
    projected=float(np.clip(x,*fam['K']))
    practical=beta*sqrt(budget/fam['fisher'](projected))
    matched=sqrt(2*budget*x*(1-x))
    return ((q-x)/matched if 0<x<1 else np.nan, (q-x)/practical)


def horizon(args):
    hs=tuple(args.horizons);fam=family('bernoulli');_,Q=local_constants(fam,GAMMA,BETA_CERT)
    series=batch(args,'horizon',fam,
                 [('KL',None,None),('Fisher practical',1.25,None),
                  ('Fisher high_beta',BETA_CERT,None),('Fisher certified',BETA_CERT,Q)],
                 T=max(hs),horizons=hs)
    rows=[]
    for label,records in series:
        for j,h in enumerate(hs):
            vals=np.array([r['regret_at_horizons'][j] for r in records])
            rows.append(dict(policy=label,T=h,seeds=len(vals),mean=float(vals.mean()),
                             ci95=float(1.96*vals.std(ddof=1)/sqrt(len(vals))) if len(vals)>1 else 0.,
                             per_logT=float(vals.mean()/log(h))))
    write_csv(args.output/'results/horizon_scaling.csv',rows)
    # Compare the two radii on the SAME Fisher empirical state, at matched level.
    f_records=series[1][1];ratio=[]
    for r in f_records:
        values=[]
        for j,h in enumerate(hs):
            a=int(np.argmax(fam['means']));x=float(r['means'][j,a]);n=int(r['counts'][j,a])
            values.append(radius_diagnostic(fam,x,n,h))
        ratio.append(values)
    ratio=np.array(ratio);rr=[]
    for j,h in enumerate(hs):
        matched=ratio[:,j,0];practical=ratio[:,j,1]
        mvalid=matched[np.isfinite(matched)];pvalid=practical[np.isfinite(practical)]
        rr.append(dict(T=h,matched_ratio_mean=float(mvalid.mean()) if len(mvalid) else '',
                       practical_ratio_mean=float(pvalid.mean()) if len(pvalid) else '',
                       matched_valid_seeds=len(mvalid),practical_valid_seeds=len(pvalid)))
    write_csv(args.output/'results/radius_ratios.csv',rr)

def many_arms(args):
    for name,best,low,high in [('bernoulli',.8,.1,.7),('exponential',5.,.5,4.)]:
        f=family(name,[best]+list(np.linspace(low,high,99)))
        batch(args,f'many_arms_{name}',f,
                     [('UCB1',None,None),('KL',None,None),('Fisher practical',1.25,None)])


def runtime_simulation(fam,T,seed,method):
    """Same Bernoulli loop, RNG, warm-up=20, and log(t) confidence for all solvers."""
    rng=np.random.default_rng(seed);K=len(fam['means']);cnt=np.zeros(K,int);mh=np.zeros(K)
    gaps=max(fam['means'])-np.array(fam['means']);reg=np.zeros(T)
    for t in range(T):
        under=np.flatnonzero(cnt<20)
        if under.size:a=int(under[0])
        else:
            L=log(t+1)
            if method=='KL bisection':
                idx=np.array([kl_ucb_index_bounded(mh[i],cnt[i],L,fam['kl']) for i in range(K)])
            elif method=='KL scalar Newton':idx=np.array([klucb_newton_scalar(mh[i],cnt[i],L) for i in range(K)])
            elif method=='KL vectorized Newton':idx=klucb_newton_vec(mh,cnt,L)
            else:
                beta=sqrt(2) if method=='Fisher matched' else 1.25
                idx=mh+beta*np.sqrt(L/(cnt*fam['fisher'](np.clip(mh,*fam['K']))))
            a=int(np.argmax(idx))
        x=fam['sample'](rng,fam['means'][a]);cnt[a]+=1;mh[a]+=(x-mh[a])/cnt[a];reg[t]=gaps[a]
    return np.cumsum(reg)
