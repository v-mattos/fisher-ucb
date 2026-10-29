"""Numerical identities, trajectory regression and execution checks."""
import argparse
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runner import atomic_json, run_job
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import brentq
from verification import Kgap_c2 as gap
from verification import poisson_verify as pv
from verification import binom_verify as bv
from verification.beta0_certificate import check_bregman_lower_bound, check_hessian_sandwich
from certified_benchmark import FAMILIES, local_constants
from certification import certificate, computable_bernoulli_quota
from policies import newton_check
from fisher_ucb import run_fisher_ucb, BernoulliArm

BASE=Path(__file__).resolve().parent


def require(value,message):
    if not bool(value):raise AssertionError(message)
    print('PASS:',message,flush=True)


def verify():
    s,rho,rho4=sp.symbols('s rho rho4',real=True)
    require(sp.simplify(gap.derive_c2_symbolic()-(5*rho**2-3*rho4)/72)==0,
            'Symbolic geometric second-order coefficient')
    for f,mus in [(gap.bernoulli(),[.2,.5,.8]),(gap.poisson(),[1.,3.]),
                  (gap.binomial(20),[4.,12.,16.])]:
        require(gap.check_family(*f,mus=mus),'Geometric expansion: '+f[0])
        require(gap.check_operational(*f,mus=[mus[0],mus[-1]]),'Operational expansion in s: '+f[0])
    require(pv.check_divergence() and pv.check_skewness_selfconcordance() and pv.check_gap_c2(),
            'Poisson divergence, skewness, and coefficient identities')
    require(bv.check_IF_rho() and bv.check_kl() and bv.check_gap_both_regimes(),
            'Binomial variance, skewness, divergence, and sign reversal')
    for name,f in FAMILIES.items():
        require(check_bregman_lower_bound(f,.25)[0] and check_hessian_sandwich(f,.25),
                'Local Bregman and Hessian bounds: '+name)
        _,Q=local_constants(f,.25,2.2)
        require(certificate(f,2.2,.25,Q,f['K'])['certified'],'Valid certificate: '+name)
        require(not certificate(f,1.25,.25,Q,f['K'])['certified'],'Practical beta excluded: '+name)
        require(not certificate(f,2.2,.25,Q/2,f['K'])['certified'],'Insufficient quota excluded: '+name)
    ctr=dict(FAMILIES['bernoulli']);ctr['means']=[.005,.01,.02,.008]
    _,Q=local_constants(ctr,.25,2.2)
    require(not certificate(ctr,2.2,.25,Q,(.01,.99))['certified'],'Invalid CTR projection rejected')
    require(certificate(ctr,2.2,.25,Q,(.001,.999))['certified'],'Corrected CTR certificate accepted')
    require(np.isclose(computable_bernoulli_quota((.01,.99),(.005,.995))['Q_J'],27225),
            'Computable Q_J reproduces broad-interval example')
    # Check finite Hessian-based bounds directly.
    for x in [.1,.3,.5,.7,.9]:
        for D in [.005,.02,.05]:
            q=brentq(lambda q:gap.bernoulli()[1](x,q)-D,x,1.-1e-10)
            I=1/(q*(1-q));t=(q-x)*np.sqrt(I)
            C=max(abs((2*u-1)/np.sqrt(u*(1-u))) for u in [x,q]);tau=C*t/2
            if tau>=1:continue
            lower=quad(lambda u:(t-u)/(1+C*u/2)**2,0,t)[0]
            upper=quad(lambda u:(t-u)/(1-C*u/2)**2,0,t)[0]
            require(lower-1e-10<=D<=upper+1e-10,f'Finite quadratic bounds x={x}, D={D}')
    with tempfile.TemporaryDirectory(prefix='fucb-final-verification-') as tmp:
        require(newton_check(str(Path(tmp)/'newton.txt')),'Newton accuracy on 672 boundary/interior cases')
        with np.load(BASE/'trajectory_cases.npz',allow_pickle=False) as expected:
            for name,f in FAMILIES.items():
                from certified_benchmark import _Arm
                arms=[_Arm(f,m) for m in f['means']];_,Q=local_constants(f,.25,2.2)
                for seed in range(3):
                    for mode,beta,quota in [('practical',1.25,None),('certified',2.2,Q)]:
                        actual,forced,cost=run_fisher_ucb(arms,500,f['fisher'],f['K'],beta=beta,mode=mode,Q=quota,seed=seed,return_forced=True)
                        require(np.array_equal(expected[f'{name}_{mode}_{seed}'],actual),
                                f'Trajectory regression: {name}, {mode}, seed={seed}')
        arms=[BernoulliArm(.2),BernoulliArm(.4)]
        _,pulls,cost=run_fisher_ucb(arms,40,lambda m:1/(m*(1-m)),(.01,.99),return_forced=True)
        require(pulls==40 and np.isclose(cost,4.),'Initialization included in forced accounting')
        calls=[];spec=dict(experiment='test',T=5,policy='fixture')
        def execute(seed):calls.append(seed);return np.arange(5,dtype=float)
        records=run_job(tmp,spec,2,execute)
        run_job(tmp,spec,3,execute)
        require(calls==[0,1,2],'Resume adds only missing seeds')
        run_job(tmp,dict(spec,beta=2.2),1,execute)
        require(calls==[0,1,2,0],'Changed configuration uses a separate checkpoint')
        require(float(records[-1]['final'])==4.,'Checkpoint final regret survives serialization')
        from experiment_definitions import family,runtime_simulation
        from experiment_definitions import simulate
        from certified_benchmark import _Arm
        g=family('gaussian');arms=[_Arm(g,m) for m in g['means']]
        for seed in range(3):
            a=simulate(g,'KL',300,seed)
            b=run_fisher_ucb(arms,300,g['fisher'],g['K'],beta=np.sqrt(2),warmup=1,seed=seed)
            require(np.array_equal(a,b),'Gaussian exact-radius control with matched initialization')
        f=family('bernoulli',[.5,.2,.3,.45])
        a=runtime_simulation(f,300,0,'KL bisection')
        for method in ['KL scalar Newton','KL vectorized Newton']:
            require(np.array_equal(a,runtime_simulation(f,300,0,method)),
                    'Matched runtime trajectories: '+method)
    return dict(passed=True,scope='Numerical and pipeline checks; not a proof',checks_version=1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=verify()
    if args.output:atomic_json(args.output,result)
    print('All final-pipeline checks passed.')

if __name__=='__main__':main()
