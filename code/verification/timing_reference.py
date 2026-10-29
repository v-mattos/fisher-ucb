"""Independent policy loop used by numerical verification."""
from math import ceil,log,sqrt
import numpy as np
from experiment_definitions import BETA_CERT
from fisher_ucb import kl_ucb_index_bounded
from policies import klucb_newton_scalar,klucb_newton_vec

def simulation(fam,T,seed,method,Q=None):
    """Independent scalar policy reference for timing-kernel checks."""
    rng=np.random.default_rng(seed);K=len(fam['means'])
    cnt=np.zeros(K,int);mh=np.zeros(K);ssq=np.zeros(K)
    gaps=max(fam['means'])-np.array(fam['means']);reg=np.zeros(T)
    forced=0;cost=0.
    for t in range(T):
        if method=='Fisher certified' and t<K:
            a=t;is_forced=True
        else:
            quota=ceil(Q*log(t+1)) if method=='Fisher certified' else 20
            under=np.flatnonzero(cnt<quota)
            is_forced=bool(under.size)
            if is_forced:a=int(under[0])
            else:
                L=log(t+1)
                if method=='KL bisection':idx=np.array([kl_ucb_index_bounded(mh[i],cnt[i],L,fam['kl']) for i in range(K)])
                elif method=='KL scalar Newton':idx=np.array([klucb_newton_scalar(mh[i],cnt[i],L) for i in range(K)])
                elif method=='KL vectorized Newton':idx=klucb_newton_vec(mh,cnt,L)
                elif method=='UCB-V':
                    var=np.maximum(ssq/cnt-mh**2,0.)
                    idx=mh+np.sqrt(2*var*L/cnt)+3*fam['upper']*L/cnt
                elif method in ['Fisher certified','Fisher matched','Fisher practical']:
                    beta=BETA_CERT if method=='Fisher certified' else sqrt(2) if method=='Fisher matched' else 1.25
                    idx=mh+beta*np.sqrt(L/(cnt*fam['fisher'](np.clip(mh,*fam['K']))))
                else:raise ValueError(method)
                a=int(np.argmax(idx))
        x=fam['sample'](rng,fam['means'][a]);cnt[a]+=1;mh[a]+=(x-mh[a])/cnt[a]
        if method=='UCB-V':ssq[a]+=x*x
        reg[t]=gaps[a]
        if is_forced:forced+=1;cost+=gaps[a]
    return dict(curve=np.cumsum(reg),forced_pulls=forced,forced_regret=cost)
