"""Fisher-UCB numerical routines and checks."""
from math import sqrt
import numpy as np
from certified_benchmark import local_constants

def check_bregman_lower_bound(fam, gamma, ngrid=400):
    rows, _ = local_constants(fam, gamma, 2.2)
    ok = True
    worst = np.inf
    for r in rows:
        mu, I, theta = (r['mu'], r['I_F'], r['theta_b'])
        for sgn in (+1, -1):
            for h in np.linspace(1e-06, theta, ngrid):
                u = mu + sgn * h / sqrt(I)
                lo, hi = fam['domain']
                if not lo < u < hi:
                    continue
                D = fam['kl'](u, mu)
                ratio = D / (h * h / (2 * (1 + gamma)))
                worst = min(worst, ratio)
                ok &= ratio >= 1 - 1e-09
    return (ok, worst)

def check_hessian_sandwich(fam, gamma, ngrid=400):
    rows, _ = local_constants(fam, gamma, 2.2)
    ok = True
    for r in rows:
        mu, I, theta = (r['mu'], r['I_F'], r['theta_b'])
        for u in np.linspace(mu - theta / sqrt(I), mu + theta / sqrt(I), ngrid):
            lo, hi = fam['domain']
            if not lo < u < hi:
                continue
            Iu = fam['fisher'](u)
            ok &= I / (1 + gamma) ** 2 - 1e-12 <= Iu <= I / (1 - gamma) ** 2 + 1e-12
    return ok
