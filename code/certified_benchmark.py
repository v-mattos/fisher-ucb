"""Fisher-UCB numerical routines and checks."""
from math import log, sqrt
import numpy as np
FAMILIES = {'bernoulli': dict(means=[0.3, 0.35, 0.4, 0.32], K=(0.01, 0.99), domain=(0.0, 1.0), fisher=lambda u: 1.0 / (u * (1.0 - u)), rho=lambda u: (2.0 * u - 1.0) / sqrt(u * (1.0 - u)), kl=lambda p, q: _kl_bern(p, q), bounded=True, upper=1.0, sample=lambda rng, m: float(rng.binomial(1, m))), 'exponential': dict(means=[1.0, 1.5, 2.0, 0.7], K=(0.05, 1000000.0), domain=(0.0, np.inf), fisher=lambda u: 1.0 / (u * u), rho=lambda u: -2.0, kl=lambda a, b: log(b / a) + a / b - 1.0, bounded=False, upper=None, sample=lambda rng, m: float(rng.exponential(m))), 'poisson': dict(means=[2.0, 2.1, 2.2, 1.9], K=(0.05, 1000000.0), domain=(0.0, np.inf), fisher=lambda u: 1.0 / u, rho=lambda u: -1.0 / sqrt(u), kl=lambda a, b: a * log(a / b) - a + b, bounded=False, upper=None, sample=lambda rng, m: float(rng.poisson(m)))}

def _kl_bern(p, q):
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))

def local_constants(fam, gamma, beta):
    lo, hi = fam['domain']
    rows = []
    for mu in fam['means']:
        I = fam['fisher'](mu)
        s = sqrt(I)
        cands = [abs(mu - lo) if np.isfinite(lo) else np.inf, abs(hi - mu) if np.isfinite(hi) else np.inf]
        d_b = min(cands) * s
        r_b = min(0.5, d_b / 2.0)
        a, b = (mu - r_b / s, mu + r_b / s)
        # For the supported Bernoulli, Exponential, and Poisson references,
        # the maximum absolute skewness on a local interval is at an endpoint.
        C_b = max(abs(fam['rho'](a)), abs(fam['rho'](b)))
        theta = min(r_b, 2.0 * gamma / C_b) if C_b > 0 else r_b
        rows.append(dict(mu=mu, I_F=I, d_b=d_b, r_b=r_b, C_b=C_b, theta_b=theta))
    Q_star = beta ** 2 * (1 - gamma) ** 2 * max((1.0 / r['theta_b'] ** 2 for r in rows))
    return (rows, Q_star)

def beta0(gamma):
    return sqrt(2.0 * (1.0 + gamma)) / (1.0 - gamma)

class _Arm:

    def __init__(self, fam, m):
        self._fam, self._m = (fam, m)

    def sample(self, rng):
        return self._fam['sample'](rng, self._m)

    def mean(self):
        return self._m
