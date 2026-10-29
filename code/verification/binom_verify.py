"""Fisher-UCB numerical routines and checks."""
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
N = 20

def IF(mu, n=N):
    return n / (mu * (n - mu))

def rho(mu, n=N):
    return (2 * mu - n) / np.sqrt(mu * n * (n - mu))

def rho4(mu, n=N):
    return 2 * (mu * (mu - n) - (2 * mu - n) ** 2) / (mu * n * (mu - n))

def kl_binom(mu_a, mu_b, n=N):
    pa, pb = (mu_a / n, mu_b / n)
    e = 1e-12
    pa = min(max(pa, e), 1 - e)
    pb = min(max(pb, e), 1 - e)
    return n * (pa * np.log(pa / pb) + (1 - pa) * np.log((1 - pa) / (1 - pb)))

def check_IF_rho():
    ok = True
    for p in [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]:
        mu = N * p
        var = binom.var(N, p)
        skew = binom.stats(N, p, moments='s')
        m_if = np.isclose(1 / var, IF(mu))
        m_rho = np.isclose(-skew, rho(mu))
        sign = 'rho<0' if rho(mu) < 0 else 'rho=0' if abs(rho(mu)) < 1e-09 else 'rho>0 ADVERSO'
        ok = ok and m_if and m_rho
    return ok

def check_kl():
    ok = True
    for mu_a in [4.0, 10.0, 16.0]:
        for e in [0.5, 1.0]:
            d1 = kl_binom(mu_a, mu_a + e)
            pa, pb = (mu_a / N, (mu_a + e) / N)
            d2 = N * (pa * np.log(pa / pb) + (1 - pa) * np.log((1 - pa) / (1 - pb)))
            match = np.isclose(d1, d2)
            ok = ok and match
    return ok

def check_gap_both_regimes():
    ok = True
    for mu_a, regime in [(4.0, 'rho<0'), (16.0, 'rho>0 ADVERSO')]:
        last_ratio = c2pred = r_n = None
        for n_pulls in [10000.0, 100000.0, 1000000.0]:
            D = 1.0 / n_pulls
            hi = N - mu_a - 1e-09
            eps = brentq(lambda e: kl_binom(mu_a, mu_a + e) - D, 1e-13, hi)
            q = mu_a + eps
            s = np.sqrt(2 * D)
            t = eps * np.sqrt(IF(q))
            r_n = t / s
            ratio = (r_n - (1 + rho(q) * s / 6)) / s ** 2
            c2pred = (5 * rho(q) ** 2 - 3 * rho4(q)) / 72
            last_ratio = ratio
        m_c2 = abs(last_ratio - c2pred) < 0.005
        m_sign = (r_n < 1) == (rho(mu_a) < 0)
        ok = ok and m_c2 and m_sign
    return ok
