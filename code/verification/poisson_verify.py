"""Fisher-UCB numerical routines and checks."""
import numpy as np
from scipy.optimize import brentq

def phi(mu):
    return mu * np.log(mu) - mu

def Dphi(x, y):
    return phi(x) - phi(y) - np.log(y) * (x - y)

def kl_poisson(a, b):
    return a * np.log(a / b) - a + b

def IF(mu):
    return 1.0 / mu

def rho(mu):
    return -1.0 / np.sqrt(mu)

def rho4(mu):
    return 2.0 / mu

def check_divergence(tol=1e-12):
    ok = True
    for a in [0.3, 1.0, 2.0, 5.0]:
        for e in [0.05, 0.2, 0.5]:
            b = a + e
            d1, d2 = (Dphi(a, b), kl_poisson(a, b))
            match = abs(d1 - d2) < tol
            ok = ok and match
    return ok

def check_skewness_selfconcordance():
    ok = True
    for mu in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        skew = 1.0 / np.sqrt(mu)
        match_skew = abs(rho(mu) - -skew) < 1e-12
        sc = abs(rho(mu)) <= 2.0 + 1e-12
        expected_sc = mu >= 0.25 - 1e-12
        match_sc = sc == expected_sc
        ok = ok and match_skew and match_sc
    return ok

def check_gap_c2():
    ok = True
    for mu_a in [1.0, 3.0]:
        last_ratio, c2pred = (None, None)
        for n in [10000.0, 100000.0, 1000000.0, 10000000.0]:
            D = 1.0 / n
            eps = brentq(lambda e: kl_poisson(mu_a, mu_a + e) - D, 1e-14, 50.0)
            q = mu_a + eps
            s = np.sqrt(2 * D)
            t = eps * np.sqrt(IF(q))
            r_n = t / s
            R = r_n - (1.0 + rho(q) * s / 6.0)
            ratio = R / s ** 2
            c2pred = (5 * rho(q) ** 2 - 3 * rho4(q)) / 72.0
            last_ratio = ratio
        match = abs(last_ratio - c2pred) < 0.005
        ok = ok and match
    return ok
