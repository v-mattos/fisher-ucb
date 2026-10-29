"""Fisher-UCB numerical routines and checks."""
import numpy as np
import sympy as sp
from scipy.optimize import brentq

def derive_c2_symbolic():
    s, rho, rho4 = sp.symbols('s rho rho4', real=True)
    a2, a3 = sp.symbols('a2 a3', real=True)
    tt = s + a2 * s ** 2 + a3 * s ** 3
    D = sp.Rational(1, 2) * tt ** 2 - sp.Rational(1, 6) * rho * tt ** 3 + sp.Rational(1, 24) * rho4 * tt ** 4
    eqn = sp.expand(D - s ** 2 / 2)
    a2v = sp.solve(sp.Eq(eqn.coeff(s, 3), 0), a2)[0]
    a3v = sp.solve(sp.Eq(eqn.coeff(s, 4).subs(a2, a2v), 0), a3)[0]
    return sp.simplify(a3v)

def bernoulli():

    def kl(a, b):
        e = 1e-300
        a = min(max(a, e), 1 - e)
        b = min(max(b, e), 1 - e)
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    IF = lambda m: 1 / (m * (1 - m))
    rho = lambda m: (2 * m - 1) / np.sqrt(m * (1 - m))
    rho4 = lambda m: 2 * (-3 * m ** 2 + 3 * m - 1) / (m * (m - 1))
    return ('Bernoulli', kl, IF, rho, rho4, lambda mu: 1 - mu)

def poisson():

    def kl(a, b):
        return a * np.log(a / b) - a + b
    IF = lambda m: 1 / m
    rho = lambda m: -1 / np.sqrt(m)
    rho4 = lambda m: 2 / m
    return ('Poisson', kl, IF, rho, rho4, lambda mu: 50.0)

def binomial(n=20):

    def kl(a, b):
        pa, pb = (a / n, b / n)
        e = 1e-12
        pa = min(max(pa, e), 1 - e)
        pb = min(max(pb, e), 1 - e)
        return n * (pa * np.log(pa / pb) + (1 - pa) * np.log((1 - pa) / (1 - pb)))
    IF = lambda m: n / (m * (n - m))
    rho = lambda m: (2 * m - n) / np.sqrt(m * n * (n - m))
    rho4 = lambda m: 2 * (m * (m - n) - (2 * m - n) ** 2) / (m * n * (m - n))
    return (f'Binomial(n={n})', kl, IF, rho, rho4, lambda mu: n - mu)

def check_family(name, kl, IF, rho, rho4, sup, mus):
    ok = True
    for mu_a in mus:
        last_ratio = c2 = None
        for N in [10000.0, 100000.0, 1000000.0]:
            D = 1.0 / N
            hi = sup(mu_a) - 1e-09
            eps = brentq(lambda e: kl(mu_a, mu_a + e) - D, 1e-13, hi)
            q = mu_a + eps
            s = np.sqrt(2 * D)
            t = eps * np.sqrt(IF(q))
            r_n = t / s
            last_ratio = (r_n - (1 + rho(q) * s / 6)) / s ** 2
            c2 = (5 * rho(q) ** 2 - 3 * rho4(q)) / 72
        match = abs(last_ratio - c2) < 0.005
        ok &= match
    return ok

def check_operational(name, kl, IF, rho, rho4, sup, mus):
    ok = True
    for q in mus:
        S, Y = ([], [])
        for s_alvo in np.geomspace(0.002, 0.04, 16):
            try:
                eps = brentq(lambda e: np.sqrt(2 * max(kl(q - e, q), 0.0)) - s_alvo, 1e-09, q * 0.999)
            except ValueError:
                continue
            D = kl(q - eps, q)
            s = np.sqrt(2 * D)
            r_til = eps * np.sqrt(IF(q - eps) / (2 * D))
            S.append(s)
            Y.append(r_til - 1)
        S = np.array(S)
        Y = np.array(Y)
        A = np.vstack([S, S ** 2, S ** 3]).T
        (a1, a2, a3), *_ = np.linalg.lstsq(A, Y, rcond=None)
        a1_th = -rho(q) / 3
        a2_th = -2 * rho(q) ** 2 / 9 + 5 * rho4(q) / 24
        a2_t = -rho(q) ** 2 / 6 + 5 * rho4(q) / 24
        match = abs(a1 - a1_th) < 0.0002 and abs(a2 - a2_th) < 0.005
        ok &= match
    return ok
