# -*- coding: utf-8 -*-
"""
Kgap_c2.py
==========
Verifies the SECOND-ORDER COEFFICIENT of the gap (Theorem 5.5 / appendix):

    r_n = 1 + (1/6) rho(q) s + c2(q) s^2 + O(s^3),
    c2(q) = (5 rho(q)^2 - 3 rho4(q)) / 72,   rho4 = phi''''/I_F^2.

Supersedes the older Kconst.py/Kconst2.py, which claimed a remainder bound
|R(s)| <= C^2 s^2/144 -- FALSE under the paper's own hypotheses (the
second-order term depends on phi'''', the fourth derivative, not on C). Here
we derive c2 symbolically and confirm numerically that R/s^2 -> c2(q) across
THREE independent families (Bernoulli, Poisson, Binomial), evaluating rho,
rho4 at the base point q.

Requires sympy (only for the symbolic derivation; the numeric checks in
binom_verify.py/poisson_verify.py corroborate the same c2 formula without it).

Run:
    python Kgap_c2.py
"""
import numpy as np
import sympy as sp
from scipy.optimize import brentq


def derive_c2_symbolic():
    """Derives c2 by series inversion. Confirms c2 = (5 rho^2 - 3 rho4)/72."""
    s, rho, rho4 = sp.symbols('s rho rho4', real=True)
    a2, a3 = sp.symbols('a2 a3', real=True)
    tt = s + a2*s**2 + a3*s**3
    # D normalized in the local norm: (1/2)t^2 - (1/6)rho t^3 + (1/24)rho4 t^4 = s^2/2
    D = sp.Rational(1, 2)*tt**2 - sp.Rational(1, 6)*rho*tt**3 + sp.Rational(1, 24)*rho4*tt**4
    eqn = sp.expand(D - s**2/2)
    a2v = sp.solve(sp.Eq(eqn.coeff(s, 3), 0), a2)[0]
    a3v = sp.solve(sp.Eq(eqn.coeff(s, 4).subs(a2, a2v), 0), a3)[0]
    print("Symbolic derivation (series inversion):")
    print(f"  linear term a2 = {sp.simplify(a2v)}   (expected rho/6)")
    print(f"  c2 = a3 = {sp.simplify(a3v)}   (expected (5 rho^2 - 3 rho4)/72)")
    return sp.simplify(a3v)


# --- families: (name, kl(a,b), IF(mu), rho(mu), rho4(mu), domain sup) ---
def bernoulli():
    def kl(a, b):
        e = 1e-300; a = min(max(a, e), 1-e); b = min(max(b, e), 1-e)
        return a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b))
    IF = lambda m: 1/(m*(1-m))
    rho = lambda m: (2*m-1)/np.sqrt(m*(1-m))
    rho4 = lambda m: 2*(-3*m**2 + 3*m - 1)/(m*(m-1))   # phi''''/IF^2 for Bernoulli
    return "Bernoulli", kl, IF, rho, rho4, lambda mu: 1-mu

def poisson():
    def kl(a, b): return a*np.log(a/b) - a + b
    IF = lambda m: 1/m
    rho = lambda m: -1/np.sqrt(m)
    rho4 = lambda m: 2/m
    return "Poisson", kl, IF, rho, rho4, lambda mu: 50.0

def binomial(n=20):
    def kl(a, b):
        pa, pb = a/n, b/n
        e = 1e-12; pa = min(max(pa, e), 1-e); pb = min(max(pb, e), 1-e)
        return n*(pa*np.log(pa/pb) + (1-pa)*np.log((1-pa)/(1-pb)))
    IF = lambda m: n/(m*(n-m))
    rho = lambda m: (2*m-n)/np.sqrt(m*n*(n-m))
    rho4 = lambda m: 2*(m*(m-n) - (2*m-n)**2)/(m*n*(m-n))
    return f"Binomial(n={n})", kl, IF, rho, rho4, lambda mu: n-mu


def check_family(name, kl, IF, rho, rho4, sup, mus):
    print(f"\n--- {name} ---")
    ok = True
    for mu_a in mus:
        last_ratio = c2 = None
        for N in [1e4, 1e5, 1e6]:
            D = 1.0/N
            hi = sup(mu_a) - 1e-9
            eps = brentq(lambda e: kl(mu_a, mu_a+e) - D, 1e-13, hi)
            q = mu_a + eps
            s = np.sqrt(2*D); t = eps*np.sqrt(IF(q)); r_n = t/s
            last_ratio = (r_n - (1 + rho(q)*s/6))/s**2
            c2 = (5*rho(q)**2 - 3*rho4(q))/72
        match = abs(last_ratio - c2) < 5e-3
        ok &= match
        print(f"  mu_a={mu_a:6.2f}: R/s^2={last_ratio:+.5f}  c2(q)={c2:+.5f}  "
              f"{'OK' if match else 'DIVERGES'}")
    return ok


def check_operational(name, kl, IF, rho, rho4, sup, mus):
    """
    Verifies the OPERATIONAL ratio r_til = eps_KL * sqrt(IF(mu_hat)/(2D)),
    r_til = 1 - (rho/3) s + c_til2 s^2 + O(s^3), c_til2 = -rho^2/6 + 5 rho4/24.
    Extracts c_til2 by REGRESSION over a moderate range of s (avoids the
    catastrophic cancellation that occurs when measuring by direct difference
    at small s). Here the base is q = mu_a and the deficit goes downward
    (mu_hat = q - eps).
    """
    print(f"\n--- {name} (operational) ---")
    ok = True
    for q in mus:
        eps_list = np.geomspace(2e-4, 4e-3, 12)
        S, Y = [], []
        for eps in eps_list:
            if q - eps <= 0:
                continue
            s = eps*np.sqrt(IF(q))
            D = kl(q - eps, q)                 # D_phi(mu_hat || q)
            r_til = eps*np.sqrt(IF(q - eps)/(2*D))
            S.append(s); Y.append(r_til - 1)
        S = np.array(S); Y = np.array(Y)
        A = np.vstack([S, S**2]).T             # Y = a1 s + a2 s^2
        (a1, a2), *_ = np.linalg.lstsq(A, Y, rcond=None)
        a1_th = -rho(q)/3
        a2_th = -rho(q)**2/6 + 5*rho4(q)/24
        match = abs(a1 - a1_th) < 5e-3 and abs(a2 - a2_th) < 3e-2
        ok &= match
        print(f"  q={q:6.2f}: a1={a1:+.4f}(theory {a1_th:+.4f})  "
              f"a2={a2:+.4f}(theory {a2_th:+.4f})  {'OK' if match else 'DIVERGES'}")
    return ok


if __name__ == "__main__":
    derive_c2_symbolic()
    results = {}
    # Geometric ratio (base q, I_F at q): c2 = (5 rho^2 - 3 rho4)/72
    results["Bernoulli (geom)"] = check_family(*bernoulli(), mus=[0.2, 0.5, 0.8])
    results["Poisson (geom)"] = check_family(*poisson(), mus=[1.0, 3.0])
    results["Binomial (geom)"] = check_family(*binomial(20), mus=[4.0, 12.0, 16.0])
    # Operational ratio (I_F at mu_hat): c_til2 = -rho^2/6 + 5 rho4/24
    results["Bernoulli (op)"] = check_operational(*bernoulli(), mus=[0.2, 0.7])
    results["Poisson (op)"] = check_operational(*poisson(), mus=[1.0, 3.0])
    results["Binomial (op)"] = check_operational(*binomial(20), mus=[4.0, 16.0])
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:18s}: {'OK' if v else 'FAIL'}")
    print("  ALL OK" if all(results.values()) else "  SOME FAILURES")
