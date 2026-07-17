# -*- coding: utf-8 -*-
"""
poisson_verify.py
=================
INDEPENDENT numerical verification of the paper's theoretical claims for the
Poisson family. Does not depend on the bandit code; only checks the
information-geometry identities and the gap expansion (Theorem 5.5).

Checks:
  1. D_phi(a||b) == KL(Poisson(a)||Poisson(b)) = a log(a/b) - a + b.
  2. rho(mu)  = -1/sqrt(mu) = -skewness(P_mu); self-concordance |rho|<=2
     holds exactly for mu >= 1/4.
  3. rho4(mu) = 2/mu (related to the excess kurtosis 1/mu).
  4. Gap expansion r_n = 1 + (1/6) rho(q) s + c2(q) s^2 + O(s^3),
     with c2(q) = (5 rho(q)^2 - 3 rho4(q)) / 72 (appendix formula).
     Verifies that R/s^2 -> c2(q) in the limit s -> 0, evaluating rho, rho4 at q.

Run:
    python poisson_verify.py
"""
import numpy as np
from scipy.optimize import brentq


# ---- Poisson family objects (mean parameterization) ----
def phi(mu):
    return mu * np.log(mu) - mu          # negentropy (dual of psi)

def Dphi(x, y):
    # D_phi(x||y) = phi(x) - phi(y) - phi'(y)(x-y), with phi'(y) = log y
    return phi(x) - phi(y) - np.log(y) * (x - y)

def kl_poisson(a, b):
    return a * np.log(a / b) - a + b

def IF(mu):
    return 1.0 / mu                      # phi''(mu)

def rho(mu):
    return -1.0 / np.sqrt(mu)            # phi'''/IF^{3/2} = -skew

def rho4(mu):
    return 2.0 / mu                      # phi''''/IF^2


def check_divergence(tol=1e-12):
    print("[1] D_phi == Poisson KL")
    ok = True
    for a in [0.3, 1.0, 2.0, 5.0]:
        for e in [0.05, 0.2, 0.5]:
            b = a + e
            d1, d2 = Dphi(a, b), kl_poisson(a, b)
            match = abs(d1 - d2) < tol
            ok = ok and match
            print(f"    a={a:4}, b={b:4}: D_phi={d1:.8f}  KL={d2:.8f}  "
                  f"{'OK' if match else 'FAIL'}")
    return ok


def check_skewness_selfconcordance():
    print("\n[2] rho(mu) = -1/sqrt(mu) = -skew ; |rho|<=2  <=>  mu>=1/4")
    ok = True
    for mu in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        # Poisson skewness = 1/sqrt(mu); rho must be its negative
        skew = 1.0 / np.sqrt(mu)
        match_skew = abs(rho(mu) - (-skew)) < 1e-12
        sc = abs(rho(mu)) <= 2.0 + 1e-12
        expected_sc = mu >= 0.25 - 1e-12
        match_sc = (sc == expected_sc)
        ok = ok and match_skew and match_sc
        print(f"    mu={mu:5}: rho={rho(mu):+.4f}  |rho|<=2? {sc}  "
              f"(expected {expected_sc})  rho4={rho4(mu):.3f}  "
              f"{'OK' if (match_skew and match_sc) else 'FAIL'}")
    return ok


def check_gap_c2():
    print("\n[3] gap: R/s^2 -> c2(q) = (5 rho^2 - 3 rho4)/72  (rho,rho4 at q)")
    ok = True
    for mu_a in [1.0, 3.0]:
        print(f"    --- mu_a = {mu_a} ---")
        last_ratio, c2pred = None, None
        for n in [1e4, 1e5, 1e6, 1e7]:
            D = 1.0 / n
            # solve KL(mu_a || mu_a+eps) = D, eps>0
            eps = brentq(lambda e: kl_poisson(mu_a, mu_a + e) - D, 1e-14, 50.0)
            q = mu_a + eps
            s = np.sqrt(2 * D)            # Fisher radius in the local norm
            t = eps * np.sqrt(IF(q))      # KL radius in the local norm (at q)
            r_n = t / s
            R = r_n - (1.0 + rho(q) * s / 6.0)
            ratio = R / s**2
            c2pred = (5 * rho(q)**2 - 3 * rho4(q)) / 72.0
            last_ratio = ratio
            print(f"      n={n:7.0e}: s={s:.3e}  r_n={r_n:.6f}  "
                  f"R/s^2={ratio:+.4f}  c2(q)={c2pred:+.4f}")
        # at the smallest s, R/s^2 should be close to c2(q)
        match = abs(last_ratio - c2pred) < 5e-3
        ok = ok and match
        print(f"      => limit {'MATCHES' if match else 'DIVERGES'} "
              f"(|R/s^2 - c2| = {abs(last_ratio - c2pred):.2e})")
    return ok


if __name__ == "__main__":
    r1 = check_divergence()
    r2 = check_skewness_selfconcordance()
    r3 = check_gap_c2()
    print("\n=== SUMMARY ===")
    print(f"  [1] divergence       : {'OK' if r1 else 'FAIL'}")
    print(f"  [2] skew/self-conc.  : {'OK' if r2 else 'FAIL'}")
    print(f"  [3] gap c2           : {'OK' if r3 else 'FAIL'}")
    print("  ALL OK" if (r1 and r2 and r3) else "  SOME FAILURES")
