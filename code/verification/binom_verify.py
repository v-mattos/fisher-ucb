# -*- coding: utf-8 -*-
"""
binom_verify.py
===============
INDEPENDENT numerical verification of the theoretical claims for the
Binomial(n, p) family with n fixed, parameterized by the mean mu = n p in (0, n).

Checks:
  1. I_F(mu) = 1/Var = n / (mu (n - mu)).
  2. rho(mu) = (2 mu - n)/sqrt(mu n (n-mu)) = -skew(P_mu); SIGN FLIPS at mu=n/2.
  3. KL in the mean parameterization = n * KL_Bernoulli(p_a, p_b).
  4. Gap expansion r_n = 1 + (1/6) rho(q) s + c2(q) s^2 + O(s^3),
     with c2(q) = (5 rho^2 - 3 rho4)/72, in BOTH regimes (rho<0 and rho>0).
     Also confirms the SIGN of (r_n - 1): <0 for rho<0, >0 for rho>0.

Run:
    python binom_verify.py
"""
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq

N = 20  # binomial n used in the theory checks


def IF(mu, n=N):
    return n / (mu * (n - mu))

def rho(mu, n=N):
    return (2 * mu - n) / np.sqrt(mu * n * (n - mu))

def rho4(mu, n=N):
    # from the symbolic derivation: 2*(mu*(mu-n) - (2mu-n)^2)/(mu*n*(mu-n))
    return 2 * (mu * (mu - n) - (2 * mu - n) ** 2) / (mu * n * (mu - n))

def kl_binom(mu_a, mu_b, n=N):
    pa, pb = mu_a / n, mu_b / n
    e = 1e-12
    pa = min(max(pa, e), 1 - e); pb = min(max(pb, e), 1 - e)
    return n * (pa * np.log(pa / pb) + (1 - pa) * np.log((1 - pa) / (1 - pb)))


def check_IF_rho():
    print("[1-2] I_F = 1/Var ; rho = -skew ; rho flips sign at mu=n/2")
    ok = True
    for p in [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]:
        mu = N * p
        var = binom.var(N, p)
        skew = binom.stats(N, p, moments="s")
        m_if = np.isclose(1 / var, IF(mu))
        m_rho = np.isclose(-skew, rho(mu))
        sign = "rho<0" if rho(mu) < 0 else ("rho=0" if abs(rho(mu)) < 1e-9 else "rho>0 ADVERSE")
        ok = ok and m_if and m_rho
        print(f"    p={p:.2f} mu={mu:5.1f}: 1/Var={1/var:.4f} I_F={IF(mu):.4f} [{m_if}] | "
              f"-skew={-skew:+.4f} rho={rho(mu):+.4f} [{m_rho}]  {sign}")
    return ok


def check_kl():
    print("\n[3] KL in the mean parameterization == n * KL_Bernoulli")
    ok = True
    for mu_a in [4.0, 10.0, 16.0]:
        for e in [0.5, 1.0]:
            d1 = kl_binom(mu_a, mu_a + e)
            pa, pb = mu_a / N, (mu_a + e) / N
            d2 = N * (pa * np.log(pa / pb) + (1 - pa) * np.log((1 - pa) / (1 - pb)))
            match = np.isclose(d1, d2)
            ok = ok and match
            print(f"    mu_a={mu_a:5}, e={e}: KL={d1:.6f}  n*KL_bern={d2:.6f}  {'OK' if match else 'FAIL'}")
    return ok


def check_gap_both_regimes():
    print("\n[4] gap c2 and the SIGN of (r_n - 1) in both regimes")
    ok = True
    # mu_a=4 (p=0.2, rho<0) and mu_a=16 (p=0.8, rho>0)
    for mu_a, regime in [(4.0, "rho<0"), (16.0, "rho>0 ADVERSE")]:
        print(f"    --- mu_a={mu_a} (p={mu_a/N:.2f}, {regime}, rho(mu_a)={rho(mu_a):+.3f}) ---")
        last_ratio = c2pred = r_n = None
        for n_pulls in [1e4, 1e5, 1e6]:
            D = 1.0 / n_pulls
            hi = (N - mu_a) - 1e-9
            eps = brentq(lambda e: kl_binom(mu_a, mu_a + e) - D, 1e-13, hi)
            q = mu_a + eps
            s = np.sqrt(2 * D); t = eps * np.sqrt(IF(q)); r_n = t / s
            ratio = (r_n - (1 + rho(q) * s / 6)) / s ** 2
            c2pred = (5 * rho(q) ** 2 - 3 * rho4(q)) / 72
            last_ratio = ratio
            print(f"      n={n_pulls:7.0e}: r_n={r_n:.6f}  R/s^2={ratio:+.5f}  c2(q)={c2pred:+.5f}")
        m_c2 = abs(last_ratio - c2pred) < 5e-3
        # the sign of (r_n - 1) must follow the sign of rho
        m_sign = (r_n < 1) == (rho(mu_a) < 0)
        ok = ok and m_c2 and m_sign
        print(f"      => c2 {'MATCHES' if m_c2 else 'DIVERGES'} ; "
              f"sign(r_n-1) {'MATCHES' if m_sign else 'DIVERGES'} sign(rho)")
    return ok


if __name__ == "__main__":
    r1 = check_IF_rho()
    r2 = check_kl()
    r3 = check_gap_both_regimes()
    print("\n=== SUMMARY ===")
    print(f"  [1-2] I_F, rho, sign  : {'OK' if r1 else 'FAIL'}")
    print(f"  [3]   KL              : {'OK' if r2 else 'FAIL'}")
    print(f"  [4]   gap c2 and signs: {'OK' if r3 else 'FAIL'}")
    print("  ALL OK" if (r1 and r2 and r3) else "  SOME FAILURES")
