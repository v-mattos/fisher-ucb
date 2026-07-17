# -*- coding: utf-8 -*-
"""
Kverify2.py
===========
Verifies the CONE (framing) of Theorem 5.5 (KL-vs-Fisher gap): the empirical
radius ratio r_emp = eps_KL/eps_F falls inside the first-order cone

    1 - (C_loc/6) s  <=  r_emp  <=  1 + (C_loc/6) s   (+ uncontrolled O(s^2))

where C_loc is the LOCAL self-concordance constant on the interval [mu, q],
q = mu + eps is the base point of the Bregman divergence, and s = eps sqrt(I_F(q)).

NOTE (fix vs. an earlier version): there is NO "+ C^2 s^2/144" term. That
remainder bound was false under the paper's own hypotheses (see the appendix
and Kgap_c2.py). This script verifies only the linear cone; the second-order
term is O(s^2) with a constant depending on phi'''' (not on C), and is not
certifiable by self-concordance alone. We therefore use an empirical O(s^2)
slack here, not the C^2/144 formula.
"""
import numpy as np
from scipy.optimize import brentq

def phi(m): return m*np.log(m) + (1-m)*np.log(1-m)
def Dphi(x, y):
    # D_phi(x || y), base at the 2nd argument y; phi'(y)=log(y/(1-y))
    return phi(x) - phi(y) - (np.log(y) - np.log(1-y)) * (x - y)
def IF(m): return 1.0/(m*(1-m))
def rho(m): return (2*m-1)/np.sqrt(m*(1-m))   # = phi'''/IF^{3/2}

print("Gap cone with the correct base (2nd arg q = mu+eps):")
print(f"{'mu':>5} {'eps':>6} {'s_q':>7} {'C_loc':>6} {'r_lo':>8} {'r_emp':>8} {'r_hi':>8} {'ok':>4}")
allok = True
for mu in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for Ln in [0.005, 0.02, 0.05]:
        eps = brentq(lambda e: Dphi(mu, mu+e) - Ln, 1e-12, 1-mu-1e-9)
        q = mu + eps
        s_q = eps*np.sqrt(IF(q))                  # local norm at q
        grid = np.linspace(mu, q, 40)
        Cloc = max(abs(rho(m)) for m in grid)     # C_loc on the interval [mu,q]
        r_emp = eps/np.sqrt(2*Ln/IF(q))           # eps_KL / eps_F (geometric ratio)
        tau = Cloc/2*s_q
        if tau >= 1:
            print(f"{mu:5.2f} tau>=1 skip"); continue
        # LINEAR cone + empirical O(s^2) slack (NOT C^2/144)
        slack = s_q**2                            # generous 2nd-order slack
        r_lo = 1 - Cloc*s_q/6 - slack
        r_hi = 1 + Cloc*s_q/6 + slack
        ok = (r_lo - 1e-9 <= r_emp <= r_hi + 1e-9)
        allok &= ok
        print(f"{mu:5.2f} {eps:6.3f} {s_q:7.4f} {Cloc:6.2f} {r_lo:8.4f} {r_emp:8.4f} {r_hi:8.4f} {'OK' if ok else 'X':>4}")
print(f"\nAll inside the linear cone (correct base): {allok}")
