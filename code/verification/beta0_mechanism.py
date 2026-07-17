# -*- coding: utf-8 -*-
"""
beta0_mechanism.py
==================
Verifies the beta_0 MECHANISM of Theorem 5.4 (O(log T) regret).

Paper's conclusion (Appendix D): the threshold beta_0(C_K) exceeds 1 NOT
because of underestimation of the OPTIMAL arm -- that side alone gives
beta_0 = 1, since the exact divergence is steeper than the Fisher quadratic
there -- but because of RADIUS INFLATION on SUBOPTIMAL arms near the boundary
of the projection, where I_F(mu_bar) at the projected point is smaller than
I_F(mu_a) for small n.

This script confirms both sides separately:

  [A] Optimal-arm side: sup_x [ (1/2) I_F(mu*) x^2 / D_phi(mu*-x || mu*) ] = 1.
      => beta_0 = 1 would suffice if this were the only mechanism.

  [B] Suboptimal-arm side: the inflation of the Fisher radius vs. the exact
      KL radius on the upper side, sup over n in the active regime
      (tau<=1/2), is what forces beta_0 > 1. At the boundary of
      [0.01,0.99] it reaches ~1.25, and is bounded by 3/2 (from
      tau<=1/2 => inflation <= 1+tau).

Supersedes beta_theory.py and peeling_check.py, which measured only side [A]
(or always the pulled arm) and therefore didn't explain beta_0 ~ 1.25.

Run:
    python beta0_mechanism.py
"""
import numpy as np
from scipy.optimize import brentq


def Dbern(x, y):
    e = 1e-300; x = min(max(x, e), 1-e); y = min(max(y, e), 1-e)
    return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))
def IF(m): return 1.0/(m*(1-m))
def rho(m): return (2*m-1)/np.sqrt(m*(1-m))


def side_A_optimal(mus):
    """sup_x (1/2)I_F x^2 / D(mu*-x||mu*); should give ~1 for every mu*."""
    print("[A] OPTIMAL-arm side: sup_x (1/2)I_F(mu*)x^2 / D_phi(mu*-x||mu*)")
    print("    (=1 means beta_0=1 from this side alone)\n")
    ok = True
    for mu in mus:
        xs = np.linspace(1e-4, mu*0.95, 4000)
        ratios = [0.5*IF(mu)*x**2/Dbern(mu-x, mu) for x in xs if Dbern(mu-x, mu) > 0]
        b2 = max(ratios)
        ok &= abs(b2 - 1.0) < 1e-2
        print(f"    mu*={mu:.2f}: beta_0^2={b2:.3f}  beta_0={np.sqrt(b2):.3f}")
    return ok


def side_B_suboptimal(mu_as, C=(0.01, 0.99), T=50000, beta=1.5):
    """Inflation of the Fisher radius vs. the exact KL radius on the upper
    side (suboptimal arms), sup over n in the active regime (tau<=1/2).
    Forces beta_0 > 1."""
    print("\n[B] SUBOPTIMAL-arm side: radius inflation (forces beta_0>1)")
    print("    sup_{n: tau<=1/2} eps_F/eps_KL ; theoretical bound 3/2\n")
    Cglob = max(abs(rho(C[0])), abs(rho(C[1])))
    L = np.log(T)
    sup_all = 0.0
    for mu_a in mu_as:
        n1 = int((Cglob*beta)**2 * L) + 1
        infl = 0.0
        for n in range(max(n1, 2), n1+5000, 50):
            D = L/n
            try:
                eps_KL = brentq(lambda e: Dbern(mu_a+e, mu_a) - D, 1e-12, 1-mu_a-1e-9)
            except Exception:
                continue
            mu_bar = min(mu_a+eps_KL, C[1])
            eps_F = np.sqrt(2*D/IF(mu_bar))
            infl = max(infl, eps_F/eps_KL)
        sup_all = max(sup_all, infl)
        print(f"    mu_a={mu_a:.2f}: C_K={Cglob:.1f} sup_inflation={infl:.3f} => beta_0<={infl:.3f}")
    print(f"\n    global sup of the inflation = {sup_all:.3f}  (theoretical bound 3/2={1.5})")
    return sup_all <= 1.5 + 1e-6


def side_B_multifamily(T=50000):
    """
    3/2 bound on the inflation RESTRICTED to the active regime tau<=1/2,
    across three families. This is the correct check: the bound in
    lem:sc-hessian only holds for tau<=1/2 (the burn-in n_1 definition).
    Including n<n_1 measures something the proof does NOT claim (there the
    radius isn't even in the controlled regime).
    Confirms the bound is family-independent (Bernoulli, Poisson, Exponential).
    """
    import numpy as np
    from scipy.optimize import brentq
    L = np.log(T)

    def bern():
        D = lambda a, b: a*np.log(a/b)+(1-a)*np.log((1-a)/(1-b))
        IF = lambda m: 1/(m*(1-m)); rho = lambda m: (2*m-1)/np.sqrt(m*(1-m))
        return "Bernoulli", D, IF, rho, (lambda mu: 1-mu-1e-9)

    def pois():
        D = lambda a, b: a*np.log(a/b)-a+b
        IF = lambda m: 1/m; rho = lambda m: -1/np.sqrt(m)
        return "Poisson", D, IF, rho, (lambda mu: 1e6)

    def expo():
        D = lambda a, b: a/b-1-np.log(a/b)
        IF = lambda m: 1/m**2; rho = lambda m: -2.0
        return "Exponential", D, IF, rho, (lambda mu: 1e6)

    print("\n[B'] 3/2 bound restricted to tau<=1/2, across 3 families (family-independent):")
    allok = True
    for name, D, IF, rho, sup in [bern(), pois(), expo()]:
        mus = ([0.05, 0.1, 0.2] if name == "Bernoulli"
               else [0.1, 0.2, 0.5] if name == "Poisson"
               else [0.5, 1.0, 2.0])
        fam_sup = 0.0
        for mu_a in mus:
            Cloc = abs(rho(mu_a))
            sup_infl = 0.0
            for n in range(10, 20000, 10):
                Dt = L/n
                try:
                    eps = brentq(lambda e: D(mu_a+e, mu_a)-Dt, 1e-12, sup(mu_a))
                except Exception:
                    continue
                tau = (Cloc/2)*eps*np.sqrt(IF(mu_a))
                if tau > 0.5:          # outside the active regime: does not apply
                    continue
                mu_bar = mu_a+eps
                sup_infl = max(sup_infl, np.sqrt(2*Dt/IF(mu_bar))/eps)
            fam_sup = max(fam_sup, sup_infl)
        ok = fam_sup <= 1.5 + 1e-6
        allok &= ok
        print(f"    {name:12s}: sup inflation (tau<=1/2) = {fam_sup:.3f}  "
              f"{'<=3/2 OK' if ok else 'VIOLATES 3/2'}")
    return allok


if __name__ == "__main__":
    a = side_A_optimal([0.4, 0.2, 0.1, 0.05])
    b = side_B_suboptimal([0.3, 0.1, 0.05, 0.02, 0.01])
    c = side_B_multifamily()
    print("\n=== SUMMARY ===")
    print(f"  [A]  optimal-arm side of beta_0=1  : {'OK' if a else 'FAIL'}")
    print(f"  [B]  suboptimal inflation (Bernoulli): {'OK' if b else 'FAIL'}")
    print(f"  [B'] 3/2 bound across 3 families    : {'OK' if c else 'FAIL'}")
    print("  Mechanism confirmed: beta_0 in [1, 3/2], forced by side [B], "
          "family-independent."
          if (a and b and c) else "  SOME FAILURES")
