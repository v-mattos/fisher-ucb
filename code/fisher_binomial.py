# -*- coding: utf-8 -*-
"""
fisher_binomial.py
==================
Binomial(n, p) extension with a known FIXED n, parameterized by the mean
mu = n*p, mu in (0, n). Reward X in {0, 1, ..., n}.

Motivation (the missing test): in the families already tested (Bernoulli<1/2,
Exponential, Poisson), rho <= 0 over the whole relevant domain -- exactly the
regime where Fisher-UCB under-explores and wins. Binomial is the natural
family to probe the ADVERSE regime: the asymmetry coefficient

    rho(mu) = (2 mu - n) / sqrt(mu * n * (n - mu)) = -skew(P_mu)

FLIPS SIGN at mu = n/2 (p = 1/2):
    - p < 1/2  =>  rho < 0  (Fisher radius SMALLER than KL-UCB; favourable regime)
    - p > 1/2  =>  rho > 0  (Fisher radius LARGER than KL-UCB; ADVERSE regime)

So an instance with arms at p>1/2 tests the side where theory predicts
r_n > 1 (Theorem 5.5) and Fisher-UCB should NOT dominate KL-UCB.

Theoretical objects (n fixed; verified in verification/binom_verify.py):
    V(mu)    = mu (n - mu) / n              (= Var; note V = mu - mu^2/n)
    I_F(mu)  = 1 / V(mu) = n / (mu (n-mu))  (Fisher information = 1/Var)
    rho(mu)  = (2 mu - n) / sqrt(mu n (n-mu))
    KL in the mean parameterization: KL(Bin(n, mu_a/n) || Bin(n, mu_b/n)) = n * KL_Bernoulli(p_a, p_b)

Usage:
    python fisher_binomial.py     # runs the favourable (p<1/2) and adverse (p>1/2) instances
"""
import time
from math import log, sqrt

import numpy as np

from fisher_ucb import run_ucb1, run_fisher_ucb, run_kl_ucb


# ============================================================
#  Binomial(n, p) arm with n fixed
# ============================================================

class BinomialArm:
    """Binomial(n, p) arm. The mean is mu = n*p; the reward is X in {0,...,n}."""

    def __init__(self, n, p):
        assert n >= 1 and 0.0 < p < 1.0, "require n>=1 and 0<p<1"
        self.n = int(n)
        self.p = float(p)
        self._mean = self.n * self.p

    def sample(self, rng):
        return float(rng.binomial(self.n, self.p))

    def mean(self):
        return self._mean


# ============================================================
#  Fisher information I_F(mu) of the Binomial (n fixed, closed form)
#  Receives mu AFTER projection. n is fixed by closure in make_fisher_binomial.
# ============================================================

def make_fisher_binomial(n):
    """Returns I_F(mu) = n / (mu (n - mu)) for the given n."""
    def fisher_binomial(mu):
        # mu must lie in (0, n); the projection Pi_K guarantees this.
        return n / (mu * (n - mu))
    return fisher_binomial


# ============================================================
#  Binomial KL divergence (mean parameterization)
# ============================================================

def make_kl_binomial(n):
    """
    Returns KL(Bin(n, mu_a/n) || Bin(n, mu_b/n)) = n * KL_Bernoulli(p_a, p_b),
    as a function of the MEANS mu_a, mu_b in (0, n).
    """
    def kl_binomial(mu_a, mu_b):
        eps = 1e-12
        pa = min(max(mu_a / n, eps), 1.0 - eps)
        pb = min(max(mu_b / n, eps), 1.0 - eps)
        return n * (pa * log(pa / pb) + (1 - pa) * log((1 - pa) / (1 - pb)))
    return kl_binomial


# ============================================================
#  Demo experiment (sanity check, not part of the paper pipeline)
# ============================================================

def _final_stats(finals, z=1.96):
    finals = np.asarray(finals, dtype=float)
    nrep = len(finals)
    half = z * finals.std(ddof=1) / sqrt(nrep)
    return {"mean": float(finals.mean()), "half": float(half),
            "median": float(np.median(finals)), "max": float(finals.max())}


def binomial_experiment(arms, n, T=50000, reps=40, beta=1.25,
                        warmup=20, label="", verbose=True):
    """
    Runs UCB1, KL-UCB and Fisher-UCB on a Binomial(n, .) instance.

    The projection Pi_K is [delta, n - delta] (margin on both ends, since the
    mean lives in (0, n)), with delta small relative to n.
    """
    delta = 0.05 * 1.0          # absolute margin at the ends of (0,n)
    proj_interval = (delta, n - delta)
    fisher_fn = make_fisher_binomial(n)
    kl_fn = make_kl_binomial(n)
    # KL-UCB with the index bounded in [mu_hat, n] (upper support = n).
    specs = [
        ("UCB1",       run_ucb1,       {}),
        ("KL-UCB",     run_kl_ucb,     dict(kl_fun=kl_fn, bounded=True,
                                            upper_bound=float(n))),
        ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=fisher_fn, proj_interval=proj_interval,
                                            beta=beta, warmup=warmup)),
    ]
    if verbose:
        best = max(a.mean() for a in arms)
        ps = [round(a.p, 2) for a in arms]
        print(f"=== Binomial n={n} {label} (K={len(arms)}, T={T}, {reps} seeds, "
              f"beta={beta}, p={ps}, best_mu={best}) ===")
    results = {}
    for name, fn, kw in specs:
        t0 = time.time()
        finals = [fn(arms, T, seed=s, **kw)[-1] for s in range(reps)]
        runtime = (time.time() - t0) / reps
        st = _final_stats(finals)
        st["runtime"] = runtime
        results[name] = st
        if verbose:
            print(f"  {name:11s}: final regret = {st['mean']:8.2f} "
                  f"(+/- {st['half']:6.2f})  median={st['median']:7.2f}  "
                  f"max={st['max']:7.1f}  runtime/seed={runtime:.3f}s")
    return results


if __name__ == "__main__":
    # MIRRORED pair (n=10): same |rho| on both, only the sign changes.
    # Favourable: p<1/2 (rho<0), best p=0.22. Adverse: mirrored p<->1-p (rho>0), best p=0.92.
    n = 10
    fav = [BinomialArm(n, 0.08), BinomialArm(n, 0.15),
           BinomialArm(n, 0.22), BinomialArm(n, 0.12)]   # rho<0
    binomial_experiment(fav, n, reps=20, label="[favourable rho<0]")
    print()
    adv = [BinomialArm(n, 0.92), BinomialArm(n, 0.85),
           BinomialArm(n, 0.78), BinomialArm(n, 0.88)]   # rho>0 (mirror)
    binomial_experiment(adv, n, reps=20, label="[adverse rho>0]")
