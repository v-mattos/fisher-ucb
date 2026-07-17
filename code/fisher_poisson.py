# -*- coding: utf-8 -*-
"""
fisher_poisson.py
=================
Poisson-family extension for the Fisher-UCB experiments.

The Poisson family is NOT in fisher_ucb.py (which covers Bernoulli, Gaussian
and Exponential). This module adds:

  - PoissonArm        : bandit arm sampling from Poisson(mu).
  - fisher_poisson    : I_F(mu) = 1/mu (closed form; I_F = 1/Var = 1/mu).
  - kl_poisson        : KL(P_a || P_b) = a log(a/b) - a + b
                        (= the Bregman divergence of the Poisson negentropy).
  - poisson_experiment: quick sanity-check demo running UCB1, KL-UCB and
                        Fisher-UCB on a Poisson instance, with the SAME
                        methodology as fisher_ucb.py (same anytime log t
                        level, 95% CI). Not used by the paper's reproduction
                        pipeline -- see gen_poisson_data.py.

Theoretical objects (mean parameterization mu):
    phi(mu)   = mu log mu - mu          (negentropy / Legendre dual of psi)
    phi'(mu)  = log mu = eta(mu)        (natural parameter)
    phi''(mu) = 1/mu   = I_F(mu)        (Fisher information = 1/Var)
    phi'''(mu)= -1/mu^2
    phi''''(mu)= 2/mu^3
    rho(mu)   = phi'''/I_F^{3/2} = -1/sqrt(mu) = -skewness(P_mu)   (always < 0)
    rho4(mu)  = phi''''/I_F^2     = 2/mu        (related to kurtosis: excess kurtosis=1/mu)

Since rho(mu) < 0 over the whole domain, Fisher-UCB uses a SMALLER radius
than KL-UCB (under-exploration rewarded in the rho<0 regime), the same
pattern as the other families tested. Since rho(mu) -> -inf as mu -> 0,
standard self-concordance (|rho| <= 2) only holds for mu >= 1/4; near zero
the projection Pi_K is required for a well-defined radius and a finite
constant C_K.

Usage:
    python fisher_poisson.py            # runs the small-gap instance
"""
import time
from math import log, sqrt

import numpy as np

# Reuses the algorithms from the core module (no logic duplication).
from fisher_ucb import run_ucb1, run_fisher_ucb, run_kl_ucb


# ============================================================
#  Poisson arm
# ============================================================

class PoissonArm:
    """Poisson(mu) bandit arm. The reward is a count (>= 0)."""

    def __init__(self, mean):
        assert mean > 0.0, "Poisson mean must be positive"
        self._mean = mean

    def sample(self, rng):
        # numpy.random.Generator.poisson returns an int; cast to float to
        # match the continuous empirical mean used by the algorithms.
        return float(rng.poisson(self._mean))

    def mean(self):
        return self._mean


# ============================================================
#  Fisher information I_F(mu) of the Poisson family (closed form)
#  Receives mu AFTER projection (consistent with fisher_ucb.py).
# ============================================================

def fisher_poisson(mu):
    """I_F(mu) = 1/Var = 1/mu for Poisson in the mean parameterization."""
    return 1.0 / mu


# ============================================================
#  Poisson KL divergence (mean parameterization)
# ============================================================

def kl_poisson(a, b):
    """
    KL(Poisson(a) || Poisson(b)) = a log(a/b) - a + b.

    Coincides with the Bregman divergence D_phi(a||b) of the negentropy
    phi(mu) = mu log mu - mu, which is the theoretical basis of the KL-UCB
    index for this family. The eps clamp avoids log(0) when the empirical
    mean hits zero (which happens with positive probability for small mu
    and finite n).
    """
    eps = 1e-12
    a = max(a, eps)
    b = max(b, eps)
    return a * log(a / b) - a + b


# ============================================================
#  Demo experiment (sanity check, not part of the paper pipeline)
# ============================================================

def _final_stats(finals, z=1.96):
    """Final-regret statistics: mean, CI half-width, median, max."""
    finals = np.asarray(finals, dtype=float)
    n = len(finals)
    half = z * finals.std(ddof=1) / sqrt(n)
    return {
        "mean": float(finals.mean()),
        "half": float(half),
        "median": float(np.median(finals)),
        "max": float(finals.max()),
    }


def poisson_experiment(arms=None, T=50000, reps=40, beta=1.25,
                       proj_interval=(0.05, 1e6), warmup=20, verbose=True):
    """
    Runs UCB1, KL-UCB and Fisher-UCB on a Poisson instance.

    Default: the SMALL-GAP instance (means (2.0,2.1,2.2,1.9), best=2.2). This
    is the instance tabulated in the paper: the honest regime where the
    Fisher-UCB < KL-UCB ordering is clean and UCB1 (scale-mis-specified)
    shows a heavy tail rather than a spurious win.

    Parameters
    ----------
    arms          : list of PoissonArm; defaults to the small-gap instance.
    T             : horizon.
    reps          : number of seeds (40 in the paper for Poisson).
    beta          : Fisher-UCB radius scale (1.25 = the located beta_0 threshold).
    proj_interval : (mu_min, mu_max) of the Pi_K operator. mu_min=0.05 keeps
                    away from zero, where I_F=1/mu blows up.
    warmup        : pulls per arm before the Fisher radius is activated.

    Returns
    -------
    dict {algorithm_name: {mean, half, median, max, runtime}}.
    """
    if arms is None:
        # best = 2.2; gaps from 0.1 to 0.3 (the hard instance tabulated in the paper)
        arms = [PoissonArm(2.0), PoissonArm(2.1), PoissonArm(2.2), PoissonArm(1.9)]

    specs = [
        ("UCB1",       run_ucb1,       {}),
        ("KL-UCB",     run_kl_ucb,     dict(kl_fun=kl_poisson, bounded=False)),
        ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=fisher_poisson,
                                            proj_interval=proj_interval, beta=beta,
                                            warmup=warmup)),
    ]

    if verbose:
        best = max(a.mean() for a in arms)
        print(f"=== Poisson (K={len(arms)}, T={T}, {reps} seeds, "
              f"beta={beta}, best={best}) ===")

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


def poisson_projection_ablation(T=20000, reps=20, beta=1.25, verbose=True):
    """
    Projection ablation on a NEAR-ZERO instance (means (0.10,0.15,0.20,0.08)).

    Isolates the role of Pi_K: with projection onto [0.05, 1e6] the regret
    stabilizes; without projection (an (almost) unbounded interval) the
    MEDIAN is the same, but the mean and tail explode. Reproduces, on
    Poisson, the paper's Bernoulli finding (Remark "Roles of warm-up and
    projection").

    NOTE: the catastrophic effect of the missing projection is a TAIL event
    (only a fraction of seeds derail). With few seeds (< ~15) no derailment
    may appear and the two configurations may look identical; the paper uses
    reps=20, where the mean without projection rises to ~146 with max ~1e3
    while the median stays ~50. Use enough reps to sample the tail.
    """
    arms = [PoissonArm(0.10), PoissonArm(0.15), PoissonArm(0.20), PoissonArm(0.08)]
    if verbose:
        print(f"=== Poisson near-zero, projection ablation "
              f"(T={T}, {reps} seeds) ===")
    configs = [
        ("Fisher-UCB (proj)",    (0.05, 1e6)),
        ("Fisher-UCB (no proj)", (1e-9, 1e9)),  # effectively no projection
    ]
    results = {}
    for name, proj_interval in configs:
        finals = [run_fisher_ucb(arms, T, fisher_fn=fisher_poisson, proj_interval=proj_interval,
                                 beta=beta, seed=s)[-1] for s in range(reps)]
        st = _final_stats(finals)
        results[name] = st
        if verbose:
            print(f"  {name:22s}: mean={st['mean']:7.2f} "
                  f"(+/- {st['half']:6.2f})  median={st['median']:6.2f}  "
                  f"max={st['max']:7.1f}")
    return results


if __name__ == "__main__":
    # Tabulated instance (small gaps) + projection ablation.
    poisson_experiment()
    print()
    poisson_projection_ablation()
