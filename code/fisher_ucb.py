"""
fisher_ucb.py
================================================================
Core module: bandit arms, closed-form Fisher information, KL-UCB numerical
inversion, and the three algorithms compared in the paper (UCB1, KL-UCB,
Fisher-UCB). Used directly by gen_regret_data.py (Bernoulli/Exponential) and
imported by fisher_poisson.py/fisher_binomial.py for their family-specific
extensions.

Conventions:
  - All rewards are such that a LARGER mean is BETTER (best = argmax). To
    model "smaller is better" (e.g. latency), negate the reward instead.
  - KL-UCB and Fisher-UCB use the SAME anytime confidence level log(t) (no
    delta, no horizon T in the index) so the comparison is on equal footing.
  - Fisher-UCB projects the empirical mean onto a compact interval
    `proj_interval = (mu_min, mu_max)` (written Pi_K in the paper) before
    evaluating the closed-form Fisher information I_F; the index centre still
    uses the raw (unprojected) empirical mean.
"""

import numpy as np
from math import log, sqrt
import time


# ============================================================
#  Arms (bandit environments)
# ============================================================

class BernoulliArm:
    def __init__(self, p):
        assert 0.0 <= p <= 1.0
        self.p = p

    def sample(self, rng):
        return rng.binomial(1, self.p)

    def mean(self):
        return self.p


class GaussianArm:
    def __init__(self, mu, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def sample(self, rng):
        return rng.normal(self.mu, self.sigma)

    def mean(self):
        return self.mu


class ExponentialArm:
    def __init__(self, mean):
        assert mean > 0.0
        self._mean = mean

    def sample(self, rng):
        # numpy uses `scale = mean` for the exponential distribution
        return rng.exponential(self._mean)

    def mean(self):
        return self._mean



# ============================================================
#  Fisher information I_F(mu) per family (closed form)
#  Kept separate from the arms: I_F is a property of the family, not of the
#  environment. These receive mu AFTER projection (mu_bar = Pi_K(mu_hat)).
# ============================================================

def fisher_bernoulli(mu):
    return 1.0 / (mu * (1.0 - mu))

def fisher_gaussian(mu, sigma=1.0):
    return 1.0 / (sigma ** 2)

def fisher_exponential(mu):
    return 1.0 / (mu ** 2)


# ============================================================
#  KL divergences (mean parameterization)
# ============================================================

def kl_bernoulli(p, q):
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    q = min(max(q, eps), 1.0 - eps)
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))

def kl_exponential_mean(mu_p, mu_q):
    """KL(Exp(mean mu_p) || Exp(mean mu_q)) = log(mu_q/mu_p) + mu_p/mu_q - 1."""
    eps = 1e-12
    mu_p = max(mu_p, eps)
    mu_q = max(mu_q, eps)
    return log(mu_q / mu_p) + (mu_p / mu_q) - 1.0


# ============================================================
#  Numerical inversion of the KL-UCB index (bisection)
# ============================================================

def kl_ucb_index_bounded(p_hat, n, L, kl_fun, upper_bound=1.0, tol=1e-6, max_iter=50):
    """
    Solves  sup { q in [p_hat, upper_bound] : n * KL(p_hat || q) <= L }.
    L is the confidence level (e.g. log t). Bisection.
    """
    if n == 0:
        return upper_bound
    c = L / n
    low, high = p_hat, upper_bound
    if kl_fun(p_hat, high) <= c:
        return high
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if kl_fun(p_hat, mid) > c:
            high = mid
        else:
            low = mid
        if high - low < tol:
            break
    return low


def kl_ucb_index_unbounded(p_hat, n, L, kl_fun, tol=1e-6, max_iter=50):
    """
    KL-UCB for a mean unbounded above (e.g. Exponential). Safe bracketing:
    expand `high` until KL>c; if the cap is reached without bracketing,
    return `high` (an optimistic index) rather than an unbracketed `low`.
    """
    if n == 0:
        return p_hat + 1.0
    c = L / n
    low = max(p_hat, 1e-6)
    high = max(low * 2.0, 1e-3)
    expanded = False
    while high < 1e9:
        if kl_fun(p_hat, high) > c:
            expanded = True
            break
        high *= 2.0
    if not expanded:
        # bracketing failed: return the cap (optimistic) instead of an invalid low.
        return high
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if kl_fun(p_hat, mid) > c:
            high = mid
        else:
            low = mid
        if high - low < tol:
            break
    return low


# ============================================================
#  Algorithms
# ============================================================

def run_ucb1(arms, T, c=2.0, seed=0):
    """Hoeffding UCB1: index = mu_hat + sqrt(c log t / (2 n))."""
    rng = np.random.default_rng(seed)
    K = len(arms)
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K, dtype=float)
    regrets = np.zeros(T, dtype=float)
    true_means = np.array([arm.mean() for arm in arms])
    best_mean = np.max(true_means)
    for t in range(T):
        if t < K:
            a = t
        else:
            ucb = means + np.sqrt(c * log(t + 1) / (2.0 * counts))
            a = int(np.argmax(ucb))
        x = arms[a].sample(rng)
        counts[a] += 1
        means[a] += (x - means[a]) / counts[a]
        regrets[t] = best_mean - true_means[a]
    return np.cumsum(regrets)


def run_fisher_ucb(arms, T, fisher_fn, proj_interval, beta=1.0,
                   warmup=20, seed=0):
    """
    Fisher-UCB with PROJECTION and warm-up (the paper's algorithm).
        UCB_t(a) = mu_hat_a + beta sqrt( log(t) / (n_a I_F(Pi_K mu_hat_a)) )
    Anytime level log(t), same as basic KL-UCB (Garivier-Cappe). No delta,
    no horizon: the index does not depend on T.
    fisher_fn: closed-form I_F(mu) for the family. MUST accept a numpy array
    (every fisher_fn in this project is an elementary formula and already
    works this way with no change needed).
    proj_interval = (mu_min, mu_max): the projection interval K = Pi_K's domain.
    warmup: minimum pulls per arm before the Fisher radius is activated.

    VECTORIZED (2026-07-11): the index for all K arms is computed in one
    numpy call (clip + fisher_fn + sqrt), like run_ucb1, instead of a
    per-arm Python loop. Elementwise arithmetic is identical -- results are
    bit-for-bit equal to the earlier loop-based version (verified by hash);
    only the Python overhead changes. This matters for the runtime/scaling
    comparisons at large K, where the old Python loop artificially inflated
    Fisher-UCB's per-round cost relative to UCB1 (which was already vectorized).
    """
    rng = np.random.default_rng(seed)
    K = len(arms)
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K, dtype=float)
    regrets = np.zeros(T, dtype=float)
    true_means = np.array([arm.mean() for arm in arms])
    best_mean = np.max(true_means)
    lo, hi = proj_interval

    for t in range(T):
        if t < K:
            a = t
        else:
            under = np.where(counts < warmup)[0]
            if under.size > 0:
                a = int(under[0])               # warm-up phase
            else:
                Lt = log(max(t + 1, 2))          # anytime level log(t)
                mu_bar = np.clip(means, lo, hi)  # projection Pi_K (vectorized)
                I = fisher_fn(mu_bar)
                rad = beta * np.sqrt(Lt / (counts * I))
                ucb = means + rad
                a = int(np.argmax(ucb))
        x = arms[a].sample(rng)
        counts[a] += 1
        means[a] += (x - means[a]) / counts[a]
        regrets[t] = best_mean - true_means[a]
    return np.cumsum(regrets)


def run_kl_ucb(arms, T, kl_fun, bounded=True, upper_bound=1.0, seed=0):
    """
    Generic KL-UCB. Confidence level L = log(t) (basic Garivier-Cappe),
    the SAME level used by Fisher-UCB, for a fair comparison.
    """
    rng = np.random.default_rng(seed)
    K = len(arms)
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K, dtype=float)
    regrets = np.zeros(T, dtype=float)
    true_means = np.array([arm.mean() for arm in arms])
    best_mean = np.max(true_means)
    cap = float(true_means.max() * 2.0)

    for t in range(T):
        if t < K:
            a = t
        else:
            L = log(max(t + 1, 2))
            ucb = np.zeros(K)
            for i in range(K):
                if counts[i] == 0:
                    ucb[i] = upper_bound if bounded else cap
                elif bounded:
                    ucb[i] = kl_ucb_index_bounded(means[i], counts[i], L,
                                                  kl_fun, upper_bound)
                else:
                    ucb[i] = kl_ucb_index_unbounded(max(means[i], 1e-6),
                                                    counts[i], L, kl_fun)
            a = int(np.argmax(ucb))
        x = arms[a].sample(rng)
        counts[a] += 1
        means[a] += (x - means[a]) / counts[a]
        regrets[t] = best_mean - true_means[a]
    return np.cumsum(regrets)


def _multi(fn, arms, T, reps, **kw):
    """Runs fn over seeds 0..reps-1; returns (final_regrets, times)."""
    regs, times = [], []
    for s in range(reps):
        t0 = time.perf_counter()
        r = fn(arms, T, seed=s, **kw)
        times.append(time.perf_counter() - t0)
        regs.append(r[-1])
    return np.array(regs), np.array(times)


def bernoulli_experiment(T=50000, reps=5):
    """Quick sanity-check demo: Bernoulli, K=4, best = 0.40. Not used by the
    paper's reproduction pipeline (see gen_regret_data.py)."""
    arms = [BernoulliArm(0.30), BernoulliArm(0.35),
            BernoulliArm(0.40), BernoulliArm(0.32)]   # best = 0.40
    proj_interval = (0.01, 0.99)
    print(f"\n=== Bernoulli (K={len(arms)}, T={T}, {reps} seeds, best=0.40) ===")
    for name, fn, kw in [
        ("UCB1",       run_ucb1, {}),
        ("KL-UCB",     run_kl_ucb, dict(kl_fun=kl_bernoulli, bounded=True)),
        ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=fisher_bernoulli, proj_interval=proj_interval)),
    ]:
        reg, tm = _multi(fn, arms, T, reps, **kw)
        print(f"  {name:12s} regret {reg.mean():7.1f} +- {reg.std():5.1f}   "
              f"time {tm.mean():6.3f}s")


def exponential_experiment(T=50000, reps=5):
    """Quick sanity-check demo: Exponential, K=4, best = 2.0 (larger mean is
    better; reward = X). Not used by the paper's reproduction pipeline."""
    arms = [ExponentialArm(1.0), ExponentialArm(1.5),
            ExponentialArm(2.0), ExponentialArm(0.7)]  # best = 2.0
    proj_interval = (0.05, 1e6)
    print(f"\n=== Exponential (K={len(arms)}, T={T}, {reps} seeds, best=2.0) ===")
    for name, fn, kw in [
        ("UCB1",       run_ucb1, {}),
        ("KL-UCB",     run_kl_ucb, dict(kl_fun=kl_exponential_mean, bounded=False)),
        ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=fisher_exponential, proj_interval=proj_interval)),
    ]:
        reg, tm = _multi(fn, arms, T, reps, **kw)
        print(f"  {name:12s} regret {reg.mean():7.1f} +- {reg.std():5.1f}   "
              f"time {tm.mean():6.3f}s")


def gaussian_experiment(T=50000, reps=5):
    """Quick sanity-check demo: Gaussian sigma=1, K=4, best=0.5. Fisher-UCB ~
    UCB1 since I_F is constant. Not used by the paper's reproduction pipeline."""
    arms = [GaussianArm(0.0), GaussianArm(0.2), GaussianArm(0.5), GaussianArm(0.1)]
    proj_interval = (-1e6, 1e6)  # projection inactive
    print(f"\n=== Gaussian (K={len(arms)}, T={T}, {reps} seeds, best=0.5) ===")
    for name, fn, kw in [
        ("UCB1",       run_ucb1, {}),
        ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=fisher_gaussian, proj_interval=proj_interval)),
    ]:
        reg, tm = _multi(fn, arms, T, reps, **kw)
        print(f"  {name:12s} regret {reg.mean():7.1f} +- {reg.std():5.1f}   "
              f"time {tm.mean():6.3f}s")

def beta_study(T=50000, reps=5):
    """Quick sanity-check demo: effect of beta on Fisher-UCB (fixed anytime
    level log t). Not used by the paper's reproduction pipeline."""
    arms = [BernoulliArm(0.30), BernoulliArm(0.35),
            BernoulliArm(0.40), BernoulliArm(0.32)]
    proj_interval = (0.01, 0.99)
    print(f"\n=== Beta study (Bernoulli, T={T}, {reps} seeds) ===")
    for beta in [1.0, 1.5, 2.0, 2.5]:
        reg, _ = _multi(run_fisher_ucb, arms, T, reps,
                        fisher_fn=fisher_bernoulli, proj_interval=proj_interval, beta=beta)
        print(f"  Fisher-UCB beta={beta:<4} regret {reg.mean():7.1f} +- {reg.std():5.1f}")


if __name__ == "__main__":
    bernoulli_experiment()
    gaussian_experiment()
    exponential_experiment()
    beta_study()
