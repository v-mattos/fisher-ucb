"""Fisher-UCB numerical routines and checks."""
import numpy as np
from math import log, ceil

class BernoulliArm:

    def __init__(self, p):
        assert 0.0 <= p <= 1.0
        self.p = p

    def sample(self, rng):
        return rng.binomial(1, self.p)

    def mean(self):
        return self.p




def fisher_gaussian(mu, sigma=1.0):
    return 1.0 / sigma ** 2


def kl_bernoulli(p, q):
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    q = min(max(q, eps), 1.0 - eps)
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))


def kl_ucb_index_bounded(p_hat, n, L, kl_fun, upper_bound=1.0, tol=1e-06, max_iter=50):
    if n == 0:
        return upper_bound
    c = L / n
    low, high = (p_hat, upper_bound)
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

def kl_ucb_index_unbounded(p_hat, n, L, kl_fun, tol=1e-06, max_iter=50):
    if n == 0:
        return p_hat + 1.0
    c = L / n
    low = max(p_hat, 1e-06)
    high = max(low * 2.0, 0.001)
    expanded = False
    while high < 1000000000.0:
        if kl_fun(p_hat, high) > c:
            expanded = True
            break
        high *= 2.0
    if not expanded:
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

def run_ucb1(arms, T, c=2.0, seed=0):
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

def run_fisher_ucb(arms, T, fisher_fn, C, beta=1.0, warmup=20, seed=0, mode='practical', Q=None, return_forced=False, snapshot_horizons=(), return_details=False):
    if mode not in {'practical', 'certified'}:
        raise ValueError('Unknown exploration mode')
    if not (beta > 0 and warmup >= 1 and T >= 1):
        raise ValueError('Invalid horizon, confidence parameter, or initialization')
    if mode == 'certified' and (Q is None or not np.isfinite(Q) or Q <= 0):
        raise ValueError("Quota mode requires a finite positive Q")
    rng = np.random.default_rng(seed)
    K = len(arms)
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K, dtype=float)
    regrets = np.zeros(T, dtype=float)
    true_means = np.array([arm.mean() for arm in arms])
    best_mean = np.max(true_means)
    gaps = best_mean - true_means
    lo, hi = C
    forced, forced_regret = (0, 0.0)
    snapshot_set, states = set(snapshot_horizons), {}
    for t in range(T):
        if t < K:
            a = t
            forced += 1
            forced_regret += gaps[a]
        else:
            m_t = warmup if mode == 'practical' else ceil(Q * log(t + 1))
            under = np.flatnonzero(counts < m_t)
            if under.size > 0:
                a = int(under[0])
                forced += 1
                forced_regret += gaps[a]
            else:
                Lt = log(max(t + 1, 2))
                mu_bar = np.clip(means, lo, hi)
                rad = beta * np.sqrt(Lt / (counts * fisher_fn(mu_bar)))
                a = int(np.argmax(means + rad))
        x = arms[a].sample(rng)
        counts[a] += 1
        means[a] += (x - means[a]) / counts[a]
        regrets[t] = gaps[a]
        if t + 1 in snapshot_set:
            states[t + 1] = (counts.copy(), means.copy())
    cum = np.cumsum(regrets)
    if return_details:
        return {'curve': cum, 'forced_pulls': forced, 'forced_regret': forced_regret, 'states': states}
    return (cum, forced, forced_regret) if return_forced else cum

def run_kl_ucb(arms, T, kl_fun, bounded=True, upper_bound=1.0, seed=0, snapshot_horizons=(), return_details=False):
    rng = np.random.default_rng(seed)
    K = len(arms)
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K, dtype=float)
    regrets = np.zeros(T, dtype=float)
    true_means = np.array([arm.mean() for arm in arms])
    best_mean = np.max(true_means)
    snapshot_set, states = set(snapshot_horizons), {}
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
                    ucb[i] = kl_ucb_index_bounded(means[i], counts[i], L, kl_fun, upper_bound)
                else:
                    ucb[i] = kl_ucb_index_unbounded(max(means[i], 1e-06), counts[i], L, kl_fun)
            a = int(np.argmax(ucb))
        x = arms[a].sample(rng)
        counts[a] += 1
        means[a] += (x - means[a]) / counts[a]
        regrets[t] = best_mean - true_means[a]
        if t + 1 in snapshot_set:
            states[t + 1] = (counts.copy(), means.copy())
    curve = np.cumsum(regrets)
    return {'curve': curve, 'states': states} if return_details else curve
