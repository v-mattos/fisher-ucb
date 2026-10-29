"""Fisher-UCB numerical routines and checks."""
from math import log, sqrt
import numpy as np
from fisher_ucb import kl_bernoulli

def _kl_bern_vec(x, q):
    e = 1e-15
    x = np.clip(x, e, 1 - e)
    q = np.clip(q, e, 1 - e)
    return x * np.log(x / q) + (1 - x) * np.log((1 - x) / (1 - q))

def klucb_newton_vec(x, n, L, iters=60, tol=1e-10):
    x = np.asarray(x, float)
    n = np.asarray(n, float)
    c = L / n
    q = np.full_like(x, 1 - 1e-12)
    done = _kl_bern_vec(x, q) <= c
    q[done] = 1.0
    for _ in range(iters):
        act = ~done
        if not act.any():
            break
        xa, qa = (x[act], q[act])
        f = _kl_bern_vec(xa, qa) - c[act]
        fp = (qa - xa) / (qa * (1 - qa))
        step = f / np.maximum(fp, 1e-300)
        qn = np.maximum(qa - step, xa)
        conv = np.abs(f) < tol * np.maximum(c[act], 1.0)
        q[act] = qn
        idx = np.flatnonzero(act)
        done[idx[conv]] = True
    # A completed iteration budget is not a convergence certificate.
    for i in np.flatnonzero(~done):
        q[i] = klucb_bisect_hp(float(x[i]), float(n[i]), L)
    return q

def klucb_newton_scalar(x, n, L):
    c = L / n
    if kl_bernoulli(x, 1 - 1e-12) <= c:
        return 1.0
    q = 1 - 1e-12
    for _ in range(60):
        f = kl_bernoulli(x, q) - c
        fp = (q - x) / (q * (1 - q))
        qn = max(q - f / max(fp, 1e-300), x)
        if abs(f) < 1e-10 * max(c, 1.0):
            return qn
        q = qn
    lo, hi = (x, 1.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(x, mid) > c:
            hi = mid
        else:
            lo = mid
    return lo

def klucb_bisect_hp(x, n, L, tol=1e-12):
    c = L / n
    lo, hi = (x, 1.0)
    if kl_bernoulli(x, hi - 1e-15) <= c:
        return 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(x, mid) > c:
            hi = mid
        else:
            lo = mid
    return lo

def newton_check(path):
    xs = np.concatenate([[0.0, 0.0001, 0.001, 0.01], np.linspace(0.02, 0.98, 25), [0.99, 0.999, 1.0]])
    ns = [1, 2, 5, 20, 100, 1000, 10000]
    Ls = [log(2), log(1000), log(50000)]
    worst_v = worst_s = 0.0
    for L in Ls:
        for n in ns:
            ref = np.array([klucb_bisect_hp(x, n, L) for x in xs])
            qv = klucb_newton_vec(xs, np.full_like(xs, n), L)
            qs = np.array([klucb_newton_scalar(x, n, L) for x in xs])
            worst_v = max(worst_v, np.max(np.abs(qv - ref)))
            worst_s = max(worst_s, np.max(np.abs(qs - ref)))
    msg = f'Newton verification against bisection tolerance 1e-12 over {len(xs) * len(ns) * len(Ls)} cases\n  max |q_newton_vec - q_ref| = {worst_v:.3e}\n  max |q_newton_scalar - q_ref| = {worst_s:.3e}\n'
    open(path, 'w').write(msg)
    return max(worst_v, worst_s) < 1e-08

def _loop(arms, T, seed, choose):
    rng = np.random.default_rng(seed)
    K = len(arms)
    cnt = np.zeros(K, int)
    mh = np.zeros(K)
    ssq = np.zeros(K)
    mu = np.array([a.mean() for a in arms])
    gaps = mu.max() - mu
    reg = np.zeros(T)
    for t in range(T):
        a = t if t < K else choose(t, cnt, mh, ssq)
        x = arms[a].sample(rng)
        cnt[a] += 1
        mh[a] += (x - mh[a]) / cnt[a]
        ssq[a] += x * x
        reg[t] = gaps[a]
    return np.cumsum(reg)

def run_klucb_newton(arms, T, seed=0, vectorized=True):

    def choose(t, cnt, mh, ssq):
        L = log(t + 1)
        if vectorized:
            idx = klucb_newton_vec(mh, cnt, L)
        else:
            idx = np.array([klucb_newton_scalar(mh[i], cnt[i], L) for i in range(len(mh))])
        return int(np.argmax(idx))
    return _loop(arms, T, seed, choose)

def run_ucbv(arms, T, seed=0, b=1.0):

    def choose(t, cnt, mh, ssq):
        L = log(t + 1)
        var = np.maximum(ssq / cnt - mh ** 2, 0.0)
        return int(np.argmax(mh + np.sqrt(2 * var * L / cnt) + 3 * b * L / cnt))
    return _loop(arms, T, seed, choose)

def run_endpoint(arms, T, seed=0, beta=sqrt(2.0)):

    def choose(t, cnt, mh, ssq):
        s2 = beta ** 2 * log(t + 1) / cnt
        qW = (2 * mh + s2 + np.sqrt(s2 ** 2 + 4 * s2 * mh * (1 - mh))) / (2 * (1 + s2))
        return int(np.argmax(qW))
    return _loop(arms, T, seed, choose)
