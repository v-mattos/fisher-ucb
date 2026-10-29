"""Bernoulli inverse-KL solvers with a common absolute index-error target.

The returned value is a feasible lower bracket endpoint. No clipping of the
empirical Bernoulli mean changes the divergence at p=0 or p=1.
"""
from math import expm1, log, log1p, sqrt, nextafter
import numpy as np

Q_TOL = 1e-6


def divergence(p, q):
    if p == q:
        return 0.0
    if q == 1.0:
        return float('inf') if p < 1.0 else 0.0
    if p == 0.0:
        return -log1p(-q)
    return p * log(p / q) + (1.0 - p) * log1p((q - p) / (1.0 - q))


def divergence_vec(p, q):
    p, q = np.broadcast_arrays(np.asarray(p, float), np.asarray(q, float))
    out = np.zeros(p.shape)
    zero = (p == 0) & (q < 1)
    out[zero] = -np.log1p(-q[zero])
    interior = (p > 0) & (p < 1) & (q < 1)
    a, b = p[interior], q[interior]
    out[interior] = a * np.log(a / b) + (1 - a) * np.log1p((b - a) / (1 - b))
    out[(q == 1) & (p < 1)] = np.inf
    return out


def _validate(p, n, L, tol):
    if not (0 <= p <= 1 and n > 0 and L >= 0 and 0 < tol < 1):
        raise ValueError('Need 0<=p<=1, n>0, L>=0, 0<tol<1.')
    if not all(np.isfinite(v) for v in (p, n, L, tol)):
        raise ValueError('Non-finite solver input.')


def bisection(p, n, L, tol=Q_TOL):
    _validate(p, n, L, tol)
    if p == 1 or L == 0:
        return p
    c = L / n
    if p == 0:
        return min(-expm1(-c), nextafter(1.0, 0.0))
    lo, hi = p, min(1.0, p + sqrt(c / 2))  # Pinsker upper bracket.
    for _ in range(128):
        if hi - lo <= tol:
            return lo
        q = (lo + hi) / 2
        if divergence(p, q) <= c:
            lo = q
        else:
            hi = q
    raise ArithmeticError('Bisection failed to attain the index-error target.')


def newton_scalar(p, n, L, tol=Q_TOL):
    _validate(p, n, L, tol)
    if p == 1 or L == 0:
        return p
    c = L / n
    if p == 0:
        return min(-expm1(-c), nextafter(1.0, 0.0))
    lo, hi = p, min(1.0, p + sqrt(c / 2))
    # Local Fisher radius plus a positive skew/budget correction, safeguarded
    # by the Pinsker bracket. This is fresh initialization, not a time warm start.
    q = min(p + sqrt(2 * p * (1 - p) * c) + 2 * c / 3, hi)
    q = min(q, nextafter(1.0, 0.0))
    for _ in range(128):
        value = divergence(p, q) - c
        if value <= 0:
            lo = q
        else:
            hi = q
        if hi - lo <= tol:
            return lo
        derivative = (q - p) / (q * (1 - q))
        # Residual only gates the check; convergence requires a q bracket.
        if abs(value) <= 2 * tol * derivative:
            a, b = max(lo, q - tol / 2), min(hi, q + tol / 2)
            if divergence(p, a) <= c <= divergence(p, b):
                return a
        candidate = q - value / derivative if derivative > 0 else float('nan')
        q = candidate if lo < candidate < hi else (lo + hi) / 2
    raise ArithmeticError('Safeguarded Newton failed to attain the target.')


def newton_vector(p, n, L, tol=Q_TOL):
    p = np.asarray(p, float)
    n = np.broadcast_to(np.asarray(n, float), p.shape)
    if (p.ndim != 1 or not np.isfinite(p).all() or not np.isfinite(n).all()
            or not np.isfinite(L) or not np.isfinite(tol)
            or np.any((p < 0) | (p > 1)) or np.any(n <= 0)
            or L < 0 or not 0 < tol < 1):
        raise ValueError('Invalid vector solver inputs.')
    if L == 0:
        return p.copy()
    c = L / n
    result = p.copy()
    result[p == 0] = np.minimum(-np.expm1(-c[p == 0]), np.nextafter(1., 0.))
    active = (p > 0) & (p < 1)
    lo = p.copy()
    hi = np.minimum(1., p + np.sqrt(c / 2))
    q = np.minimum(p + np.sqrt(2 * p * (1 - p) * c) + 2 * c / 3, hi)
    q = np.minimum(q, np.nextafter(1., 0.))
    for _ in range(128):
        indices = np.flatnonzero(active)
        if not len(indices):
            return result
        x, z, budget = p[indices], q[indices], c[indices]
        value = divergence_vec(x, z) - budget
        below = value <= 0
        lo[indices] = np.where(below, z, lo[indices])
        hi[indices] = np.where(below, hi[indices], z)
        a = np.maximum(lo[indices], z - tol / 2)
        b = np.minimum(hi[indices], z + tol / 2)
        narrow = hi[indices] - lo[indices] <= tol
        derivative = (z - x) / (z * (1 - z))
        near = np.abs(value) <= 2 * tol * derivative
        certified = np.zeros(len(indices), dtype=bool)
        certified[near] = ((divergence_vec(x[near], a[near]) <= budget[near])
                           & (budget[near] <= divergence_vec(x[near], b[near])))
        done = narrow | certified
        result[indices[done]] = np.where(narrow, lo[indices], a)[done]
        active[indices[done]] = False
        remaining = indices[~done]
        z, x, value = z[~done], x[~done], value[~done]
        derivative = (z - x) / (z * (1 - z))
        candidate = z - value / np.maximum(derivative, np.finfo(float).tiny)
        valid = np.isfinite(candidate) & (candidate > lo[remaining]) & (candidate < hi[remaining])
        q[remaining] = np.where(valid, candidate, (lo[remaining] + hi[remaining]) / 2)
    raise ArithmeticError('Vector Newton failed to attain the target.')
