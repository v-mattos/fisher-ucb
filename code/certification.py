"""Check configurations for the final theorem; simulated rewards use exact families."""
from math import sqrt
import numpy as np
from certified_benchmark import local_constants, beta0


def certificate(fam, beta, gamma, Q, projection):
    if not 0 < gamma <= .5:
        raise ValueError('gamma must be in (0, 0.5]')
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be positive and finite')
    lo, hi = projection
    if not lo < hi:
        raise ValueError('Invalid projection interval')
    domain_lo, domain_hi = fam['domain']
    interior_compact = domain_lo < lo < hi < domain_hi
    contains = all(lo < m < hi for m in fam['means'])
    rows, required = local_constants(fam, gamma, beta)
    passed = bool(interior_compact and contains and beta > beta0(gamma)
                  and Q is not None and np.isfinite(Q) and Q >= required)
    return dict(certified=passed, exact_reference_rewards=True,
                projection_interior=bool(interior_compact), means_interior=bool(contains),
                beta=float(beta), gamma=float(gamma), beta0=beta0(gamma),
                Q=Q, Q_star=required, arms=rows)


def computable_bernoulli_quota(K, J, beta=2.2, gamma=.25):
    """Analytic extrema for the manuscript's Bernoulli Q_J examples."""
    a, b = K; c, d = J
    if not (0 < c < a < b < d < 1 and 0 < gamma <= .5 and beta > 0):
        raise ValueError('Need interior K strictly inside J, and valid beta/gamma')
    delta = min(a-c, d-b)
    variance_max = .25 if a <= .5 <= b else max(a*(1-a), b*(1-b))
    I_min = 1/variance_max
    rho = lambda m: (2*m-1)/sqrt(m*(1-m))
    C_J = max(abs(rho(c)), abs(rho(d)))
    theta = min(delta*sqrt(I_min), 2*gamma/C_J if C_J else np.inf)
    return dict(K=list(K), J=list(J), beta=beta, gamma=gamma,
                delta=delta, I_min=I_min, C_J=C_J, theta_J=theta,
                Q_J=beta**2*(1-gamma)**2/theta**2)
