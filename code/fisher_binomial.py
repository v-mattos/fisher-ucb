"""Fisher-UCB numerical routines and checks."""
from math import log


def make_fisher_binomial(n):

    def fisher_binomial(mu):
        return n / (mu * (n - mu))
    return fisher_binomial

def make_kl_binomial(n):

    def kl_binomial(mu_a, mu_b):
        eps = 1e-12
        pa = min(max(mu_a / n, eps), 1.0 - eps)
        pb = min(max(mu_b / n, eps), 1.0 - eps)
        return n * (pa * log(pa / pb) + (1 - pa) * log((1 - pa) / (1 - pb)))
    return kl_binomial
