# -*- coding: utf-8 -*-
"""
gen_arms_scaling_data.py
==========================
Runs the K-scaling (number of arms) experiment and saves the RAW results to
two CSVs:

  data/arms_scaling_runtime.csv : mean runtime per (K, family, algorithm).
      Columns: K, family, algorithm, runtime_mean, runtime_std
  data/arms_scaling_regret_k100.csv : regret trajectories at K=100 (the
      largest K tested), same schema as gen_regret_data.py (round,
      panel="Bernoulli"/"Exponential", algorithm, mean_regret, std_regret,
      p95_regret, max_final_regret).

Runtime dominates the cost (KL-UCB does one bisection per arm per round);
see CLAUDE.md for time estimates per seed count.

Run:
    python gen_arms_scaling_data.py
"""
import csv
import time
import numpy as np

from fisher_ucb import (BernoulliArm, ExponentialArm,
                        fisher_bernoulli, fisher_exponential,
                        kl_bernoulli, kl_exponential_mean,
                        run_ucb1, run_fisher_ucb, run_kl_ucb)

OUT_RUNTIME_CSV = "../data/arms_scaling_runtime.csv"
OUT_REGRET_CSV = "../data/arms_scaling_regret_k100.csv"


def generate_bernoulli_arms(K):
    means = np.linspace(0.1, 0.7, K - 1).tolist() + [0.8]
    return [BernoulliArm(m) for m in means]


def generate_exponential_arms(K):
    means = np.linspace(0.5, 4.0, K - 1).tolist() + [5.0]
    return [ExponentialArm(m) for m in means]


def main(K_list=(4, 20, 50, 100), T=50000, reps=5, K_regret=100):
    runtime_rows = []
    regret_rows = []

    for K in K_list:
        print(f"\n--- K={K} ---")
        bern_arms, exp_arms = generate_bernoulli_arms(K), generate_exponential_arms(K)
        specs_bern = [
            ("UCB1",       run_ucb1,       {}),
            ("KL-UCB",     run_kl_ucb,     dict(kl_fun=kl_bernoulli, bounded=True)),
            ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=fisher_bernoulli, proj_interval=(0.01, 0.99), beta=1.25)),
        ]
        specs_exp = [
            ("UCB1",       run_ucb1,       {}),
            ("KL-UCB",     run_kl_ucb,     dict(kl_fun=kl_exponential_mean, bounded=False)),
            ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=fisher_exponential, proj_interval=(0.05, 1e6), beta=1.25)),
        ]

        for family, specs, arms in [("Bernoulli", specs_bern, bern_arms),
                                     ("Exponential", specs_exp, exp_arms)]:
            for name, fn, kw in specs:
                times, finals = [], []
                for s in range(reps):
                    t0 = time.perf_counter()
                    r = fn(arms, T, seed=s, **kw)
                    times.append(time.perf_counter() - t0)
                    finals.append(r)
                times = np.array(times)
                runtime_rows.append((K, family, name, times.mean(), times.std()))
                print(f"  {family:12s} {name:12s} (K={K}): time {times.mean():.4f}s +- {times.std():.4f}s")

                if K == K_regret:
                    R = np.array(finals)
                    m = R.mean(0)
                    sd = R.std(0, ddof=1) if reps > 1 else np.zeros(T)
                    p95 = np.percentile(R, 95, axis=0)
                    mx = R[:, -1].max()
                    for t in range(T):
                        regret_rows.append((t, family, name, m[t], sd[t], p95[t], mx, reps))

    with open(OUT_RUNTIME_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "family", "algorithm", "runtime_mean", "runtime_std"])
        w.writerows(runtime_rows)
    print(f"\n{OUT_RUNTIME_CSV} saved ({len(runtime_rows)} rows).")

    with open(OUT_REGRET_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "panel", "algorithm", "mean_regret", "std_regret",
                    "p95_regret", "max_final_regret", "n_seeds"])
        w.writerows(regret_rows)
    print(f"{OUT_REGRET_CSV} saved ({len(regret_rows)} rows, K={K_regret}).")


if __name__ == "__main__":
    main()
