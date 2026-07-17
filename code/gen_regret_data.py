# -*- coding: utf-8 -*-
"""
gen_regret_data.py
===================
Runs the regret_curves.pdf simulations (Figure 1: Bernoulli + Exponential)
and saves the RAW results to data/regret_curves.csv. Plotting itself (colors,
styles, titles) lives in plot_regret_curves.py, which only READS this CSV --
so changing the figure's appearance never requires re-running the bandits.

Also measures each seed's wall-clock time (same runs, no extra cost) and
saves data/table3_summary.csv (Table 3 of the paper: Bernoulli/Exponential
runtime + speedup vs KL-UCB), since the paper measures runtime on the SAME
pipeline used for Table 1.

Schema of data/regret_curves.csv (long format), columns:
    round, panel, algorithm, mean_regret, std_regret, p95_regret, max_final_regret, n_seeds

  round            : round index t (0-indexed)
  panel            : "Bernoulli" or "Exponential"
  algorithm        : "UCB1" | "KL-UCB" | "Fisher-UCB"
  mean_regret      : mean cumulative regret over seeds, at round t
  std_regret       : SAMPLE standard deviation (ddof=1) of the cumulative
                      regret, at round t (same convention as
                      fisher_binomial.py::_final_stats, for a consistent 95%
                      CI: ci95_half = 1.96*std_regret/sqrt(n_seeds))
  p95_regret       : 95th percentile of the cumulative regret over seeds, at round t
  max_final_regret : final regret (t=T-1) maximum over seeds (constant per group)
  n_seeds          : number of seeds used (constant per group; stored here so
                      downstream scripts, e.g. gen_table_data.py, can compute a
                      CI without being told the rep count out of band)

Schema of data/table3_summary.csv:
    family, algorithm, runtime_mean, runtime_std, speedup_vs_klucb, n_seeds

Run:
    python gen_regret_data.py
"""
import csv
import time
import numpy as np

from fisher_ucb import (BernoulliArm, ExponentialArm,
                        fisher_bernoulli, fisher_exponential,
                        kl_bernoulli, kl_exponential_mean,
                        run_ucb1, run_fisher_ucb, run_kl_ucb)

OUT_CSV = "../data/regret_curves.csv"
OUT_TABLE3_CSV = "../data/table3_summary.csv"


def _curves_and_times(specs, arms, T, reps):
    out = {}
    for name, fn, kw in specs:
        times = np.zeros(reps)
        R = np.empty((reps, T))
        for s in range(reps):
            t0 = time.perf_counter()
            R[s] = fn(arms, T, seed=s, **kw)
            times[s] = time.perf_counter() - t0
        out[name] = (R, times)
    return out


def main(T=50000, reps=20):
    bern = [BernoulliArm(0.30), BernoulliArm(0.35),
            BernoulliArm(0.40), BernoulliArm(0.32)]
    specs_b = [
        ("UCB1",       run_ucb1,       {}),
        ("KL-UCB",     run_kl_ucb,     dict(kl_fun=kl_bernoulli, bounded=True)),
        ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=fisher_bernoulli, proj_interval=(0.01, 0.99), beta=1.25)),
    ]
    expo = [ExponentialArm(1.0), ExponentialArm(1.5),
            ExponentialArm(2.0), ExponentialArm(0.7)]
    specs_e = [
        ("UCB1",       run_ucb1,       {}),
        ("KL-UCB",     run_kl_ucb,     dict(kl_fun=kl_exponential_mean, bounded=False)),
        ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=fisher_exponential, proj_interval=(0.05, 1e6), beta=1.25)),
    ]

    rows = []
    table3_rows = []
    for panel, specs, arms in [("Bernoulli", specs_b, bern), ("Exponential", specs_e, expo)]:
        print(f"=== {panel}: {reps} seeds, T={T} ===")
        curves = _curves_and_times(specs, arms, T, reps)
        for name, (R, times) in curves.items():
            m = R.mean(0)
            sd = R.std(0, ddof=1) if reps > 1 else np.zeros(T)
            p95 = np.percentile(R, 95, axis=0)
            mx = R[:, -1].max()
            for t in range(T):
                rows.append((t, panel, name, m[t], sd[t], p95[t], mx, reps))
            print(f"  {name:12s}: final mean={m[-1]:.2f} std={sd[-1]:.2f}  "
                  f"runtime={times.mean():.3f}s+-{times.std():.3f}s")
            table3_rows.append([panel, name, times.mean(), times.std(ddof=1) if reps > 1 else 0.0])

    # add the speedup-vs-KL-UCB column
    kl_runtime = {panel: rt for panel, name, rt, _ in table3_rows if name == "KL-UCB"}
    table3_final = [(panel, name, rt, rtsd, kl_runtime[panel] / rt, reps)
                     for panel, name, rt, rtsd in table3_rows]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "panel", "algorithm", "mean_regret", "std_regret",
                    "p95_regret", "max_final_regret", "n_seeds"])
        w.writerows(rows)
    print(f"\n{OUT_CSV} saved ({len(rows)} rows).")

    with open(OUT_TABLE3_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "algorithm", "runtime_mean", "runtime_std",
                    "speedup_vs_klucb", "n_seeds"])
        w.writerows(table3_final)
    print(f"{OUT_TABLE3_CSV} saved ({len(table3_final)} rows).")


if __name__ == "__main__":
    main()
