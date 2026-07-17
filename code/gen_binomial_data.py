# -*- coding: utf-8 -*-
"""
gen_binomial_data.py
======================
Runs the binomial_curves.pdf simulations (Figure 3: sign reversal) and saves
the RAW results to a CSV (default data/binomial_curves.csv; pass out_csv= for
the paper's 100-seed replication). Same schema as gen_regret_data.py (round,
panel, algorithm, mean_regret, std_regret, p95_regret, max_final_regret,
n_seeds); panel="favorable" (rho<0) or "adverse" (rho>0). gen_table_data.py
reads the last round of this CSV to build the paper's Table 2.

Run:
    python gen_binomial_data.py
"""
import csv
import numpy as np

from fisher_binomial import BinomialArm, make_fisher_binomial, make_kl_binomial
from fisher_ucb import run_ucb1, run_fisher_ucb, run_kl_ucb

OUT_CSV = "../data/binomial_curves.csv"


def main(n=10, T=50000, reps=20, out_csv=OUT_CSV):
    ff = make_fisher_binomial(n)
    kf = make_kl_binomial(n)

    fav = [BinomialArm(n, 0.08), BinomialArm(n, 0.15),
           BinomialArm(n, 0.22), BinomialArm(n, 0.12)]
    adv = [BinomialArm(n, 0.92), BinomialArm(n, 0.85),
           BinomialArm(n, 0.78), BinomialArm(n, 0.88)]

    rows = []
    for panel, arms in [("favorable", fav), ("adverse", adv)]:
        print(f"=== Binomial n={n} {panel}: {reps} seeds, T={T} ===")
        specs = [
            ("UCB1",       run_ucb1,       {}),
            ("KL-UCB",     run_kl_ucb,     dict(kl_fun=kf, bounded=True, upper_bound=float(n))),
            ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=ff, proj_interval=(0.05, n - 0.05), beta=1.25)),
        ]
        for name, fn, kw in specs:
            R = np.array([fn(arms, T, seed=s, **kw) for s in range(reps)])
            m = R.mean(0)
            sd = R.std(0, ddof=1) if reps > 1 else np.zeros(T)
            p95 = np.percentile(R, 95, axis=0)
            mx = R[:, -1].max()
            for t in range(T):
                rows.append((t, panel, name, m[t], sd[t], p95[t], mx, reps))
            print(f"  {name:12s}: final mean={m[-1]:.2f} std={sd[-1]:.2f}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "panel", "algorithm", "mean_regret", "std_regret",
                    "p95_regret", "max_final_regret", "n_seeds"])
        w.writerows(rows)
    print(f"\n{out_csv} saved ({len(rows)} rows).")


if __name__ == "__main__":
    main()
