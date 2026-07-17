# -*- coding: utf-8 -*-
"""
gen_poisson_data.py
====================
Runs the poisson_curves.pdf simulations (Figure 2) and saves the RAW results
to a CSV (default data/poisson_curves.csv; pass out_csv= for the paper's
higher-seed replication). See gen_regret_data.py for why gen/plot are split.

Two panels, same schema as gen_regret_data.py (round, panel, algorithm,
mean_regret, std_regret, p95_regret, max_final_regret, n_seeds):

  panel="small_gaps" : small gaps (means (2.0,2.1,2.2,1.9)), 3 algorithms,
                        reps seeds, T=50000. The paper's Table 1/Figure 2
                        (left) uses this panel at reps=20 (canonical) and,
                        as a separate higher-power replication, at reps=1000
                        (see reproduce_paper.py).
  panel="projection_ablation" : near-zero (means (0.10,0.15,0.20,0.08)),
                        Fisher-UCB with and without projection, reps_ablation
                        seeds, T=20000. Here p95_regret and max_final_regret
                        ARE the point (the projection effect is a TAIL
                        effect, hence the large reps_ablation).

Run:
    python gen_poisson_data.py
"""
import csv
import numpy as np

from fisher_poisson import PoissonArm, fisher_poisson, kl_poisson
from fisher_ucb import run_ucb1, run_fisher_ucb, run_kl_ucb

OUT_CSV = "../data/poisson_curves.csv"


def _run_all(specs, arms, T, reps):
    out = {}
    for name, fn, kw in specs:
        R = np.array([fn(arms, T, seed=s, **kw) for s in range(reps)])
        out[name] = R
    return out


def main(reps=40, reps_ablation=200, out_csv=OUT_CSV):
    rows = []

    # --- panel 1: small gaps, 3 algorithms, T=50000 ---
    T1 = 50000
    arms1 = [PoissonArm(2.0), PoissonArm(2.1), PoissonArm(2.2), PoissonArm(1.9)]
    specs1 = [
        ("UCB1",       run_ucb1,       {}),
        ("KL-UCB",     run_kl_ucb,     dict(kl_fun=kl_poisson, bounded=False)),
        ("Fisher-UCB", run_fisher_ucb, dict(fisher_fn=fisher_poisson, proj_interval=(0.05, 1e6), beta=1.25)),
    ]
    print(f"=== Poisson small_gaps: {reps} seeds, T={T1} ===")
    for name, R in _run_all(specs1, arms1, T1, reps).items():
        m = R.mean(0)
        sd = R.std(0, ddof=1) if reps > 1 else np.zeros(T1)
        p95 = np.percentile(R, 95, axis=0)
        mx = R[:, -1].max()
        for t in range(T1):
            rows.append((t, "small_gaps", name, m[t], sd[t], p95[t], mx, reps))
        print(f"  {name:12s}: final mean={m[-1]:.2f} std={sd[-1]:.2f} max={mx:.1f}")

    # --- panel 2: near-zero, projection ablation, T=20000 ---
    T2 = 20000
    arms2 = [PoissonArm(0.10), PoissonArm(0.15), PoissonArm(0.20), PoissonArm(0.08)]
    specs2 = [("Fisher-UCB (proj)", (0.05, 1e6)), ("Fisher-UCB (no proj)", (1e-9, 1e9))]
    print(f"=== Poisson projection_ablation: {reps_ablation} seeds, T={T2} ===")
    for name, proj_interval in specs2:
        R = np.array([run_fisher_ucb(arms2, T2, fisher_fn=fisher_poisson, proj_interval=proj_interval,
                                     beta=1.25, seed=s) for s in range(reps_ablation)])
        m = R.mean(0)
        sd = R.std(0, ddof=1) if reps_ablation > 1 else np.zeros(T2)
        p95 = np.percentile(R, 95, axis=0)
        mx = R[:, -1].max()
        for t in range(T2):
            rows.append((t, "projection_ablation", name, m[t], sd[t], p95[t], mx, reps_ablation))
        print(f"  {name:22s}: mean={m[-1]:.2f} max={mx:.1f}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "panel", "algorithm", "mean_regret", "std_regret",
                    "p95_regret", "max_final_regret", "n_seeds"])
        w.writerows(rows)
    print(f"\n{out_csv} saved ({len(rows)} rows).")


if __name__ == "__main__":
    main()
