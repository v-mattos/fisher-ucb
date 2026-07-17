# -*- coding: utf-8 -*-
"""
gen_table_data.py
====================
Builds the data CSVs for Tables 1 and 2 of the paper (Table 3 is written
directly by gen_regret_data.py, which measures runtime on the same runs used
for Table 1/Figure 1).

Does NOT run any bandits from scratch for Bernoulli/Exponential/Poisson/
Binomial -- it READS the last round of the trajectories already saved by
gen_regret_data.py, gen_poisson_data.py and gen_binomial_data.py (so those
scripts must have run first, including the paper's higher-seed replications:
Poisson at 1000 seeds via poisson_curves_n1000_merged.csv, Binomial at 100
seeds via binomial_curves_n100.csv -- see CLAUDE.md section 5). The ONLY new
simulation here is Gaussian (Table 1 mentions Fisher-UCB=UCB1 for Gaussian,
but this family has no panel of its own in any figure, so its final number
only needs to be generated here).

Outputs:
  data/table1_summary.csv : family, algorithm, final_regret_mean, ci95_half, n_seeds
      family in {Bernoulli, Exponential, "Poisson (20)", "Poisson (1000)", Gaussian}
      -- the two Poisson rows correspond to the two columns of tab:exp-family
  data/table2_summary.csv : panel, algorithm, final_regret_mean, ci95_half, n_seeds
      panel in {favorable, adverse}  (Binomial(10), 100 seeds, Table 2)

ci95_half = 1.96 * std_regret / sqrt(n_seeds), with std_regret already sample
(ddof=1) as saved by the current gen_*_data.py scripts.

Prerequisites (run once, beforehand): gen_regret_data.py, gen_poisson_data.py
(canonical + the 1000-seed replication merged into
poisson_curves_n1000_merged.csv), gen_binomial_data.py (at 100 seeds,
binomial_curves_n100.csv). See reproduce_paper.py for the full chain.

Run:
    python gen_table_data.py
"""
import csv
from collections import defaultdict

import numpy as np

from fisher_ucb import GaussianArm, fisher_gaussian, run_ucb1, run_fisher_ucb

IN_REGRET_CSV = "../data/regret_curves.csv"
IN_POISSON_CSV = "../data/poisson_curves.csv"
IN_POISSON_1000_CSV = "../data/poisson_curves_n1000_merged.csv"
IN_BINOMIAL_CSV = "../data/binomial_curves_n100.csv"
OUT_TABLE1_CSV = "../data/table1_summary.csv"
OUT_TABLE2_CSV = "../data/table2_summary.csv"

Z95 = 1.96


def _final_rows(path, panel_filter=None):
    """Reads a regret-curve-style CSV and returns, per (panel, algorithm),
    the row at the MAXIMUM round (the final regret). panel_filter, if given,
    keeps only those panels (e.g. Poisson wants only 'small_gaps', not
    'projection_ablation')."""
    best = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            panel = row["panel"]
            if panel_filter is not None and panel not in panel_filter:
                continue
            key = (panel, row["algorithm"])
            r = int(row["round"])
            if key not in best or r > best[key][0]:
                best[key] = (r, row)
    return {k: v[1] for k, v in best.items()}


def _ci95_half(std_regret, n_seeds):
    return Z95 * std_regret / np.sqrt(n_seeds)


def gaussian_summary(reps=20, T=50000):
    """Runs Gaussian (UCB1 + Fisher-UCB, no KL-UCB) and returns rows ready
    for table1_summary.csv.

    Gaussian has a CONSTANT I_F(mu)=1/sigma^2 (rho=0, projection inactive),
    so Fisher-UCB reduces to UCB1 *up to the choice of beta*: UCB1's radius
    is sqrt(log t/n) (effective beta = 1), while Fisher uses
    beta*sqrt(log t/n). Hence:
      - Fisher-UCB (beta=1.25)  -> explores ~25% more than UCB1 -> LARGER regret.
      - Fisher-UCB (beta=1)     -> SAME index as UCB1           -> ~EQUAL regret
                                   (the tiny residual comes only from the
                                   warm-up n0=20, which UCB1 doesn't have;
                                   with warmup=0 it would be bit-identical).
    We run all three variants to make this equivalence explicit in the
    table, instead of asserting "coincides" without showing the role of beta."""
    arms = [GaussianArm(0.0), GaussianArm(0.2), GaussianArm(0.5), GaussianArm(0.1)]
    specs = [
        ("UCB1",                run_ucb1,       {}),
        ("Fisher-UCB (b=1.25)", run_fisher_ucb, dict(fisher_fn=fisher_gaussian, proj_interval=(-1e6, 1e6), beta=1.25)),
        ("Fisher-UCB (b=1)",    run_fisher_ucb, dict(fisher_fn=fisher_gaussian, proj_interval=(-1e6, 1e6), beta=1.0)),
        # beta=1 AND warmup=0: reduces EXACTLY to UCB1 (identical radius, no
        # forced warm-up exploration). Should be bit-identical to UCB1.
        ("Fisher-UCB (b=1,w=0)", run_fisher_ucb, dict(fisher_fn=fisher_gaussian, proj_interval=(-1e6, 1e6), beta=1.0, warmup=0)),
    ]
    rows = []
    print(f"=== Gaussian (table-only, no figure): {reps} seeds, T={T} ===")
    for name, fn, kw in specs:
        R = np.array([fn(arms, T, seed=s, **kw) for s in range(reps)])
        finals = R[:, -1]
        mean = finals.mean()
        sd = finals.std(ddof=1) if reps > 1 else 0.0
        rows.append(("Gaussian", name, mean, _ci95_half(sd, reps), reps))
        print(f"  {name:22s}: final mean={mean:.2f} std={sd:.2f}")
    return rows


def main(reps_gaussian=20):
    rows1 = []

    regret_final = _final_rows(IN_REGRET_CSV)
    for (panel, algo), row in regret_final.items():
        n = int(row["n_seeds"])
        rows1.append((panel, algo, float(row["mean_regret"]),
                      _ci95_half(float(row["std_regret"]), n), n))

    poisson_final = _final_rows(IN_POISSON_CSV, panel_filter={"small_gaps"})
    for (panel, algo), row in poisson_final.items():
        n = int(row["n_seeds"])
        rows1.append(("Poisson (20)", algo, float(row["mean_regret"]),
                      _ci95_half(float(row["std_regret"]), n), n))

    poisson_1000_final = _final_rows(IN_POISSON_1000_CSV, panel_filter={"small_gaps"})
    for (panel, algo), row in poisson_1000_final.items():
        n = int(row["n_seeds"])
        rows1.append(("Poisson (1000)", algo, float(row["mean_regret"]),
                      _ci95_half(float(row["std_regret"]), n), n))

    rows1.extend(gaussian_summary(reps=reps_gaussian))

    with open(OUT_TABLE1_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "algorithm", "final_regret_mean", "ci95_half", "n_seeds"])
        w.writerows(rows1)
    print(f"\n{OUT_TABLE1_CSV} saved ({len(rows1)} rows).")

    rows2 = []
    binomial_final = _final_rows(IN_BINOMIAL_CSV)
    for (panel, algo), row in binomial_final.items():
        n = int(row["n_seeds"])
        rows2.append((panel, algo, float(row["mean_regret"]),
                      _ci95_half(float(row["std_regret"]), n), n))

    with open(OUT_TABLE2_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "algorithm", "final_regret_mean", "ci95_half", "n_seeds"])
        w.writerows(rows2)
    print(f"{OUT_TABLE2_CSV} saved ({len(rows2)} rows).")


if __name__ == "__main__":
    main()
