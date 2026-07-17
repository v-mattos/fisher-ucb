# -*- coding: utf-8 -*-
"""
gen_horizon_data.py
======================
Runs the horizon-scaling experiment (T up to 1e7) and saves the RAW results
to three CSVs:

  data/horizon_summary.csv      : final regret and time per (T, algorithm).
      Columns: T, algorithm, final_regret_mean, final_regret_std, runtime_mean
  data/horizon_regret_diff.csv  : regret difference (Fisher-KL) over time at
      the maximum horizon (T_max=1e7), sampled at ~1000 log-spaced points
      (we don't save all 1e7 points -- the file would be huge and the curve
      looks visually identical sampled on a log-x axis).
      Columns: round, regret_diff
  data/horizon_radius_ratio.csv : ratio eps_KL/eps_F for the tracked arm,
      same log-spaced sampling.
      Columns: round, radius_ratio

COST WARNING: KL-UCB at T=1e7 costs ~660s/seed (measured); this script
dominates the whole project's time budget (see CLAUDE.md). A low reps
(3-5) is intentional.

Run:
    python gen_horizon_data.py
"""
import csv
import numpy as np
from math import log, sqrt

from fisher_ucb import (BernoulliArm, fisher_bernoulli, kl_bernoulli,
                        run_fisher_ucb, run_kl_ucb, kl_ucb_index_bounded)

OUT_SUMMARY_CSV = "../data/horizon_summary.csv"
OUT_DIFF_CSV = "../data/horizon_regret_diff.csv"
OUT_RATIO_CSV = "../data/horizon_radius_ratio.csv"


def track_arm_states(arms, T, fisher_fn, proj_interval, beta=1.25, track_arm=1, seed=42):
    rng = np.random.default_rng(seed)
    K = len(arms)
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K, dtype=float)
    lo, hi = proj_interval
    history_means = np.zeros(T, dtype=float)
    history_counts = np.zeros(T, dtype=int)

    for t in range(T):
        if t < K:
            a = t
        else:
            Lt = log(max(t + 1, 2))
            mu_bar = np.clip(means, lo, hi)
            I = fisher_fn(mu_bar)
            rad = beta * np.sqrt(Lt / (counts * I))
            ucb = means + rad
            a = int(np.argmax(ucb))
        x = arms[a].sample(rng)
        counts[a] += 1
        means[a] += (x - means[a]) / counts[a]
        history_means[t] = means[track_arm]
        history_counts[t] = counts[track_arm]
    return history_means, history_counts


def compute_radius_ratio_csv(bern_arms, proj_interval, beta, T_max, track_arm, x_plot=None):
    """Runs Fisher-UCB (1 seed) tracking arm `track_arm`, computes the radius
    ratio eps_KL/eps_F over time, and writes OUT_RATIO_CSV. CHEAP (~1 run of
    Fisher-UCB at T_max, ~70s at T=1e7); does NOT redo the expensive regret loop.

    Tracking the OPTIMAL arm (n ~ t) shows the ratio CONVERGING smoothly to
    sqrt(2)/beta (the curvature mismatch vanishes as s = eps*sqrt(I_F) -> 0,
    because the optimal arm's empirical mean concentrates and the radius
    shrinks). Tracking a suboptimal arm (n ~ log t, so s = O(1) FIXED) shows
    the ratio NOT converging. This is the central theoretical distinction of
    the horizon-scaling analysis -- see the paper's Section 6.4."""
    if x_plot is None:
        x_plot = np.unique(np.geomspace(100, T_max, num=1000).astype(int))
    history_means, history_counts = track_arm_states(
        bern_arms, T_max, fisher_bernoulli, proj_interval, beta=beta, track_arm=track_arm)
    ratio_rows = []
    for t in x_plot:
        n = history_counts[t - 1]
        mu = history_means[t - 1]
        if n > 0:
            Lt = log(max(t, 2))
            mu_bar = min(max(mu, proj_interval[0]), proj_interval[1])
            rad_f = beta * sqrt(Lt / (n * fisher_bernoulli(mu_bar)))
            kl_idx = kl_ucb_index_bounded(mu, n, Lt, kl_bernoulli, upper_bound=1.0)
            rad_k = kl_idx - mu
            if rad_f > 0:
                ratio_rows.append((int(t), float(rad_k / rad_f)))
    with open(OUT_RATIO_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "radius_ratio"])
        w.writerows(ratio_rows)
    print(f"{OUT_RATIO_CSV} saved ({len(ratio_rows)} rows, track_arm={track_arm} [optimal]).")
    return x_plot


def main(horizons=(10**4, 10**5, 10**6, 10**7), reps=3):
    bern_arms = [BernoulliArm(0.30), BernoulliArm(0.35),
                 BernoulliArm(0.40), BernoulliArm(0.32)]
    proj_interval = (0.01, 0.99)
    beta = 1.25

    summary_rows = []
    regs_fisher_max, regs_kl_max = [], []
    T_max = max(horizons)

    for T in horizons:
        print(f"\n--- Horizon T = {T} ---")
        f_regs, k_regs = [], []
        for s in range(reps):
            reg_f = run_fisher_ucb(bern_arms, T, fisher_fn=fisher_bernoulli, proj_interval=proj_interval, beta=beta, seed=s)
            f_regs.append(reg_f[-1])
            reg_k = run_kl_ucb(bern_arms, T, kl_fun=kl_bernoulli, bounded=True, seed=s)
            k_regs.append(reg_k[-1])
            if T == T_max:
                regs_fisher_max.append(reg_f)
                regs_kl_max.append(reg_k)
        summary_rows.append((T, "Fisher-UCB", float(np.mean(f_regs)), float(np.std(f_regs)), None))
        summary_rows.append((T, "KL-UCB", float(np.mean(k_regs)), float(np.std(k_regs)), None))
        print(f"  Fisher-UCB : final regret = {np.mean(f_regs):.1f} +- {np.std(f_regs):.1f}")
        print(f"  KL-UCB     : final regret = {np.mean(k_regs):.1f} +- {np.std(k_regs):.1f}")

    mean_reg_fisher = np.mean(regs_fisher_max, axis=0)
    mean_reg_kl = np.mean(regs_kl_max, axis=0)
    regret_diff = mean_reg_fisher - mean_reg_kl

    # Track the OPTIMAL arm (not a suboptimal one): only the optimal arm has
    # n ~ t, s -> 0, and hence a convergent radius ratio. See
    # compute_radius_ratio_csv / the paper's Section 6.4.
    opt = int(np.argmax([a.mean() for a in bern_arms]))
    x_plot = np.unique(np.geomspace(100, T_max, num=1000).astype(int))
    diff_rows = [(int(t), float(regret_diff[t - 1])) for t in x_plot]

    with open(OUT_SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T", "algorithm", "final_regret_mean", "final_regret_std", "runtime_mean"])
        w.writerows(summary_rows)
    print(f"\n{OUT_SUMMARY_CSV} saved ({len(summary_rows)} rows).")

    with open(OUT_DIFF_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "regret_diff"])
        w.writerows(diff_rows)
    print(f"{OUT_DIFF_CSV} saved ({len(diff_rows)} rows).")

    compute_radius_ratio_csv(bern_arms, proj_interval, beta, T_max, track_arm=opt, x_plot=x_plot)


def regen_radius_ratio_only(T_max=10**7):
    """Regenerates ONLY data/horizon_radius_ratio.csv, tracking the optimal
    arm, without redoing the expensive regret loop (which already produced
    summary/diff). Useful for fixing panel (right) of the figure without
    paying the ~4.5h of the full experiment."""
    bern_arms = [BernoulliArm(0.30), BernoulliArm(0.35),
                 BernoulliArm(0.40), BernoulliArm(0.32)]
    opt = int(np.argmax([a.mean() for a in bern_arms]))
    compute_radius_ratio_csv(bern_arms, proj_interval=(0.01, 0.99), beta=1.25, T_max=T_max, track_arm=opt)


if __name__ == "__main__":
    main()
