# -*- coding: utf-8 -*-
"""
reproduce_paper.py
====================
SINGLE ENTRY POINT that reproduces every figure and table used by
main_submission.tex: runs the bandit simulations, builds the data CSVs,
plots the 5 PDFs the paper includes, and copies them into latex/images/.

Reproduces exactly (no more, no less):
  - Figure 1  : regret_curves.pdf              (Bernoulli + Exponential, 20 seeds)
  - Figure 2  : poisson_curves_n1000_merged.pdf (small-gap panel at 1000 seeds,
                                                  projection-ablation panel at 200 seeds)
  - Figure 3  : binomial_curves_n100.pdf        (mirrored favorable/adverse, 100 seeds)
  - Figure    : horizon_scaling.pdf             (T up to 1e7, 20 seeds)
  - Figure    : scaling_runtime.pdf             (K in {4,20,50,100}, 20 seeds)
  - Table 1   : data/table1_summary.csv         (Bernoulli/Exponential/Poisson(20)/
                                                  Poisson(1000)/Gaussian)
  - Table 2   : data/table2_summary.csv         (Binomial favorable/adverse, 100 seeds)
  - Table 3   : data/table3_summary.csv         (runtime, written by the regret step)

TIME BUDGET (measured on an Intel Core i5-12450H; machine-dependent, see
CLAUDE.md section 6b): ~6-7h total, dominated by the horizon step (~4.5h,
KL-UCB at T=1e7 alone costs ~660s/seed) and the 1000-seed Poisson replication
(~1h). Ctrl+C is safe at ANY point: each step only writes its CSV/PDF when it
finishes, so a partial run leaves every completed artifact valid and does not
corrupt anything.

Usage:
    python reproduce_paper.py            # full reproduction (~6-7h)
    python reproduce_paper.py --smoke    # fast structural check (seconds), in an
                                          # isolated scratch directory -- touches
                                          # nothing under data/, figures/, or latex/
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import gen_regret_data
import gen_binomial_data
import gen_poisson_data
import gen_arms_scaling_data
import gen_table_data
import gen_horizon_data

import plot_regret_curves
import plot_poisson_curves
import plot_binomial_curves
import plot_arms_scaling_curves
import plot_horizon_curves

# Paths for the paper's higher-seed replications (Poisson 1000 seeds,
# Binomial 100 seeds) -- distinct from the canonical 20-seed CSVs so neither
# overwrites the other (see CLAUDE.md section 1a).
POISSON_1000_CSV = "../data/poisson_curves_n1000_merged.csv"
POISSON_1000_PDF = "../figures/poisson_curves_n1000_merged.pdf"
BINOMIAL_100_CSV = "../data/binomial_curves_n100.csv"
BINOMIAL_100_PDF = "../figures/binomial_curves_n100.pdf"

# The exact 5 figures main_submission.tex includes, in the order copied into
# latex/images/ (see CLAUDE.md section 10).
FIGURES_FOR_PAPER = [
    "../figures/regret_curves.pdf",
    POISSON_1000_PDF,
    BINOMIAL_100_PDF,
    "../figures/horizon_scaling.pdf",
    "../figures/scaling_runtime.pdf",
]

LATEX_IMAGES_DIR = "../latex/images"


def _step(desc, fn, *args, **kwargs):
    print(f"\n{'='*70}\n>>> {desc}\n{'='*70}")
    t0 = time.perf_counter()
    try:
        fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f">>> OK ({dt:.1f}s)")
        return True
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f">>> FAILED after {dt:.1f}s: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def run_data_steps(reps_tables=20, reps_poisson_ablation=200, reps_arms_scaling=20,
                    reps_horizon=20, reps_binomial_100=100, reps_poisson_1000=1000,
                    T=50000, K_list=(4, 20, 50, 100),
                    horizons=(10**4, 10**5, 10**6, 10**7)):
    """Runs every data-generation step in cheapest-to-priciest order (so a
    Ctrl+C early still leaves the cheap, high-value CSVs -- Tables 1-3 --
    already written)."""
    results = {}
    results["regret"] = _step(
        "1/7 regret: Fig 1 + Table 1 (Bernoulli/Exponential) + Table 3 (runtime)",
        gen_regret_data.main, T=T, reps=reps_tables)
    results["binomial_100"] = _step(
        "2/7 binomial (100 seeds): Fig 3 + Table 2",
        gen_binomial_data.main, T=T, reps=reps_binomial_100, out_csv=BINOMIAL_100_CSV)
    results["poisson_20"] = _step(
        "3/7 poisson (20 seeds, canonical): Table 1 'Poisson (20)' column",
        gen_poisson_data.main, reps=reps_tables, reps_ablation=reps_poisson_ablation)
    results["arms_scaling"] = _step(
        "4/7 arms_scaling: runtime-vs-K (Fig scaling_runtime)",
        gen_arms_scaling_data.main, T=T, reps=reps_arms_scaling, K_list=K_list)
    results["poisson_1000"] = _step(
        "5/7 poisson (1000 seeds): Fig 2 + Table 1 'Poisson (1000)' column",
        gen_poisson_data.main, reps=reps_poisson_1000, reps_ablation=reps_poisson_ablation,
        out_csv=POISSON_1000_CSV)
    results["table_data"] = _step(
        "6/7 table_data: assembles Tables 1-2 + runs Gaussian",
        gen_table_data.main, reps_gaussian=reps_tables)
    results["horizon"] = _step(
        "7/7 horizon: T up to 1e7 (the most expensive step, ~4.5h)",
        gen_horizon_data.main, horizons=horizons, reps=reps_horizon)
    return results


def run_plot_steps():
    """Plots every figure from its CSV (seconds, no bandit runs)."""
    results = {}
    results["regret_curves.pdf"] = _step(
        "Fig 1: regret_curves.pdf", plot_regret_curves.main)
    results["poisson_curves_n1000_merged.pdf"] = _step(
        "Fig 2: poisson_curves_n1000_merged.pdf",
        plot_poisson_curves.main, in_csv=POISSON_1000_CSV, out_pdf=POISSON_1000_PDF)
    results["binomial_curves_n100.pdf"] = _step(
        "Fig 3: binomial_curves_n100.pdf",
        plot_binomial_curves.main, in_csv=BINOMIAL_100_CSV, out_pdf=BINOMIAL_100_PDF)
    results["scaling_runtime.pdf"] = _step(
        "Fig: scaling_runtime.pdf", plot_arms_scaling_curves.plot_runtime)
    results["horizon_scaling.pdf"] = _step(
        "Fig: horizon_scaling.pdf", plot_horizon_curves.main)
    return results


def copy_figures_to_latex_images():
    """Copies the 5 paper figures into latex/images/ (see CLAUDE.md section
    10 -- these are plain copies, not a graphicspath link, by the author's
    choice)."""
    os.makedirs(LATEX_IMAGES_DIR, exist_ok=True)
    for src in FIGURES_FOR_PAPER:
        shutil.copy(src, LATEX_IMAGES_DIR)
        print(f"copied {src} -> {LATEX_IMAGES_DIR}/")


def main():
    print("reproduce_paper.py -- reproducing every figure and table used by the paper")
    print("Budget estimate: ~6-7h total, dominated by the horizon step (~4.5h) and the")
    print("1000-seed Poisson replication (~1h). Ctrl+C is safe at any point -- each step")
    print("only writes its CSV/PDF when it finishes.\n")

    data_results = run_data_steps()
    plot_results = run_plot_steps()
    copy_figures_to_latex_images()

    all_results = {**data_results, **plot_results}
    print(f"\n{'='*70}\nSummary\n{'='*70}")
    for name, ok in all_results.items():
        print(f"  {name:40s}: {'OK' if ok else 'FAILED'}")
    if all(all_results.values()):
        print("\nAll data, figures, and latex/images/ copies regenerated successfully.")
    else:
        print("\nSome steps failed (see above) -- steps that succeeded already have valid output.")


def run_smoke_test():
    """
    Fast structural check: runs the full orchestration with tiny T/reps in an
    ISOLATED scratch directory (never touches data/, figures/, or latex/).
    Confirms the plumbing (out_csv/in_csv/out_pdf parameters, step ordering,
    module dependencies) works end-to-end in seconds rather than hours.
    """
    scratch = tempfile.mkdtemp(prefix="fisher_ucb_smoke_")
    os.makedirs(os.path.join(scratch, "data"))
    os.makedirs(os.path.join(scratch, "figures"))
    print(f"[--smoke] scratch directory: {scratch}")

    def p(*parts):
        return os.path.join(scratch, *parts)

    # Redirect every module-level path constant into the scratch tree, so
    # nothing under the real data/, figures/, latex/ is ever touched.
    gen_regret_data.OUT_CSV = p("data", "regret_curves.csv")
    gen_regret_data.OUT_TABLE3_CSV = p("data", "table3_summary.csv")
    gen_binomial_data.OUT_CSV = p("data", "binomial_curves.csv")
    binom_100_csv = p("data", "binomial_curves_n100.csv")
    gen_poisson_data.OUT_CSV = p("data", "poisson_curves.csv")
    poisson_1000_csv = p("data", "poisson_curves_n1000_merged.csv")
    gen_arms_scaling_data.OUT_RUNTIME_CSV = p("data", "arms_scaling_runtime.csv")
    gen_arms_scaling_data.OUT_REGRET_CSV = p("data", "arms_scaling_regret_k100.csv")
    gen_horizon_data.OUT_SUMMARY_CSV = p("data", "horizon_summary.csv")
    gen_horizon_data.OUT_DIFF_CSV = p("data", "horizon_regret_diff.csv")
    gen_horizon_data.OUT_RATIO_CSV = p("data", "horizon_radius_ratio.csv")
    gen_table_data.IN_REGRET_CSV = gen_regret_data.OUT_CSV
    gen_table_data.IN_POISSON_CSV = gen_poisson_data.OUT_CSV
    gen_table_data.IN_POISSON_1000_CSV = poisson_1000_csv
    gen_table_data.IN_BINOMIAL_CSV = binom_100_csv
    gen_table_data.OUT_TABLE1_CSV = p("data", "table1_summary.csv")
    gen_table_data.OUT_TABLE2_CSV = p("data", "table2_summary.csv")

    # NOTE: gen_poisson_data.main / gen_binomial_data.main declare `out_csv=OUT_CSV` as a
    # default parameter, which Python binds at function-DEFINITION time -- monkeypatching
    # the module's OUT_CSV attribute afterwards does NOT change that already-bound default.
    # So `out_csv=` must always be passed EXPLICITLY here for both calls, never relied on
    # implicitly, or a smoke run silently writes into the real ../data/ directory instead of
    # the scratch one (this bit us once during development: it clobbered the real, canonical
    # data/poisson_curves.csv with 2-seed smoke data before this comment was added).
    ok = True
    ok &= _step("[smoke] regret", gen_regret_data.main, T=200, reps=2)
    ok &= _step("[smoke] binomial (100-seed slot)", gen_binomial_data.main,
                T=200, reps=2, out_csv=binom_100_csv)
    ok &= _step("[smoke] poisson (20-seed slot)", gen_poisson_data.main,
                reps=2, reps_ablation=2, out_csv=gen_poisson_data.OUT_CSV)
    ok &= _step("[smoke] arms_scaling", gen_arms_scaling_data.main,
                T=200, reps=2, K_list=(4, 8))
    ok &= _step("[smoke] poisson (1000-seed slot)", gen_poisson_data.main,
                reps=2, reps_ablation=2, out_csv=poisson_1000_csv)
    ok &= _step("[smoke] table_data", gen_table_data.main, reps_gaussian=2)
    ok &= _step("[smoke] horizon", gen_horizon_data.main, horizons=(100, 200), reps=1)

    plot_regret_curves.IN_CSV = gen_regret_data.OUT_CSV
    plot_regret_curves.OUT_PDF = p("figures", "regret_curves.pdf")
    plot_arms_scaling_curves.IN_RUNTIME_CSV = gen_arms_scaling_data.OUT_RUNTIME_CSV
    plot_arms_scaling_curves.OUT_RUNTIME_PDF = p("figures", "scaling_runtime.pdf")
    plot_horizon_curves.IN_DIFF_CSV = gen_horizon_data.OUT_DIFF_CSV
    plot_horizon_curves.IN_RATIO_CSV = gen_horizon_data.OUT_RATIO_CSV
    plot_horizon_curves.OUT_PDF = p("figures", "horizon_scaling.pdf")

    ok &= _step("[smoke] plot regret", plot_regret_curves.main)
    ok &= _step("[smoke] plot poisson", plot_poisson_curves.main,
                in_csv=poisson_1000_csv, out_pdf=p("figures", "poisson_curves_n1000_merged.pdf"))
    ok &= _step("[smoke] plot binomial", plot_binomial_curves.main,
                in_csv=binom_100_csv, out_pdf=p("figures", "binomial_curves_n100.pdf"))
    ok &= _step("[smoke] plot arms_scaling (runtime)", plot_arms_scaling_curves.plot_runtime)
    ok &= _step("[smoke] plot horizon", plot_horizon_curves.main)

    print(f"\n[--smoke] {'ALL STEPS OK' if ok else 'SOME STEPS FAILED'} -- scratch dir: {scratch}")
    print("[--smoke] Nothing under data/, figures/, or latex/ was touched.")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true",
                        help="Fast structural check with tiny T/reps in an isolated "
                             "scratch directory; does not touch real data/figures/latex.")
    args = parser.parse_args()
    if args.smoke:
        ok = run_smoke_test()
        sys.exit(0 if ok else 1)
    else:
        main()
