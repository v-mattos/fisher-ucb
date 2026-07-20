# Fisher-UCB — code

Code accompanying *"Fisher-UCB: A Closed-Form Surrogate for KL-UCB on
Exponential-Family Bandits"*. Fisher-UCB replaces
KL-UCB's per-round numerical inverse-KL optimization with a single
closed-form evaluation of the Fisher information at a projected empirical
mean, and quantifies the resulting finite-horizon regret price with a
first-order, signed gap bound.

## Requirements

- Python 3.12+
- `numpy`, `scipy`, `matplotlib`
- `sympy` (optional; only needed for the symbolic derivation in
  `verification/Kgap_c2.py` — its numeric checks and the ones in
  `verification/binom_verify.py`/`poisson_verify.py` corroborate the same
  claim without it)

```bash
pip install -r requirements.txt
```

## Reproducing the paper

```bash
cd code
python reproduce_paper.py
```

This single entry point runs every bandit simulation, builds the three
tables' data CSVs, plots the five figures the paper includes, and copies
them into `../latex/images/`.

`../data/` ships as a single `data.zip`, the exact CSVs used to write the
paper.

**Time budget**: ~6 hours end to end on a modest laptop CPU (measured on
an Intel Core i5-12450H). The horizon experiment
(T up to 1e7) alone takes ~4.5 hours, and the 1000-seed Poisson replication
takes ~1 hour; everything else finishes in minutes. **Ctrl+C is safe at any
point** — each step only writes its CSV/PDF when it finishes, so an
interrupted run leaves every already-completed artifact valid.

To check the orchestration works (parameter plumbing, step ordering, module
dependencies) without waiting hours, run the smoke test instead — it uses
tiny horizons/seed counts in an isolated scratch directory and touches
nothing under `data/` or `figures/`:

```bash
python reproduce_paper.py --smoke
```

## What it produces

| Paper artifact | Script(s) | Output |
|---|---|---|
| Table 1 (Bernoulli/Exponential/Poisson/Gaussian) | `gen_regret_data.py`, `gen_poisson_data.py`, `gen_table_data.py` | `data/table1_summary.csv` |
| Table 2 (Binomial, sign reversal) | `gen_binomial_data.py`, `gen_table_data.py` | `data/table2_summary.csv` |
| Table 3 (runtime) | `gen_regret_data.py` | `data/table3_summary.csv` |
| Figure 1 (regret curves) | `gen_regret_data.py` / `plot_regret_curves.py` | `figures/regret_curves.pdf` |
| Figure 2 (Poisson) | `gen_poisson_data.py` / `plot_poisson_curves.py` | `figures/poisson_curves_n1000_merged.pdf` |
| Figure 3 (Binomial) | `gen_binomial_data.py` / `plot_binomial_curves.py` | `figures/binomial_curves_n100.pdf` |
| Horizon-scaling figure | `gen_horizon_data.py` / `plot_horizon_curves.py` | `figures/horizon_scaling.pdf` |
| Arms-scaling (runtime) figure | `gen_arms_scaling_data.py` / `plot_arms_scaling_curves.py` | `figures/scaling_runtime.pdf` |

## Layout

```
code/
├── reproduce_paper.py       # single entry point (see above)
├── fisher_ucb.py             # core: arms, closed-form I_F, KL-UCB inversion,
│                              #   UCB1 / Fisher-UCB / KL-UCB algorithms
├── fisher_poisson.py          # Poisson-family extension (imports fisher_ucb)
├── fisher_binomial.py         # Binomial(n fixed)-family extension
├── gen_*_data.py               # bandit simulations -> data/*.csv (one per figure/table)
├── plot_*.py                   # data/*.csv -> figures/*.pdf (plotting only)
├── verification/               # independent numerical checks of the theory:
│   ├── Kverify2.py              #   confirms the gap cone, the c2 second-order
│   ├── Kgap_c2.py                #   coefficient, the Binomial sign flip, and
│   ├── binom_verify.py            #   the beta_0 threshold mechanism, each
│   ├── poisson_verify.py           #   independently of the bandit code above
│   └── beta0_mechanism.py
```

`fisher_ucb.py`'s `run_fisher_ucb` takes `proj_interval = (mu_min, mu_max)`
— the projection interval written $\Pi_K$ / $K$ in the paper — and evaluates
the closed-form Fisher information at the *projected* empirical mean while
the index centre uses the *raw* empirical mean, exactly as in the paper's
Algorithm 1.

## Verifying the theory independently of the bandit code

The five scripts in `verification/` are self-contained (no bandit
simulation, no dependency on `fisher_ucb.py`) and check the paper's closed-
form claims by direct computation / numerical root-finding:

```bash
cd verification
python Kverify2.py          # gap cone: r_n stays inside 1 ± (C_loc/6) s + O(s^2)
python Kgap_c2.py           # second-order coefficient c2 = (5 rho^2 - 3 rho4)/72
python binom_verify.py      # I_F=1/Var, rho=-skew sign flip at mu=n/2, gap sign
python poisson_verify.py    # same identities for the Poisson family
python beta0_mechanism.py   # why beta_0 in [1, 3/2] (boundary inflation, not the optimal arm)
```
