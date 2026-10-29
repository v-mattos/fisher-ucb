# Fisher-UCB

Implementation and experiments for *Fisher-UCB: A Closed-Form Surrogate for
KL-UCB on Exponential-Family Bandits*.

## Installation

Python 3.12 or later is required.

```bash
python -m pip install -r code/requirements.txt
```

The tested environment uses Python 3.13.11 on Linux. To install the tested
versions of the four direct dependencies:

```bash
python -m pip install -r code/requirements-tested.txt
```

## Reproduction

Generate the four tables and six figures from the published data:

```bash
python code/reproduce_paper.py --from-data
```

Check the numerical implementation, replay selected seeds and regenerate the
outputs:

```bash
python code/reproduce_paper.py --check
```

Run all stages with reduced horizons and seed counts:

```bash
python code/reproduce_paper.py --smoke
```

Run the complete experiments, including horizons up to 10^7, the 1,000-seed
certified Poisson experiment and a new timing session:

```bash
python code/reproduce_paper.py
```

Analysis and replay results are written to `build/paper/`, reduced runs to
`build/smoke/`, and complete experiments to `build/full/`. Use `--output PATH`
to choose another directory. Statistical checkpoints resume when their
configuration matches. Complete timing sessions are reused; new measurements
use a separate output directory.

## Outputs

Paths below are relative to the selected output directory.

| Artifact | Files | Experiment |
|---|---|---|
| Table 1 | `tables/table1.csv`, `.tex` | Bernoulli, Exponential and Poisson |
| Table 2 | `tables/table2.csv`, `.tex` | Reflected Binomial |
| Table 3 | `tables/table3.csv`, `.tex` | Boundary instances |
| Table 4 | `tables/table4.csv`, `.tex` | Runtime benchmark |
| Figure 1 | `figures/regret_curves.pdf` | Bernoulli/Exponential, 20 seeds |
| Figure 2 | `figures/poisson_curves.pdf` | Poisson and projection comparison, 20 seeds each |
| Figure 3 | `figures/binomial_curves_n100.pdf` | Reflected Binomial, 100 seeds |
| Figure 4 | `figures/horizon_scaling.pdf` | Paired regret differences and radius ratios |
| Figure 5 | `figures/scaling_runtime_combined.pdf` | Runtime versus number of arms |
| Figure 6 | `figures/scaling_regret_k100.pdf` | Many-arm Bernoulli/Exponential, 20 seeds |

CSV tables contain full numerical precision. The values used in the figures
are also saved in `figures/plotted_values.json`. Curve bands show population
standard deviations, clipped at zero. Table confidence half-widths and the
paired error bars in Figure 4 use 1.96 times the sample standard error.

## Data

`data/reference.zip` contains summaries, per-seed results, checkpoints and
experiment metadata. The pipeline extracts it to `data/reference/` on first
use. Select another dataset with `--data-root PATH`.

After updating the archive, replace the extracted copy with:

```bash
python code/reproduce_paper.py --from-data --refresh-data
```

`--refresh-data` replaces only `data/reference/`; simulation outputs and their
checkpoints are stored separately under `build/`.

The dataset contains 9,200 statistical records, 1,000 certified Poisson
observations and 84 timing trials. Some experiments share observations.
Validation checks seed and configuration coverage, recomputed summaries,
checkpoint consistency and the numerical values reported in the tables.

`--check` replays seed zero for the main and boundary configurations at their
saved horizons. Use `--replay-seeds N` to increase this count.

## Runtime benchmark

The published measurements use an Intel Core i7-13650HX, one pinned logical
CPU and one numerical-library thread. The benchmark uses Bernoulli rewards,
T=50,000, K in {4, 20, 50, 100}, seven methods and three repetitions of reward
seed zero. All KL solvers target absolute index error 1e-6. Repetitions
measure timing variability. New measurements depend on the CPU and system
load. Running the timing benchmark requires Linux CPU affinity support.

## Tests

```bash
python -B -m unittest discover -s tests -v
python code/reproduce_paper.py --verify-only
```

The test suite covers numerical identities, fixed trajectory cases, checkpoint
resume and incomplete datasets. Numerical and symbolic checks run with
`--verify-only`, with `--check`, and in CI. Simulation runs do not repeat them.
Verification results are written to `build/verification/` by default.
Results record experiment settings and software metadata.

## Layout

```text
code/                  implementation and reproduction commands
  verification/        numerical checks and trajectory cases
  runtime/             timing benchmark and solvers
data/                  publication data
tests/                 pipeline tests
build/                 generated outputs
```

MIT license.
