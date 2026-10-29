# Experiment data

`reference.zip` contains the data used by the reproduction pipeline and is
extracted automatically to `reference/` on first use.

- `statistics/`: results and per-seed checkpoints for the 16 statistical
  groups used by the article.
- `poisson_certified/` and `poisson_certified_checkpoints/`: the certified
  Poisson experiment with 1,000 seeds.
- `runtime/`: seven-method timing results measured on an Intel Core i7-13650HX,
  with absolute index error 1e-6.

The archive includes experiment configurations, software metadata and expected
table values. Some experiments share observations; record counts therefore
include overlapping samples. See the root README for reproduction commands.
