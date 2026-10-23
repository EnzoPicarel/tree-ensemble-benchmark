# Tree Ensemble Benchmark

How much predictive improvement justifies a more expensive model? This project
compares tree ensembles across classification and regression, implements CART in
NumPy, and makes out-of-fold stacking and evaluation costs inspectable.

The historical housing study found only a **0.29% reduction in RMSE** from stacking
compared with gradient boosting, alongside roughly **13× fitting cost and 16× inference
cost**. These are recorded machine-specific timings, not universal complexity ratios.

This is a reorganized portfolio edition of university coursework. Historical results
are preserved separately from new verification runs. It is an experimental benchmark;
no deployment or exclusive authorship of all source components is claimed.
See [provenance](docs/provenance.md) and the [publication gate](docs/publication-audit.md).

## Engineering contributions

- **NumPy CART:** sorted-feature split search, Gini/entropy, depth/leaf constraints,
  probability prediction and explicit edge-case tests. Library ensembles use scikit-learn.
- **OOF stacking:** five training-only folds, exactly-once prediction assignments,
  class-probability validation, fixed configurations and training-only final refits.
- **Evaluation:** classification/regression, repeated estimator seeds, fitting time,
  prediction latency and structural model size. Quick runs require no downloads.

## Selected historical results

| Task | Model | Recorded mean test result |
|---|---|---:|
| Fashion-MNIST | OOF stacking | 86.62% accuracy ± 0.0925 percentage points |
| California Housing | Gradient boosting | 0.47249 RMSE |
| California Housing | OOF stacking | 0.47110 RMSE |

Five estimator seeds share fixed partitions; ± is sample SD, not a confidence
interval. Housing targets use $100,000 units. Full-data models are scikit-learn
estimators. The separate custom-CART study records 90 synthetic configurations,
97.5–100% held-out prediction agreement with sklearn, and 12 exact fixtures.
Prediction agreement does not mean identical implementations.

The CSV aggregates were independently recalculated during extraction; original
five-seed full-data campaign was **not rerun**. A new Housing seed-42 replay matches
all 13 methods' historical error and node-count values; the separate 90-configuration
synthetic CART comparison also matches. See [verification](docs/verification.md). Historical artifacts have a
[SHA-256 manifest](results/historical/artifact_manifest.json).

![Historical housing accuracy-cost trade-off](figures/housing_tradeoff.svg)

Stacking is not an unconditional recommendation: gradient boosting is a compelling
simpler choice for housing given the small error difference. Timing sums include
OOF training for stacking. Node counts are structural size proxies, not memory bytes.
See [measurement definitions and limitations](docs/methodology.md) and the
[interview preparation notes](docs/interview-guide.md).

## Installation and quick verification

Python 3.12 is recommended; Python 3.11+ is supported.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python -m tree_ensemble_benchmark.cli quick --output runs/quick
```

Quick mode generates small synthetic classification/regression data, runs limited
sampling/forest/boosting/voting/stacking comparisons and exercises CART. It writes
metrics, OOF records, split indices and environment metadata. It is not a substitute
for the full-data benchmark. Choose a fresh output directory for each run.

```sh
# Re-run the 90-configuration synthetic CART comparison.
python -m tree_ensemble_benchmark.cli cart --output runs/cart
# Verify historical aggregates and regenerate the figure.
python -m tree_ensemble_benchmark.cli report --output runs/report
# Explicit expensive replay: public dataset downloads, both tasks, five seeds.
python -m tree_ensemble_benchmark.cli full --download --output runs/full
```

The full command replays the frozen final comparison, not every exploratory notebook
sweep. It preserves 10,000/5,000/10,000 Fashion-MNIST and 60/20/20 Housing partitions;
validation never joins the training or OOF rows. Detailed environment, data, output
and compute instructions are in [reproduction](docs/reproduction.md).

## Implementation map

```text
src/tree_ensemble_benchmark/
  cart.py            NumPy CART implementation
  data.py            Public dataset loaders and disjoint splits
  stacking.py        OOF features and training-only stack fitting
  models.py          Allowlisted frozen and quick model construction
  configs/frozen.json Explicit historical model configurations
  evaluation.py      Metrics, timings and artifact output
  experiments.py     Quick/full campaign orchestration
  cart_study.py      Separate synthetic CART comparison
  reporting.py       Historical consistency checks and SVG figure
  cli.py             Explicit command-line entrypoints
tests/               Inexpensive correctness and isolation tests
results/historical/  Preserved aggregate results and source hashes
figures/             Regenerated presentation artifacts
docs/                Methodology, reproduction, provenance and publication review
```

## Limits and attribution

Extensive validation reuse can bias selection. The historical pruning grid was derived
before CV and is not a nested estimate. Fixed-seed SD does not measure new-dataset
uncertainty. Random housing partitions do not test unseen regions or future years.
No corrected pruning score or production capability is inferred from the old results.

Foundation: Enzo Picarel's University of Cincinnati ML coursework. Dataset authors:
Xiao, Rasul and Vollgraf / Zalando (Fashion-MNIST); Pace and Barry / 1990 census
(California Housing). See [sources and licensing review](docs/datasets.md).
Publication and project licensing remain blocked pending course/starter-code ownership
and dataset-use confirmation. No course materials or datasets are distributed.
