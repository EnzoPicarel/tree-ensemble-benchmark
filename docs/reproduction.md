# Reproduction

## Install and verify

Use Python 3.12 (supported: 3.11+). From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python -m tree_ensemble_benchmark.cli quick --output runs/quick
python -m tree_ensemble_benchmark.cli cart --output runs/cart
python -m tree_ensemble_benchmark.cli report --output runs/report
```

Choose a new output directory for each experiment. Quick mode uses generated data,
never downloads data, uses one seed and five OOF folds, and writes synthetic metrics,
fold records, split indices, source/config hashes and environment metadata. It is an
execution/correctness demonstration, not the Fashion-MNIST or Housing benchmark.
The CART command repeats the original 90 small synthetic configurations.

## Frozen full benchmark

```sh
python -m tree_ensemble_benchmark.cli full --download --data-dir data --output runs/full
```

Downloads occur only with explicit `--download`; see [data sources](datasets.md).
Downloaded datasets remain ignored local files. To use an existing compatible cache,
set `--data-dir` and omit `--download`. No academic course dataset is needed.

The default full command uses both tasks and all five original estimator seeds.
For a clearly labeled partial replay:

```sh
python -m tree_ensemble_benchmark.cli full --task regression --seeds 42 --download --output runs/housing-seed42
```

A partial campaign is not a five-seed reproduction. Full uses frozen historical
hyperparameters, not a new exploratory search. There are 315 fits (305 base/fold fits and
five meta-models per task), 130 final model/seed score rows, and 200 OOF fold records.
No validation rows join training. Split seed/fold seed are 42. See [methodology](methodology.md).

Original software versions are preserved in [environment-original.txt](environment-original.txt).
The tested portfolio environment is recorded in [verification](verification.md) and
[environment-verified.txt](environment-verified.txt). The latter is an exact installation
snapshot; use `pip install -r docs/environment-verified.txt` before `pip install -e .`
for that environment. Version changes can affect tie-breaking, estimators and results.
The packaged frozen config is installed with wheels, not fetched from an academic repo.

Outputs include metrics.csv, oof_folds.csv, per-task split_indices.npz and metadata.json.
Result tables are checkpointed after each completed task/seed. Interrupted runs retain
status=incomplete; automatic resume is not implemented. Model caches are intentionally
not trusted or reused. No full model checkpoints are distributed.

## Compute expectations

Quick and CART timings measured during preparation are in verification.md. No GPU
is needed. Full Fashion-MNIST fitting is expensive, especially repeated AdaBoost
and OOF fits. Historical classification stacking costs sum to about 7,950 seconds
per seed, but that is cumulative fit cost under concurrent contention, not a reliable
prediction of your wall time. Original hardware and peak RSS were not recorded;
no precise RAM or wall-time guarantee is defensible. Run one task/seed first and
monitor memory. Replay fits sequentially, constrains numerical threads to one and
keeps prediction blocks rather than all fitted forests in memory. Inputs alone use
roughly 220 MB for float32 Fashion-MNIST, excluding estimator/OOF memory.

The complete five-seed, both-task campaign was not launched during portfolio
verification. The real-data regression task was rerun for seed 42, reproducing all
13 historical accuracy/error and node-count results; see verification.md. Tests
also exercise orchestration with generated data and frozen estimator construction.
The remaining full-data seeds/tasks and their resource needs remain external
reproduction checks. Wheel-only quick execution was tested separately.
