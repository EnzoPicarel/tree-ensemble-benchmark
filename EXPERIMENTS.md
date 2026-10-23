# Experiments

## Data and splits

- **Fashion-MNIST:** 10,000 training and 5,000 validation images sampled from the
  official training set; the official 10,000-image test set is held out. Pixels
  stay on their original 0–255 scale.
- **California Housing:** 12,384 training, 4,128 validation and 4,128 test rows
  (60/20/20). Targets are in units of $100,000.
- Split and fold seed: 42. Estimator seeds: 42, 123, 2026, 7, 99.

Sources: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) by
Han Xiao, Kashif Rasul and Roland Vollgraf / Zalando; [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
by Pace and Barry, derived from the 1990 US census. Dataset files are downloaded
only with `--download` and are not included here. Their original terms apply.

## Models and stacking

The library benchmark compares CART, pruning, pasting, bagging, random subspaces,
random patches, random forests, Extra Trees, AdaBoost, gradient boosting, voting
and stacking. Frozen settings are in [src/configs/frozen.json](src/configs/frozen.json).
These estimators come from scikit-learn; the separate NumPy CART implementation
is evaluated on small synthetic datasets.

Five training-only folds generate OOF predictions. Every training row receives
one held-out prediction from each member. Classification uses four ten-class
probability vectors and a logistic meta-model; regression uses four predictions
and a Ridge meta-model. Members are refitted on the original training rows.
Validation and test rows never enter meta-model training.

## Run the experiments

From the repository root, after installing requirements:

```sh
# Small generated datasets; no downloads.
python main.py quick --output runs/quick
# The 90-configuration synthetic CART comparison.
python main.py cart --output runs/cart
# Check recorded results and regenerate the trade-off figure.
python main.py report --output runs/report
# Expensive: both real datasets and all five estimator seeds.
python main.py full --download --data-dir data --output runs/full
# A partial real-data replay.
python main.py full --task regression --seeds 42 --download --output runs/housing-seed42
```

Choose a fresh output directory each time. Without `--download`, full mode expects
an existing dataset cache. Outputs include metrics, fold records, split indices,
source/config hashes and environment information. Fits run sequentially with one
numerical thread. The full campaign has 315 fits and can take substantial CPU time;
original hardware and peak memory were not recorded. Quick mode takes a few seconds
on the machine used for verification. No GPU is required.

## Results and limitations

[results/historical](results/historical) contains original recorded outputs, preserved
with file hashes. [results/verification](results/verification) contains separately
rerun synthetic experiments and the real-data Housing seed-42 replay. The latter
matches all 13 historical methods' RMSE, MAE and node counts. The complete five-seed
campaign has not been independently rerun. See [VALIDATION.md](VALIDATION.md).

Five-seed SD measures estimator randomness on fixed partitions, not uncertainty over
new datasets. Extensive validation reuse may bias model selection. The historical
pruning path was derived before CV, so its CV score is not a nested estimate.
Random Housing splits do not measure unseen-region or future-year generalization.

Stacking's recorded Housing RMSE improvement is 0.2944%, with 13.33x fitting and
15.59x prediction cost relative to gradient boosting. These timings are specific
to the original machine/protocol. Stacking fit cost includes OOF training; model
size counts retained tree nodes, not memory bytes. Combined prediction cost sums
member timings and aggregation; it is not a serving benchmark. New replay timings
are kept separate from historical costs.

The NumPy CART supports finite numeric inputs, Gini/entropy and depth/leaf limits.
It does not implement pruning, sample weights or missing-value routing. Tied split
gains can produce different predictions from scikit-learn.

## Origin

Developed from Enzo Picarel's University of Cincinnati coursework, then reorganized
with standalone tests and a command-line runner. The original study's commit and
notebook hash are recorded in [the artifact manifest](results/historical/artifact_manifest.json).
AI assistance was used for the reorganization and tests. No assignment handouts or
academic Git history are included.
