# Experimental methodology

## Historical study

Fashion-MNIST uses the official test set (10,000 rows). A stratified sample of
15,000 official training rows becomes 10,000 training and 5,000 validation rows.
California Housing uses 12,384/4,128/4,128 rows, a random 60/20/20 split.
Both constructions use split seed 42. Tree models receive unscaled features;
Fashion-MNIST pixels remain 0–255. Targets for housing are in units of $100,000.

The original studies covered criteria/depth/leaf size, pruning, row/feature sampling,
OOB coverage, bootstrap instability, forests, permutation importance, boosting learning
curves and label noise, error correlation, voting and OOF stacking. Selected aggregate
CSVs are retained in results/historical. The new full command reproduces the frozen
final comparison, not every exploratory sweep or original figure. It never silently
reselects hyperparameters from test results. Original exploration is documented by
its result tables, not copied assignment instructions.

Historical classifier choices: depth-10 Gini CART; entropy CART pruned at alpha
0.002060353; pasting 100 trees/70% rows; bagging 239 trees/70% rows; random subspaces
100 trees/30% features; random patches 100 trees/10% features; 300-tree entropy
forests with 78 features; AdaBoost 490 depth-8 trees at learning rate 0.5; histogram
boosting 126 rounds/depth 6/learning rate 0.05. Full regression configurations and
all explicit parameters are in the packaged configs/frozen.json. Unspecified
parameters follow sklearn defaults, so library versions matter.

## OOF and final fits

Five training-only folds use fold seed 42. Each training row receives exactly one
held-out prediction from every member. Classification concatenates four ten-class
probability vectors; regression stacks four scalar predictions. Logistic regression
(C=1, max_iter=2000, tol=1e-6) or Ridge(alpha=1) learns the meta-model. Members are
refitted on the original training subset; validation is not merged into training.
Round counts are frozen and early stopping is disabled for the OOF histogram model.
Estimator seeds are 42, 123, 2026, 7, 99; fold assignments remain fixed.

The extracted OOF API accepts only training X/y and fold indices. It rejects
missing, overlapping, duplicate and incomplete assignments. Instrumented tests
record exactly which row IDs each fitted estimator sees. It does not claim to detect
caller-supplied semantic leakage such as target-derived input features.

Soft voting combines uncalibrated probability scales; the benchmark does not claim
that equal-weight probability averaging is intrinsically well calibrated.

## Limits and interpretation

- Five seeds measure estimator randomness on fixed partitions, not uncertainty
  across independent datasets. SD is sample SD (ddof=1), not a confidence interval.
- Broad repeated validation use may overfit model selection; OOF training does not
  undo hyperparameter-selection bias. This is not nested model-selection validation.
- The historical pruning alpha candidate path was derived from the whole training
  subset before CV. Its CV scores are conditional diagnostics, not unbiased nested
  performance estimates. Frozen replay retains the selected alpha; it does not
  relabel the old CV score as corrected. No new pruning-selection metric is claimed.
- Random housing splits mix regions. New-region and temporal generalization are
  untested. Housing is a historical benchmark, not a valuation deployment.
- Custom CART's 90 synthetic comparisons are separate from library full-data models.
  Tied optimal gains can yield different predictions; agreement is not identity.
- CART omits missing-value routing, sample weights and pruning; it supports finite
  numeric features and scalar class labels, not a drop-in sklearn replacement.
- The five-seed full-data means are historical. The new regression seed-42 replay
  matches that seed's error/node metrics; it does not reproduce a five-seed mean.

## Cost measurement

Historical costs are recorded elapsed fitting seconds, prediction milliseconds per
1,000 rows, tree node counts, and meta-model parameter counts. Nodes are a structural
proxy, not serialized bytes or resident memory. Stacking fitting includes OOF fits,
full member fits, and meta-fitting. Inference size excludes discarded OOF trees.

The historical fits used concurrent workers (up to 16) with one numerical thread
per fit; timings reflect contention. The new replay is sequential with one thread,
includes feature-validation overhead in prediction, and performs no cache reuse.
Its times are new measurements even if accuracy matches. They cannot be expected
to reproduce old machine timings. Combined latency adds member latency and aggregation;
it is not a separate repeatedly warmed end-to-end serving benchmark. One-shot latency
is noisy. Internal histogram-tree node counting uses a private sklearn attribute;
version pinning and focused tests guard this maintenance risk.
