# Verification record

Prepared 2026-10-23 using Python 3.12.3, NumPy 2.5.4 and scikit-learn 1.9.1.
Exact installed versions are in [environment-verified.txt](environment-verified.txt).

## Executed checks

- `python -m pytest -q`: **43 passed**, final suite completed in 4.40 seconds.
  Covers the 12 original CART fixture variants, degenerate input handling, stopping
  constraints, instrumented OOF row isolation, meta-training input capture, class
  alignment, partition integrity, deterministic quick metrics, frozen model
  construction/fitting on small data, historical aggregate consistency, and the
  independently recorded Housing replay.
- `ruff check src tests scripts`: passed; code formatted with Ruff.
- Editable package installation and dependency consistency check: passed.
- Source distribution and wheel build: passed. The wheel installs into a separate
  environment and runs its CLI quick benchmark from outside the checkout; the
  frozen config is available through installed package resources.
- Historical artifact checksums and means/sample SDs independently recalculated
  from 130 raw rows and compared with original summary tables.
- Synthetic quick comparison: both tasks completed, 23 score rows, five OOF folds
  for each of four members per task. Repeated quick runs have identical predictive
  metrics and node counts; wall-clock timing is intentionally not deterministic.
- Custom CART comparison: **90 configurations rerun**. All three per-configuration
  accuracy/agreement fields match the historical CSV within 1e-12. Held-out
  agreement range remains **97.5–100%**. These are synthetic comparisons, not
  Fashion-MNIST performance.

Measured standalone command wall times on this WSL2 developer environment:

| Command | Seconds |
|---|---:|
| quick | 2.98 |
| cart | 1.95 |
| report | 1.61 |

These include interpreter startup and are illustrative machine-specific measurements.
The full five-seed campaign, especially Fashion-MNIST, has not been rerun.

## Independently reproduced real-data subset

`tree-benchmark full --task regression --seeds 42 --download --data-dir data --output runs/housing-seed42`
completed using the public sklearn loader. The development fingerprint exactly
matches the original frozen campaign. **All 13 methods match historical seed-42
RMSE, MAE and node counts exactly.** Stacking RMSE is **0.4700182631** and gradient
boosting RMSE is **0.4718745140** for this seed. These are not the five-seed means.
The full-data subset performs 31 regression base/fold fits plus one meta-fit.

The new [metrics](../results/verification/housing-seed42/metrics.csv),
[OOF records](../results/verification/housing-seed42/oof_folds.csv), and
[run metadata](../results/verification/housing-seed42/metadata.json) are preserved.
New timings differ from the original concurrent campaign and are not substituted
into the historical cost ratios. Only seed 42 for regression was rerun; the other
four regression seeds and all full-data Fashion-MNIST runs remain historical.

New [synthetic CART results](../results/verification/cart/cart_comparison.csv) and
[quick-run results](../results/verification/quick/metrics.csv) are also retained,
with separate metadata. No raw dataset or per-example predictions are included.


## Source protection and publication

The academic source's 86 tracked-file SHA-256 values, HEAD, and Git status match
its pre-extraction snapshot. Sizes and modification times of all 1,620 inventoried
files (excluding its environment) also match, including academic Git metadata.
Snapshots remain in ignored local audit records, not public files.

A separate content audit passed for the prospective Git publication file set.
It checks local Markdown links, private path and credential
patterns, restricted file types, and large files. It is not an ownership clearance
or a complete secret-history guarantee. Publication is BLOCKED; see
[publication audit](publication-audit.md). No remote or project license was created.
