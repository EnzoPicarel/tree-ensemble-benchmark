"""OOF feature generation and final training-only refits; no evaluation labels accepted."""

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge


def prediction_features(model, X, task, classes=None):
    if task == "classification":
        if not np.array_equal(model.classes_, classes):
            raise ValueError("Fold class order differs from training classes")
        output = np.asarray(model.predict_proba(X), dtype=float)
        if (
            output.shape != (len(X), len(classes))
            or not np.isfinite(output).all()
            or (output < 0).any()
            or (output > 1).any()
            or not np.allclose(output.sum(axis=1), 1)
        ):
            raise ValueError("Invalid probability features")
        return output
    if task != "regression":
        raise ValueError("Unknown task")
    output = np.asarray(model.predict(X), dtype=float)
    if output.shape != (len(X),) or not np.isfinite(output).all():
        raise ValueError("Invalid regression predictions")
    return output[:, None]


def validate_folds(folds, n):
    folds = list(folds)
    coverage = np.zeros(n, dtype=int)
    for train, heldout in folds:
        train, heldout = np.asarray(train), np.asarray(heldout)
        if any(a.ndim != 1 or a.dtype.kind not in "iu" or len(a) == 0 for a in (train, heldout)):
            raise ValueError("Fold indices must be nonempty integer vectors")
        combined = np.concatenate([train, heldout])
        if (
            len(combined) != n
            or len(np.unique(combined)) != n
            or combined.min() != 0
            or combined.max() != n - 1
        ):
            raise ValueError("Each fold must be a disjoint complete train/holdout partition")
        coverage[heldout] += 1
    if len(folds) < 2 or not np.all(coverage == 1):
        raise ValueError("Every row must receive exactly one held-out prediction")
    return folds


def oof_features(models: dict, X, y, *, task: str, folds):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not models:
        raise ValueError("Expected aligned training features, labels, and nonempty models")
    folds = validate_folds(folds, len(y))
    classes = np.unique(y) if task == "classification" else None
    blocks, records = [], []
    for name, estimator in models.items():
        width = len(classes) if classes is not None else 1
        block = np.full((len(y), width), np.nan)
        for fold, (training, heldout) in enumerate(folds):
            model = clone(estimator)
            # External validation labels are never passed; early stopping must be frozen.
            if model.get_params().get("early_stopping", False) not in (False, None):
                raise ValueError("Freeze early stopping before OOF generation")
            start = perf_counter()
            model.fit(X[training], y[training])
            elapsed = perf_counter() - start
            block[heldout] = prediction_features(model, X[heldout], task, classes)
            records.append(
                dict(
                    method=name,
                    fold=fold,
                    training_rows=len(training),
                    heldout_rows=len(heldout),
                    fit_seconds=elapsed,
                )
            )
        if not np.isfinite(block).all():
            raise ValueError("Missing or nonfinite OOF features")
        blocks.append(block)
    return np.concatenate(blocks, axis=1), records


@dataclass
class FittedStack:
    models: dict
    meta_model: object
    task: str
    classes: object
    oof: np.ndarray
    fold_records: list
    meta_fit_seconds: float

    def predict(self, X):
        features = np.concatenate(
            [prediction_features(m, X, self.task, self.classes) for m in self.models.values()],
            axis=1,
        )
        return self.meta_model.predict(features)


def fit_stack(models: dict, X, y, *, task: str, seed=42, fold_seed=42, n_splits=5):
    """Fit meta-model on OOF rows, then refit members on the same training partition.

    Validation/test features and labels are deliberately absent from this API.
    """
    splitter = (
        StratifiedKFold(n_splits, shuffle=True, random_state=fold_seed)
        if task == "classification"
        else KFold(n_splits, shuffle=True, random_state=fold_seed)
    )
    folds = list(splitter.split(X, y))
    features, records = oof_features(models, X, y, task=task, folds=folds)
    meta = (
        LogisticRegression(C=1, max_iter=2000, tol=1e-6, random_state=seed)
        if task == "classification"
        else Ridge(alpha=1)
    )
    start = perf_counter()
    meta.fit(features, y)
    elapsed = perf_counter() - start
    if task == "classification" and meta.n_iter_.max() >= meta.max_iter:
        raise RuntimeError("Stacking meta-model did not converge")
    fitted = {name: clone(model).fit(X, y) for name, model in models.items()}
    return FittedStack(
        fitted,
        meta,
        task,
        np.unique(y) if task == "classification" else None,
        features,
        records,
        elapsed,
    )
