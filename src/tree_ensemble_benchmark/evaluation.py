"""Metrics, portable result writing, and explicitly defined timing quantities."""

import csv
import json
import platform
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from .stacking import prediction_features


def score(y, prediction, task):
    if task == "classification":
        return dict(
            accuracy_pct=100 * accuracy_score(y, prediction),
            macro_f1=f1_score(y, prediction, average="macro"),
        )
    return dict(
        rmse=float(np.sqrt(mean_squared_error(y, prediction))),
        mae=float(mean_absolute_error(y, prediction)),
    )


def node_count(model):
    if hasattr(model, "tree_"):
        return int(model.tree_.node_count)
    if hasattr(model, "_predictors"):
        # Private sklearn structure; guarded by pinned replay environment and tests.
        return sum(len(t.nodes) for stage in model._predictors for t in stage)
    if hasattr(model, "estimators_"):
        return sum(node_count(m) for m in np.asarray(model.estimators_, dtype=object).ravel())
    return 0


def fit_member(estimator, part, task):
    model = clone(estimator)
    start = perf_counter()
    model.fit(part.X_train, part.y_train)
    elapsed = perf_counter() - start
    return model, elapsed


def timed_features(model, X, task, classes):
    start = perf_counter()
    features = prediction_features(model, X, task, classes)
    labels = classes[features.argmax(axis=1)] if task == "classification" else features[:, 0]
    elapsed = perf_counter() - start
    return features, labels, elapsed * 1e6 / len(X)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write empty results")
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ["numpy", "scikit-learn", "threadpoolctl"]},
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
