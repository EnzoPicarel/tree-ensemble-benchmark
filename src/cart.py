"""NumPy CART extracted from the academic implementation; see EXPERIMENTS.md."""

import numpy as np


class NumpyCART:
    """Greedy CART for finite numeric features and one-dimensional class labels."""

    def __init__(
        self, criterion: str = "gini", max_depth: int | None = None, min_samples_leaf: int = 1
    ):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be gini or entropy")
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1
        ):
            raise ValueError("max_depth must be a positive integer or None")
        if (
            isinstance(min_samples_leaf, bool)
            or not isinstance(min_samples_leaf, int)
            or min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be a positive integer")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def _impurity(self, counts):
        probabilities = counts / counts.sum(axis=-1, keepdims=True)
        if self.criterion == "gini":
            return 1 - np.sum(probabilities**2, axis=-1)
        logs = np.zeros_like(probabilities, dtype=float)
        np.log2(probabilities, out=logs, where=probabilities > 0)
        return -np.sum(probabilities * logs, axis=-1)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if (
            X.ndim != 2
            or min(X.shape) == 0
            or not np.isfinite(X).all()
            or y.ndim != 1
            or len(y) != len(X)
        ):
            raise ValueError("Provide finite, nonempty 2D X and matching 1D y")
        if y.dtype.kind in "fc" and not np.isfinite(y).all():
            raise ValueError("Class labels must be finite")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.n_nodes_, self.n_leaves_, self.depth_ = 0, 0, 0
        self.root_ = self._grow(X, encoded, 0)
        return self

    def _grow(self, X, y, depth):
        n = len(y)
        counts = np.bincount(y, minlength=len(self.classes_)).astype(float)
        impurity = float(self._impurity(counts))
        node = {"probabilities": counts / n, "n_samples": n, "impurity": impurity, "feature": None}
        self.n_nodes_ += 1
        self.depth_ = max(self.depth_, depth)
        stop = (
            np.count_nonzero(counts) == 1
            or n < 2 * self.min_samples_leaf
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        best_gain, best_feature, best_threshold = -np.inf, None, None
        if not stop:
            positions = np.arange(1, n)
            for feature in range(X.shape[1]):
                order = np.argsort(X[:, feature], kind="stable")
                values = X[order, feature]
                valid = (
                    (values[:-1] < values[1:])
                    & (positions >= self.min_samples_leaf)
                    & (n - positions >= self.min_samples_leaf)
                )
                candidates = np.flatnonzero(valid)
                if not len(candidates):
                    continue
                left_counts = np.cumsum(np.eye(len(self.classes_))[y[order]], axis=0)[:-1]
                right_counts = counts - left_counts
                weighted = (
                    positions * self._impurity(left_counts)
                    + (n - positions) * self._impurity(right_counts)
                ) / n
                gains = impurity - weighted
                index = candidates[np.argmax(gains[candidates])]
                if gains[index] > best_gain + 1e-12:
                    best_gain, best_feature = float(gains[index]), feature
                    best_threshold = values[index] / 2 + values[index + 1] / 2
                    if best_threshold >= values[index + 1]:
                        best_threshold = values[index]
        if best_feature is None or best_gain < -1e-12:
            self.n_leaves_ += 1
            return node
        left = X[:, best_feature] <= best_threshold
        node.update(
            feature=best_feature,
            threshold=float(best_threshold),
            gain=best_gain,
            left=self._grow(X[left], y[left], depth + 1),
            right=self._grow(X[~left], y[~left], depth + 1),
        )
        return node

    def predict_proba(self, X):
        if not hasattr(self, "root_"):
            raise ValueError("Fit the tree before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X must have finite values and the fitted feature count")
        probabilities = np.empty((len(X), len(self.classes_)))
        for i, row in enumerate(X):
            node = self.root_
            while node["feature"] is not None:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            probabilities[i] = node["probabilities"]
        return probabilities

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
