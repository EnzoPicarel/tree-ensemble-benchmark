import inspect
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from tree_ensemble_benchmark.stacking import (
    oof_features,
    fit_stack,
    validate_folds,
    prediction_features,
)


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Row IDs make accidental in-fold predictions observable, not inferred."""

    fits = []
    predictions = []

    def fit(self, X, y):
        self.ids_ = set(X[:, 0].astype(int))
        type(self).fits.append(self.ids_)
        self.classes_ = np.unique(y)
        self.size_ = len(X)
        return self

    def predict_proba(self, X):
        ids = set(X[:, 0].astype(int))
        type(self).predictions.append((self.ids_, ids))
        assert self.ids_.isdisjoint(ids), "Predicted rows used during fitting"
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


def test_oof_rows_never_seen_by_fold_estimator():
    X = np.column_stack([np.arange(30), np.zeros(30)])
    y = np.tile([0, 1, 2], 10)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    RecordingClassifier.fits, RecordingClassifier.predictions = [], []
    features, records = oof_features(
        {"first": RecordingClassifier(), "second": RecordingClassifier()},
        X,
        y,
        task="classification",
        folds=folds,
    )
    assert features.shape == (30, 6)
    np.testing.assert_allclose(features, 1 / 3)
    assert len(records) == 10
    for offset in (0, 5):
        seen = [heldout for _, heldout in RecordingClassifier.predictions[offset : offset + 5]]
        assert set.union(*seen) == set(range(30))
        assert sum(map(len, seen)) == 30
    for i, (train, test) in enumerate(folds):
        assert RecordingClassifier.fits[i] == set(train)
        assert RecordingClassifier.predictions[i][1] == set(test)


def test_external_partitions_never_enter_meta_training(monkeypatch):
    from tree_ensemble_benchmark import stacking as module
    from sklearn.linear_model import LogisticRegression

    fits = []

    class RecordingMeta(LogisticRegression):
        def fit(self, X, y, **kwargs):
            fits.append((X.copy(), y.copy()))
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(module, "LogisticRegression", RecordingMeta)
    rng = np.random.default_rng(42)
    X, y = rng.normal(size=(40, 3)), np.tile([0, 1], 20)
    stack = fit_stack(
        {"tree": DecisionTreeClassifier(max_depth=2, random_state=42)}, X, y, task="classification"
    )
    assert len(fits) == 1
    np.testing.assert_array_equal(fits[0][0], stack.oof)
    np.testing.assert_array_equal(fits[0][1], y)
    assert fits[0][0].shape == (40, 2)
    coefficients = stack.meta_model.coef_.copy()
    stack.predict(rng.normal(size=(10, 3)))
    assert len(fits) == 1
    np.testing.assert_array_equal(coefficients, stack.meta_model.coef_)
    assert stack.models["tree"].tree_.n_node_samples[0] == len(y)
    assert not any(
        k in inspect.signature(fit_stack).parameters
        for k in ["X_validation", "y_validation", "X_test", "y_test"]
    )


@pytest.mark.parametrize(
    "folds",
    [
        [(np.array([1, 2, 3]), np.array([0])), (np.array([1, 2, 3]), np.array([0]))],
        [(np.array([0, 1]), np.array([1, 2]))],
        [(np.array([0, 1, 2]), np.array([3, 3]))],
        [(np.array([0, 1]), np.array([2, 4]))],
    ],
)
def test_reject_duplicate_missing_overlapping_and_invalid_folds(folds):
    with pytest.raises(ValueError):
        validate_folds(folds, 4)


def test_regression_oof_shape_and_training_only_refit():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20) * 0.1
    features, records = oof_features(
        {"a": DecisionTreeRegressor(max_depth=2)},
        X,
        y,
        task="regression",
        folds=list(KFold(5).split(X)),
    )
    assert features.shape == (20, 1) and np.isfinite(features).all()
    assert len(records) == 5
    stack = fit_stack({"a": DecisionTreeRegressor(max_depth=2)}, X, y, task="regression")
    assert stack.predict(X[:3]).shape == (3,)
    assert stack.models["a"].tree_.n_node_samples[0] == len(y)


def test_missing_classes_and_bad_probabilities_rejected():
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="class order"):
        prediction_features(model, [[0]], "classification", np.array([0, 1, 2]))

    class Invalid:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.8)

    with pytest.raises(ValueError, match="probability"):
        prediction_features(Invalid(), [[0]], "classification", np.array([0, 1]))
