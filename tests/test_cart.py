import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from tree_ensemble_benchmark.cart import NumpyCART

FIXTURES = [
    (np.column_stack([np.arange(6), np.zeros(6)]), np.array(["a"] * 3 + ["b"] * 3), 1, 1),
    (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0]), 2, 1),
    (np.zeros((6, 2)), np.array([0, 1, 1, 0, 1, 1]), None, 1),
    (np.array([[0, 0], [0, 0], [1, 1], [1, 1]]), np.array([0, 1, 1, 1]), None, 2),
    (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1]), None, 3),
]


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize(
    "X,y,depth,leaf",
    FIXTURES,
    ids=[
        "unique-string-split",
        "xor-zero-gain",
        "constant",
        "conflicting-duplicates",
        "leaf-limit",
    ],
)
def test_exact_source_fixtures(criterion, X, y, depth, leaf):
    custom = NumpyCART(criterion, depth, leaf).fit(X, y)
    reference = DecisionTreeClassifier(
        criterion=criterion, max_depth=depth, min_samples_leaf=leaf, random_state=42
    ).fit(X, y)
    np.testing.assert_array_equal(custom.predict(X), reference.predict(X))
    np.testing.assert_allclose(custom.predict_proba(X), reference.predict_proba(X))
    assert custom.depth_ == reference.get_depth()
    assert custom.n_leaves_ == reference.get_n_leaves()
    np.testing.assert_allclose(custom.predict_proba(X).sum(axis=1), 1)
    if len(y) == 4 and np.array_equal(y, [0, 1, 1, 0]):
        assert custom.root_["gain"] == pytest.approx(0)
    if y.dtype.kind in "US":
        assert custom.root_["feature"] == 0
        assert custom.root_["threshold"] == 2.5


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_pure_class_refit_resets_state(criterion):
    tree = NumpyCART(criterion, max_depth=3).fit(*FIXTURES[1][:2])
    tree.fit(np.zeros((3, 2)), ["only"] * 3)
    assert tree.n_nodes_ == tree.n_leaves_ == 1
    assert tree.depth_ == 0
    np.testing.assert_array_equal(tree.predict([[0, 0]]), ["only"])


def test_constraints_on_nontrivial_data():
    rng = np.random.default_rng(7)
    X, y = rng.normal(size=(100, 3)), rng.integers(0, 3, 100)
    tree = NumpyCART(max_depth=3, min_samples_leaf=7).fit(X, y)

    def sizes(node):
        return (
            [node["n_samples"]]
            if node["feature"] is None
            else sizes(node["left"]) + sizes(node["right"])
        )

    assert min(sizes(tree.root_)) >= 7
    assert tree.depth_ <= 3
    assert tree.n_nodes_ == 2 * tree.n_leaves_ - 1
    p = tree.predict_proba(X)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)


@pytest.mark.parametrize(
    "X,y",
    [
        ([], []),
        ([[1], [2]], [0]),
        ([[np.nan]], [1]),
        ([[np.inf]], [0]),
        ([[1]], [np.nan]),
        ([[1]], [[0]]),
        (np.empty((3, 0)), [1, 2, 3]),
    ],
)
def test_degenerate_training_input(X, y):
    with pytest.raises(ValueError):
        NumpyCART().fit(X, y)


def test_prediction_input_errors():
    with pytest.raises(ValueError, match="Fit"):
        NumpyCART().predict([[0]])
    model = NumpyCART().fit([[0], [1]], [0, 1])
    for X in [[[1, 2]], [[np.nan]], [1]]:
        with pytest.raises(ValueError):
            model.predict(X)


@pytest.mark.parametrize(
    "params",
    [
        dict(criterion="bad"),
        dict(max_depth=0),
        dict(max_depth=True),
        dict(min_samples_leaf=False),
        dict(min_samples_leaf=0),
    ],
)
def test_invalid_constraints(params):
    with pytest.raises(ValueError):
        NumpyCART(**params)
