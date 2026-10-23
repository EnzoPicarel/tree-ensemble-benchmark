"""Explicit, allowlisted construction of frozen scikit-learn configurations."""

import json
from importlib.resources import files
from sklearn import ensemble, tree

REGISTRY = {
    name: getattr(module, name)
    for module, names in [
        (tree, ["DecisionTreeClassifier", "DecisionTreeRegressor"]),
        (
            ensemble,
            [
                "BaggingClassifier",
                "BaggingRegressor",
                "RandomForestClassifier",
                "RandomForestRegressor",
                "ExtraTreesClassifier",
                "ExtraTreesRegressor",
                "AdaBoostClassifier",
                "AdaBoostRegressor",
                "HistGradientBoostingClassifier",
                "GradientBoostingRegressor",
            ],
        ),
    ]
    for name in names
}


def frozen_config():
    return json.loads(files("tree_ensemble_benchmark").joinpath("configs/frozen.json").read_text())


def build_model(spec: dict, seed: int):
    name = spec["estimator"]
    if name not in REGISTRY:
        raise ValueError(f"Unsupported estimator: {name}")
    params = {
        key: build_model(value, seed) if isinstance(value, dict) else value
        for key, value in spec["params"].items()
    }
    model = REGISTRY[name](**params)
    if "random_state" in model.get_params(deep=False):
        model.set_params(random_state=seed)
    return model


def quick_models(task: str, seed: int):
    cls = task == "classification"
    if task not in {"classification", "regression"}:
        raise ValueError("Unknown task")
    tree_type = tree.DecisionTreeClassifier if cls else tree.DecisionTreeRegressor
    bag_type = ensemble.BaggingClassifier if cls else ensemble.BaggingRegressor
    rf_type = ensemble.RandomForestClassifier if cls else ensemble.RandomForestRegressor
    extra_type = ensemble.ExtraTreesClassifier if cls else ensemble.ExtraTreesRegressor
    ada_type = ensemble.AdaBoostClassifier if cls else ensemble.AdaBoostRegressor
    boost_type = (
        ensemble.HistGradientBoostingClassifier if cls else ensemble.GradientBoostingRegressor
    )
    boost_params = {"max_iter": 12, "early_stopping": False} if cls else {"n_estimators": 12}
    return {
        "Tree": tree_type(max_depth=4, random_state=seed),
        "Pasting": bag_type(
            estimator=tree_type(max_depth=4),
            n_estimators=8,
            bootstrap=False,
            max_samples=0.7,
            random_state=seed,
            n_jobs=1,
        ),
        "Bagging": bag_type(
            estimator=tree_type(max_depth=4),
            n_estimators=8,
            max_samples=0.7,
            random_state=seed,
            n_jobs=1,
        ),
        "Random subspaces": bag_type(
            estimator=tree_type(max_depth=4),
            n_estimators=8,
            bootstrap=False,
            max_features=0.7,
            random_state=seed,
            n_jobs=1,
        ),
        "Random patches": bag_type(
            estimator=tree_type(max_depth=4),
            n_estimators=8,
            max_samples=0.7,
            max_features=0.7,
            random_state=seed,
            n_jobs=1,
        ),
        "Random forest": rf_type(n_estimators=12, max_depth=5, random_state=seed, n_jobs=1),
        "Extra trees": extra_type(n_estimators=12, max_depth=5, random_state=seed, n_jobs=1),
        "AdaBoost": ada_type(estimator=tree_type(max_depth=2), n_estimators=12, random_state=seed),
        "Gradient boosting": boost_type(max_depth=3, random_state=seed, **boost_params),
    }
