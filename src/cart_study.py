"""Independent 90-configuration synthetic comparison, separate from full-data models."""

from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from .cart import NumpyCART
from .evaluation import write_csv, write_json, environment
from .experiments import prepare_output


def run_cart_study(output: Path):
    output = prepare_output(output)
    rows = []
    with threadpool_limits(limits=1):
        for seed in [42, 123, 2026, 7, 99]:
            X, y = make_classification(
                n_samples=360,
                n_features=2,
                n_informative=2,
                n_redundant=0,
                n_classes=3,
                n_clusters_per_class=1,
                class_sep=1.2,
                flip_y=0.03,
                random_state=seed,
            )
            train, test, labels, actual = train_test_split(
                X.astype(np.float32), y, train_size=240, stratify=y, random_state=seed
            )
            for criterion in ["gini", "entropy"]:
                for depth in [1, 3, None]:
                    for leaf in [1, 5, 20]:
                        custom = NumpyCART(criterion, depth, leaf).fit(train, labels)
                        reference = DecisionTreeClassifier(
                            criterion=criterion,
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            random_state=seed,
                        ).fit(train, labels)
                        pred, ref = custom.predict(test), reference.predict(test)
                        rows.append(
                            dict(
                                seed=seed,
                                criterion=criterion,
                                max_depth=depth,
                                min_samples_leaf=leaf,
                                numpy_holdout_accuracy=float(np.mean(pred == actual)),
                                sklearn_holdout_accuracy=float(np.mean(ref == actual)),
                                holdout_agreement=float(np.mean(pred == ref)),
                            )
                        )
    write_csv(output / "cart_comparison.csv", rows)
    write_json(
        output / "metadata.json",
        dict(
            environment=environment(),
            synthetic=True,
            configurations=len(rows),
            status="complete",
            independent_rerun=True,
        ),
    )
    return rows
