import json
from pathlib import Path
import numpy as np
import pytest
from src.experiments import run
from src.models import frozen_config, build_model
from src.evaluation import node_count
from src.reporting import historical_summary


def test_quick_runs_reproduce_metrics_and_protect_output(tmp_path):
    a = run("quick", tmp_path / "one")
    b = run("quick", tmp_path / "two")
    assert len(a) == 23
    for left, right in zip(a, b):
        for key in ["task", "method", "seed", "accuracy_pct", "macro_f1", "rmse", "mae", "nodes"]:
            if key in left:
                assert (
                    left[key] == pytest.approx(right[key])
                    if isinstance(left[key], float)
                    else left[key] == right[key]
                )
    meta = json.loads((tmp_path / "one/metadata.json").read_text())
    assert meta["synthetic"] is True and meta["status"] == "complete"
    assert len(meta["source_sha256"]) >= 8
    with pytest.raises(FileExistsError):
        run("quick", tmp_path / "one")


def test_full_campaign_orchestration_with_synthetic_data(monkeypatch, tmp_path):
    # Exercise the full branch without downloads or the costly historical estimators.
    from src import experiments as ex
    from src.data import quick_data

    cfg = frozen_config()
    from src.data import data_digest

    cfg["expected_development_sha256"] = {
        t: data_digest(quick_data(t)) for t in ["classification", "regression"]
    }
    cfg["models"] = {
        task: {
            name: {
                "estimator": "DecisionTreeClassifier"
                if task == "classification"
                else "DecisionTreeRegressor",
                "params": {"max_depth": 2},
            }
            for name in members
        }
        for task, members in cfg["members"].items()
    }
    monkeypatch.setattr(ex, "frozen_config", lambda: cfg)
    monkeypatch.setattr(ex, "load_full", lambda task, *a, **k: quick_data(task))
    rows = run("full", tmp_path / "full", seeds=[42])
    assert len(rows) == 13
    meta = json.loads((tmp_path / "full/metadata.json").read_text())
    assert meta["mode"] == "full" and meta["status"] == "complete"


def test_all_frozen_models_construct_and_fit_small_data():
    from src.data import quick_data
    from threadpoolctl import threadpool_limits

    config = frozen_config()
    with threadpool_limits(limits=1):
        for task, specs in config["models"].items():
            part = quick_data(task)
            for spec in specs.values():
                model = build_model(spec, 42)
                # Construction is exact; fitting here uses reduced counts only.
                params = model.get_params(deep=False)
                if "n_estimators" in params:
                    model.set_params(n_estimators=2)
                if "max_iter" in params:
                    model.set_params(max_iter=2)
                model.fit(part.X_train, part.y_train)
                assert np.isfinite(model.predict(part.X_test)).all()
                assert node_count(model) > 0


def test_historical_results_match_raw_rows():
    path = Path(__file__).parents[1] / "results/historical"
    summary = historical_summary(path)
    assert summary["classification_accuracy_pct_mean"] == pytest.approx(86.62)
    assert summary["housing_stacking_rmse"] == pytest.approx(0.471100346)
    assert summary["housing_fit_cost_ratio"] == pytest.approx(13.3322, rel=1e-3)
    assert summary["cart_configurations"] == 90


def test_independent_replay_artifacts_match_historical_metrics():
    import csv

    root = Path(__file__).parents[1] / "results"
    with (root / "historical/part6_test_runs.csv").open() as handle:
        historical = {
            r["method"]: r
            for r in csv.DictReader(handle)
            if r["task"] == "regression" and r["seed"] == "42"
        }
    with (root / "verification/housing-seed42/metrics.csv").open() as handle:
        replay = list(csv.DictReader(handle))
    assert len(replay) == 13
    for row in replay:
        for metric in ["rmse", "mae", "nodes"]:
            assert float(row[metric]) == pytest.approx(
                float(historical[row["method"]][metric]), abs=1e-12
            )
    metadata = json.loads((root / "verification/housing-seed42/metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["seeds"] == [42]
    assert metadata["synthetic"] is False
    assert (
        metadata["data"]["regression"]["development_sha256"]
        == frozen_config()["expected_development_sha256"]["regression"]
    )
