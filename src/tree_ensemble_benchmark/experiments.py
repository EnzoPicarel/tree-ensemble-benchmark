"""Small verification and frozen full-campaign replay with no notebook state."""

from pathlib import Path
from time import perf_counter
import json
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from threadpoolctl import threadpool_limits
from .cart import NumpyCART
from .data import quick_data, load_full, data_digest
from .models import quick_models, frozen_config, build_model
from .stacking import oof_features
from .evaluation import (
    fit_member,
    timed_features,
    node_count,
    score,
    write_csv,
    write_json,
    environment,
)


def prepare_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Choose a new output directory to protect existing results")
    path.mkdir(parents=True)
    return path


def run_task(part, models, members, *, task, seed, folds=5, fold_seed=42):
    """Train on training rows only; evaluate fixed models after fitting the meta-model.

    Timing: one thread, sequential fits; prediction includes probability validation
    and label conversion. Combination prediction costs sum member costs plus aggregation.
    """
    classes = np.unique(part.y_train) if task == "classification" else None
    blocks, val_blocks, costs = {}, {}, {}
    for name, model in models.items():
        fitted, elapsed = fit_member(model, part, task)
        block, _, latency = timed_features(fitted, part.X_test, task, classes)
        val_block, _, _ = timed_features(fitted, part.X_validation, task, classes)
        blocks[name], val_blocks[name] = block, val_block
        costs[name] = dict(
            fit_seconds=elapsed,
            predict_ms_per_1k=latency,
            nodes=node_count(fitted),
            meta_parameters=0,
        )
        del fitted
    splitter = (
        StratifiedKFold(folds, shuffle=True, random_state=fold_seed)
        if task == "classification"
        else KFold(folds, shuffle=True, random_state=fold_seed)
    )
    chosen = {name: models[name] for name in members}
    oof, fold_records = oof_features(
        chosen,
        part.X_train,
        part.y_train,
        task=task,
        folds=list(splitter.split(part.X_train, part.y_train)),
    )
    meta = (
        LogisticRegression(C=1, max_iter=2000, tol=1e-6, random_state=seed)
        if task == "classification"
        else Ridge(alpha=1)
    )
    start = perf_counter()
    meta.fit(oof, part.y_train)
    meta_seconds = perf_counter() - start
    if task == "classification" and meta.n_iter_.max() >= meta.max_iter:
        raise RuntimeError("Stacking meta-model did not converge")
    records = []
    # Test labels are first used here, after all fitting. They never tune a configuration.
    for name, block in blocks.items():
        prediction = classes[block.argmax(axis=1)] if classes is not None else block[:, 0]
        validation = (
            classes[val_blocks[name].argmax(axis=1)]
            if classes is not None
            else val_blocks[name][:, 0]
        )
        records.append(
            dict(
                task=task,
                method=name,
                seed=seed,
                **score(part.y_test, prediction, task),
                **{
                    f"validation_{k}": v
                    for k, v in score(part.y_validation, validation, task).items()
                },
                **costs[name],
            )
        )
    combinations = (
        ["Hard voting", "Soft voting", "Stacking"]
        if classes is not None
        else ["Averaging", "Stacking"]
    )
    for method in combinations:
        outputs = []
        aggregation_seconds = 0
        for source, timed in [(blocks, True), (val_blocks, False)]:
            start = perf_counter()
            pieces = [source[name] for name in members]
            if method == "Stacking":
                prediction = meta.predict(np.concatenate(pieces, axis=1))
            elif method == "Hard voting":
                votes = np.column_stack([b.argmax(axis=1) for b in pieces])
                prediction = classes[
                    (votes[:, :, None] == np.arange(len(classes))).sum(axis=1).argmax(axis=1)
                ]
            elif classes is not None:
                prediction = classes[np.mean(pieces, axis=0).argmax(axis=1)]
            else:
                prediction = np.mean(pieces, axis=0)[:, 0]
            if timed:
                aggregation_seconds = perf_counter() - start
            outputs.append(prediction)
        cost = {
            key: sum(costs[name][key] for name in members)
            for key in ("fit_seconds", "predict_ms_per_1k", "nodes")
        }
        cost["predict_ms_per_1k"] += aggregation_seconds * 1e6 / len(part.y_test)
        cost["meta_parameters"] = 0
        if method == "Stacking":
            cost["fit_seconds"] += sum(r["fit_seconds"] for r in fold_records) + meta_seconds
            cost["meta_parameters"] = int(meta.coef_.size + np.asarray(meta.intercept_).size)
        records.append(
            dict(
                task=task,
                method=method,
                seed=seed,
                **score(part.y_test, outputs[0], task),
                **{
                    f"validation_{k}": v
                    for k, v in score(part.y_validation, outputs[1], task).items()
                },
                **cost,
            )
        )
    return records, fold_records


def run(mode: str, output: Path, *, data_dir=Path("data"), download=False, tasks=None, seeds=None):
    if mode not in {"quick", "full"}:
        raise ValueError("Unknown mode")
    tasks = tasks or ["classification", "regression"]
    if not tasks or not set(tasks) <= {"classification", "regression"}:
        raise ValueError("Unknown task")
    config = frozen_config()
    seeds = list(seeds) if seeds is not None else ([42] if mode == "quick" else config["seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    output = prepare_output(output)
    metadata = dict(
        mode=mode,
        status="incomplete",
        environment=environment(),
        seeds=seeds,
        tasks=tasks,
        fold_seed=42,
        folds=5,
        threads=1,
        synthetic=mode == "quick",
        data={},
        model_settings={},
        timing="Sequential single-thread fits; prediction includes feature validation; combination latency is additive.",
    )
    metadata["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    metadata["frozen_config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    write_json(output / "metadata.json", metadata)
    all_rows, all_folds = [], []
    with threadpool_limits(limits=1):
        for task in tasks:
            part = (
                quick_data(task)
                if mode == "quick"
                else load_full(task, data_dir, download=download)
            )
            if mode == "full" and data_digest(part) != config["expected_development_sha256"][task]:
                raise ValueError(
                    "Dataset development fingerprint differs from the historical campaign"
                )
            metadata["data"][task] = dict(
                development_sha256=data_digest(part),
                test_sha256=hashlib.sha256(
                    np.ascontiguousarray(part.X_test).tobytes()
                    + np.ascontiguousarray(part.y_test).tobytes()
                ).hexdigest(),
                rows={k: len(v) for k, v in part.indices.items()},
            )
            np.savez_compressed(output / f"{task}_split_indices.npz", **part.indices)
            for seed in seeds:
                models = (
                    quick_models(task, seed)
                    if mode == "quick"
                    else {
                        name: build_model(spec, seed)
                        for name, spec in config["models"][task].items()
                    }
                )
                metadata["model_settings"][f"{task}/{seed}"] = {
                    name: repr(m) for name, m in models.items()
                }
                members = (
                    ["Tree", "Random forest", "AdaBoost", "Gradient boosting"]
                    if mode == "quick"
                    else config["members"][task]
                )
                rows, fold_rows = run_task(part, models, members, task=task, seed=seed)
                all_rows.extend(rows)
                all_folds.extend(dict(task=task, seed=seed, **row) for row in fold_rows)
                # Checkpoint small result tables after every completed task/seed.
                write_csv(output / "metrics.csv", all_rows)
                write_csv(output / "oof_folds.csv", all_folds)
                write_json(output / "metadata.json", metadata)
                print(f"Completed {mode}: {task}, seed {seed}", flush=True)
        if mode == "quick":
            custom = NumpyCART(max_depth=4).fit(
                quick_data("classification").X_train, quick_data("classification").y_train
            )
            part = quick_data("classification")
            write_json(
                output / "cart.json",
                dict(
                    synthetic=True,
                    nodes=custom.n_nodes_,
                    **score(part.y_test, custom.predict(part.X_test), "classification"),
                ),
            )
    metadata["status"] = "complete"
    write_json(output / "metadata.json", metadata)
    return all_rows
