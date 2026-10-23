"""Verify historical aggregates and generate a standalone figure without plotting dependencies."""

from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
from .evaluation import write_json


def historical_summary(directory: Path):
    directory = Path(directory)
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    for name, expected in manifest["artifacts_sha256"].items():
        actual = directory / Path(name).name
        if hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Historical artifact changed: {actual.name}")
    with (directory / "part6_test_runs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 130 or len({(r["task"], r["method"], r["seed"]) for r in rows}) != 130:
        raise ValueError("Expected 130 distinct historical rows")
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["method"]), []).append(row)
    for (task, method), group in groups.items():
        if {r["seed"] for r in group} != {"42", "123", "2026", "7", "99"}:
            raise ValueError("Historical seed coverage differs")
        metric = "accuracy_pct" if task == "classification" else "rmse"
        with (directory / f"part6_{task}_comparison.csv").open() as handle:
            published = next(r for r in csv.DictReader(handle) if r["method"] == method)
        for field in [
            metric,
            "macro_f1" if task == "classification" else "mae",
            "fit_seconds",
            "predict_ms_per_1k",
            "nodes",
            "meta_parameters",
        ]:
            values = [float(r[field]) for r in group]
            for stat, actual in [
                ("mean", statistics.mean(values)),
                ("std", statistics.stdev(values)),
            ]:
                if not math.isclose(actual, float(published[f"{field}_{stat}"]), abs_tol=1e-9):
                    raise ValueError(f"Historical aggregate mismatch: {method}/{field}/{stat}")

    def mean(task, method, field):
        return statistics.mean(float(r[field]) for r in groups[(task, method)])

    cls = [float(r["accuracy_pct"]) for r in groups[("classification", "Stacking")]]
    stack = mean("regression", "Stacking", "rmse")
    boost = mean("regression", "Gradient boosting", "rmse")
    with (directory / "bonus_cart_runs.csv").open() as handle:
        cart = list(csv.DictReader(handle))
    with (directory / "bonus_cart_checks.csv").open() as handle:
        checks = list(csv.DictReader(handle))
    if len(cart) != 90 or len(checks) != 12 or any(r["passed"] != "True" for r in checks):
        raise ValueError("Unexpected recorded CART experiment")
    return dict(
        status="historical aggregates independently recalculated; training not rerun",
        classification_accuracy_pct_mean=statistics.mean(cls),
        classification_accuracy_pct_std=statistics.stdev(cls),
        housing_stacking_rmse=stack,
        housing_boosting_rmse=boost,
        housing_rmse_reduction_pct=100 * (boost - stack) / boost,
        housing_fit_cost_ratio=mean("regression", "Stacking", "fit_seconds")
        / mean("regression", "Gradient boosting", "fit_seconds"),
        housing_inference_cost_ratio=mean("regression", "Stacking", "predict_ms_per_1k")
        / mean("regression", "Gradient boosting", "predict_ms_per_1k"),
        cart_configurations=len(cart),
        cart_fixtures=len(checks),
        cart_agreement_min=min(float(r["holdout_agreement"]) for r in cart),
        cart_agreement_max=max(float(r["holdout_agreement"]) for r in cart),
    )


def report(directory: Path, output: Path):
    summary = historical_summary(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "historical_summary.json", summary)
    # Standalone exportable SVG. Values come only from verified CSV aggregates.
    fit, infer = summary["housing_fit_cost_ratio"], summary["housing_inference_cost_ratio"]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="300" viewBox="0 0 760 300">',
        '<rect width="760" height="300" fill="#f8fafc"/>',
        '<g font-family="sans-serif" fill="#172554">',
        '<text x="28" y="36" font-size="21">California Housing: small error gain, larger cost</text>',
        f'<text x="28" y="68" font-size="16">Stacking reduces recorded RMSE by {summary["housing_rmse_reduction_pct"]:.2f}%</text>',
        '<text x="28" y="98" font-size="13">Cost relative to gradient boosting (1x); historical machine-specific timings</text>',
    ]
    for y, title, ratio in [(130, "Training", fit), (195, "Inference", infer)]:
        lines.extend(
            [
                f'<text x="28" y="{y + 18}" font-size="15">{title}</text>',
                f'<rect x="130" y="{y}" width="{ratio * 28:.2f}" height="28" fill="#2563eb"/>',
                f'<text x="{140 + ratio * 28:.2f}" y="{y + 19}" font-size="15">{ratio:.1f}x</text>',
            ]
        )
    lines.extend(
        [
            '<text x="28" y="275" font-size="12">Five estimator seeds on fixed partitions; not independent dataset draws.</text>',
            "</g></svg>",
        ]
    )
    (output / "housing_tradeoff.svg").write_text("\n".join(lines) + "\n")
    return summary
