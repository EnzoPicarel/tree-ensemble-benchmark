# Validation

Verified with Python 3.12.3, NumPy 2.5.4 and scikit-learn 1.9.1.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m ruff check --isolated --select E4,E7,E9,F src tests main.py
```

Tests cover CART edge cases, probability normalization and stopping constraints;
OOF row isolation and coverage; training-only meta-model inputs; dataset splits;
reproducible quick experiments; and consistency of recorded results.

The 90-configuration synthetic CART rerun matches the recorded accuracy/agreement
values, with 97.5–100% held-out prediction agreement with scikit-learn. All 13 methods
in the Housing seed-42 replay match the historical RMSE, MAE and node counts exactly.
Stacking RMSE is 0.4700182631 for that seed; this is not the five-seed mean.

The complete five-seed benchmark remains historical evidence. Quick runs use
synthetic data and do not establish real-data performance. Results and original
software versions are recorded with their respective artifacts.

The [GitHub workflow](.github/workflows/tests.yml) runs the inexpensive tests, lint
and offline smoke benchmark. It never downloads benchmark datasets or runs the
full campaign. See [Actions](https://github.com/EnzoPicarel/tree-ensemble-benchmark/actions).
