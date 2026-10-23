import gzip
import numpy as np
import pytest
from tree_ensemble_benchmark.data import (
    check_indices,
    fashion_indices,
    quick_data,
    random_indices,
    read_idx,
    load_full,
)


def test_original_fashion_split_sizes_balance_and_isolation():
    y = np.tile(np.arange(10), 7000)
    idx = fashion_indices(y)
    assert [len(idx[k]) for k in ["train", "validation", "test"]] == [10000, 5000, 10000]
    assert idx["train"].max() < 60000 and idx["validation"].max() < 60000
    assert idx["test"].min() == 60000
    np.testing.assert_array_equal(np.bincount(y[idx["train"]]), [1000] * 10)
    np.testing.assert_array_equal(np.bincount(y[idx["validation"]]), [500] * 10)
    for key, value in idx.items():
        np.testing.assert_array_equal(value, fashion_indices(y)[key])


def test_housing_split_sizes_and_duplicate_rejection():
    idx = random_indices(np.zeros(20640))
    assert [len(idx[k]) for k in ["train", "validation", "test"]] == [12384, 4128, 4128]
    check_indices(idx, 20640)
    with pytest.raises(ValueError):
        check_indices(dict(train=np.array([0, 1]), validation=np.array([1]), test=np.array([2])), 3)


def test_quick_data_is_reproducible_and_download_is_opt_in(tmp_path):
    for task in ["classification", "regression"]:
        a, b = quick_data(task), quick_data(task)
        np.testing.assert_array_equal(a.X_train, b.X_train)
    with pytest.raises(FileNotFoundError):
        load_full("classification", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_idx_payload_validation(tmp_path):
    path = tmp_path / "labels.gz"
    with gzip.open(path, "wb") as f:
        f.write((2049).to_bytes(4, "big") + (2).to_bytes(4, "big") + bytes([0, 1]))
    np.testing.assert_array_equal(read_idx(path, image=False), [0, 1])
    with gzip.open(path, "wb") as f:
        f.write(b"broken")
    with pytest.raises(ValueError, match="header"):
        read_idx(path, image=False)
