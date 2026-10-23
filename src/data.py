"""Dataset acquisition and original fixed split construction. Downloads are opt-in."""

from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
from urllib.request import urlopen
import numpy as np
from sklearn.datasets import fetch_california_housing, make_classification, make_regression
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Partition:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    indices: dict


def check_indices(indices: dict, n: int):
    if set(indices) != {"train", "validation", "test"}:
        raise ValueError("Expected train/validation/test indices")
    arrays = [np.asarray(indices[k]) for k in ("train", "validation", "test")]
    if any(a.ndim != 1 or a.dtype.kind not in "iu" or len(a) == 0 for a in arrays):
        raise ValueError("Partitions must be nonempty integer vectors")
    flat = np.concatenate(arrays)
    if flat.min() < 0 or flat.max() >= n or len(np.unique(flat)) != len(flat):
        raise ValueError("Overlapping, duplicate, or out-of-range partition indices")


def partition(X, y, indices):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not np.isfinite(X).all():
        raise ValueError("Expected finite 2D features and aligned 1D labels")
    check_indices(indices, len(y))
    return Partition(
        *(a[indices[k]] for k in ("train", "validation", "test") for a in (X, y)), indices
    )


def random_indices(y, *, classification=False, seed=42):
    rows = np.arange(len(y))
    train, rest = train_test_split(
        rows, train_size=0.6, random_state=seed, stratify=y if classification else None
    )
    validation, test = train_test_split(
        rest, test_size=0.5, random_state=seed, stratify=y[rest] if classification else None
    )
    return dict(train=train, validation=validation, test=test)


def fashion_indices(y, seed=42):
    if len(y) != 70_000:
        raise ValueError("Expected official 60,000/10,000 Fashion-MNIST layout")
    pool, _ = train_test_split(
        np.arange(60_000), train_size=15_000, stratify=y[:60_000], random_state=seed
    )
    train, validation = train_test_split(
        pool, train_size=10_000, stratify=y[pool], random_state=seed
    )
    indices = dict(train=train, validation=validation, test=np.arange(60_000, 70_000))
    check_indices(indices, len(y))
    return indices


def quick_data(task: str, seed=42):
    if task == "classification":
        X, y = make_classification(
            n_samples=240,
            n_features=8,
            n_informative=5,
            n_redundant=0,
            n_classes=3,
            random_state=seed,
        )
    elif task == "regression":
        X, y = make_regression(
            n_samples=240, n_features=8, n_informative=5, noise=10, random_state=seed
        )
    else:
        raise ValueError("Unknown task")
    return partition(X, y, random_indices(y, classification=task == "classification", seed=seed))


def read_idx(path: Path, *, image: bool):
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    header = 16 if image else 8
    if len(raw) < header or int.from_bytes(raw[:4], "big") != (2051 if image else 2049):
        raise ValueError("Invalid IDX header")
    count = int.from_bytes(raw[4:8], "big")
    if image and (int.from_bytes(raw[8:12], "big"), int.from_bytes(raw[12:16], "big")) != (28, 28):
        raise ValueError("Expected 28x28 images")
    values = np.frombuffer(raw, dtype=np.uint8, offset=header)
    if values.size != count * (784 if image else 1):
        raise ValueError("IDX payload size differs from header")
    return values.reshape(count, 784) if image else values


def load_full(task: str, data_dir: Path, *, download=False):
    data_dir = Path(data_dir)
    if task == "classification":
        names = [
            "train-images-idx3-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]
        arrays = []
        for i, name in enumerate(names):
            path = data_dir / "fashion-mnist" / name
            if not path.exists():
                if not download:
                    raise FileNotFoundError(f"Missing {name}; pass --download explicitly")
                path.parent.mkdir(parents=True, exist_ok=True)
                url = (
                    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
                    + name
                )
                with urlopen(url, timeout=120) as response:
                    content = response.read()
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(content)
                temporary.replace(path)
            array = read_idx(path, image=i < 2)
            if len(array) != (60_000 if i % 2 == 0 else 10_000):
                raise ValueError("Unexpected official IDX partition length")
            arrays.append(array)
        X = np.concatenate(arrays[:2]).astype(np.float32)
        y = np.concatenate(arrays[2:]).astype(np.int64)
        if X.shape != (70_000, 784) or set(np.unique(y)) != set(range(10)):
            raise ValueError("Unexpected Fashion-MNIST data")
        return partition(X, y, fashion_indices(y))
    if task == "regression":
        housing = fetch_california_housing(data_home=str(data_dir), download_if_missing=download)
        X, y = (
            np.asarray(housing.data, dtype=np.float64),
            np.asarray(housing.target, dtype=np.float64),
        )
        if X.shape != (20_640, 8) or not np.isfinite(y).all():
            raise ValueError("Unexpected California Housing data")
        return partition(X, y, random_indices(y))
    raise ValueError("Unknown task")


def data_digest(part: Partition):
    digest = hashlib.sha256()
    for array in (part.X_train, part.y_train, part.X_validation, part.y_validation):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
