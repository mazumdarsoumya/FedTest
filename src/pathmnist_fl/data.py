from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import medmnist
from medmnist import INFO


@dataclass
class FederatedData:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    client_indices: list[np.ndarray]
    client_eval_indices: list[np.ndarray]
    noisy_targets: np.ndarray
    original_targets: np.ndarray
    class_names: list[str]


class IndexDataset(Dataset):
    """A dataset view whose targets can be overridden without touching source data."""

    def __init__(self, source: Dataset, indices: np.ndarray, targets: np.ndarray):
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, _ = self.source[int(self.indices[item])]
        return image, torch.tensor(int(self.targets[item]), dtype=torch.long)


def _targets(dataset: Dataset) -> np.ndarray:
    value = getattr(dataset, "targets", None)
    if value is None:
        value = getattr(dataset, "labels", None)
    if value is None:
        raise ValueError("MedMNIST dataset does not expose targets/labels")
    return np.asarray(value).reshape(-1).astype(np.int64)


def _dirichlet_partition(labels: np.ndarray, clients: int, alpha: float,
                          rng: np.random.Generator) -> list[np.ndarray]:
    if clients < 2 or alpha <= 0:
        raise ValueError("clients must be >= 2 and alpha must be > 0")
    buckets: list[list[int]] = [[] for _ in range(clients)]
    for cls in np.unique(labels):
        class_indices = np.flatnonzero(labels == cls)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(clients, alpha))
        # Balance by current partition size while retaining a Dirichlet skew.
        proportions *= np.asarray([1.0 / (len(b) + 1) for b in buckets])
        proportions /= proportions.sum()
        cuts = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        for client, part in enumerate(np.split(class_indices, cuts)):
            buckets[client].extend(part.tolist())
    result = [np.asarray(sorted(b), dtype=np.int64) for b in buckets]
    if any(len(x) == 0 for x in result):
        # Deterministic repair for rare pathological Dirichlet draws.
        sizes = [len(x) for x in result]
        donor = int(np.argmax(sizes))
        for empty in [i for i, x in enumerate(result) if len(x) == 0]:
            result[empty] = np.asarray([result[donor][-1]], dtype=np.int64)
            result[donor] = result[donor][:-1]
    flat = np.concatenate(result)
    if len(flat) != len(labels) or len(np.unique(flat)) != len(labels):
        raise RuntimeError("Partition must assign every training example exactly once")
    return result


def load_federated_data(data_dir: str, clients: int, alpha: float, seed: int,
                        noisy_client: int, noise_rate: float, download: bool) -> FederatedData:
    if not 0 <= noise_rate <= 1:
        raise ValueError("noise_rate must be in [0, 1]")
    info = INFO["pathmnist"]
    dataset_cls = getattr(medmnist, info["python_class"])
    data_root = Path(data_dir).expanduser()
    try:
        data_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot create MedMNIST data directory: {data_root}") from exc
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    ])
    train = dataset_cls(split="train", root=str(data_root), download=download, transform=transform)
    val = dataset_cls(split="val", root=str(data_root), download=download, transform=transform)
    test = dataset_cls(split="test", root=str(data_root), download=download, transform=transform)
    y = _targets(train)
    val_y = _targets(val)
    rng = np.random.default_rng(seed)
    train_parts = _dirichlet_partition(y, clients, alpha, rng)
    eval_parts = _dirichlet_partition(val_y, clients, alpha, np.random.default_rng(seed + 1))

    noisy = y.copy()
    selected = train_parts[noisy_client]
    count = int(round(len(selected) * noise_rate))
    chosen = rng.choice(selected, size=count, replace=False) if count else np.array([], dtype=np.int64)
    classes = np.unique(y)
    for idx in chosen:
        alternatives = classes[classes != y[idx]]
        noisy[idx] = int(rng.choice(alternatives))
    if count and np.any(noisy[chosen] == y[chosen]):
        raise RuntimeError("Noise injection invariant failed: a corrupted label stayed unchanged")
    return FederatedData(train, val, test, train_parts, eval_parts, noisy, y,
                         list(info["label"].values()))


def make_loader(source: Dataset, indices: np.ndarray | list[int] | range, labels: np.ndarray | None,
                batch_size: int, shuffle: bool, seed: int, workers: int = 0) -> DataLoader:
    # Evaluation callers may pass a range/list, while partitioning returns arrays.
    # Normalize here so every loader path accepts both representations.
    indices = np.asarray(indices, dtype=np.int64)
    if labels is None:
        dataset = Subset(source, indices.tolist())
    else:
        dataset = IndexDataset(source, indices, labels[indices])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      generator=generator, drop_last=False)


def client_summaries(data: FederatedData, clients: int) -> list[dict]:
    rows = []
    for i, indices in enumerate(data.client_indices):
        counts = Counter(data.original_targets[indices].tolist())
        noisy_counts = Counter(data.noisy_targets[indices].tolist())
        rows.append({"client": i, "n_train": int(len(indices)),
                     "noise_rate": (float(np.mean(data.noisy_targets[indices] != data.original_targets[indices]))),
                     "class_distribution": {str(c): int(counts.get(c, 0)) for c in range(len(data.class_names))},
                     "class_distribution_after_noise": {str(c): int(noisy_counts.get(c, 0)) for c in range(len(data.class_names))}})
    return rows
