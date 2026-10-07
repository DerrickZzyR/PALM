from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class Ali_Dataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, time_steps: int):
        root = Path(data_root)
        self.x = np.load(root / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(root / f"y_{split}.npy")
        self.ids = np.load(root / f"id_{split}.npy")
        self.time_steps = time_steps

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return (torch.tensor(self.x[index, -self.time_steps :], dtype=torch.float32,), torch.as_tensor(self.y[index], dtype=torch.long), int(self.ids[index]),)


class FoldWindowDataset(Dataset):
    """Read one migrated CV fold from X_all/y_all/id_all by row index."""

    def __init__(self, data_root: str | Path, fold: int, split: str, time_steps: int,):
        if fold < 1:
            raise ValueError(f"fold must be one-based and positive, got {fold}")
        if split not in {"train", "val"}:
            raise ValueError(f"Fold split must be 'train' or 'val', got {split!r}")

        root = Path(data_root)
        self.x = np.load(root / "X_all.npy", mmap_mode="r")
        self.y = np.load(root / "y_all.npy")
        self.ids = np.load(root / "id_all.npy")
        self.indices = np.load(root / "cv_indices_5fold_seed42" / f"fold_{fold}_{split}_indices.npy")
        self.time_steps = time_steps

        if len(self.x) != len(self.y) or len(self.y) != len(self.ids):
            raise ValueError("X_all/y_all/id_all length mismatch: " f"X={len(self.x)}, y={len(self.y)}, ids={len(self.ids)}")
        if self.indices.ndim != 1:
            raise ValueError(f"Fold indices must be one-dimensional: {self.indices.shape}")
        if len(self.indices) and (int(self.indices.min()) < 0 or int(self.indices.max()) >= len(self.y)):
            raise IndexError(f"Fold {fold} {split} contains rows outside X_all: " f"min={int(self.indices.min())}, max={int(self.indices.max())}, " f"samples={len(self.y)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        row = int(self.indices[index])
        return (torch.tensor(self.x[row, -self.time_steps :], dtype=torch.float32,), torch.as_tensor(self.y[row], dtype=torch.long), int(self.ids[row]),)
