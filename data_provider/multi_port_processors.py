"""ID-aligned access to raw multi-port groups for three-input CRP models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class MultiPortFeatureStore:
    """Gather raw 15-channel groups and target-port indices by sample ID."""

    def __init__(self, root: str | Path, split: str, expected_channels: int = 15, time_steps: int | None = None) -> None:
        root = Path(root)
        self.root = root
        paths = {'features': root / 'X_multi_groups.npy', 'pad_len': root / 'pad_len_multi_groups.npy', 'port_labels': root / 'y_multi_ports.npy', 'ids': root / f'id_{split}.npy', 'labels': root / f'y_{split}.npy', 'groups': root / f'single_to_group_{split}.npy', 'ports': root / f'single_port_index_{split}.npy'}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError('Multi-port CRP alignment files are missing:\n' + '\n'.join(missing))

        self.features = np.load(paths['features'], mmap_mode='r')
        self.pad_len = np.load(paths['pad_len'], mmap_mode='r')
        self.port_labels = np.load(paths['port_labels'], mmap_mode='r')
        ids = np.load(paths['ids'])
        labels = np.load(paths['labels'])
        self.group_indices = np.load(paths['groups'])
        self.target_port_indices = np.load(paths['ports'])

        if self.features.ndim != 4:
            raise ValueError('X_multi_groups.npy must have shape [G,P,T,C]')
        if self.features.shape[-1] != expected_channels:
            raise ValueError(f'The new CRP requires raw multi-port channels only: expected C={expected_channels}, got {self.features.shape[-1]}')
        if self.pad_len.shape != self.features.shape[:2]:
            raise ValueError('pad_len_multi_groups.npy must match [G,P]')
        if self.port_labels.shape != self.features.shape[:2]:
            raise ValueError('y_multi_ports.npy must match [G,P]')

        sample_count = len(ids)
        aligned_lengths = {'labels': len(labels), 'groups': len(self.group_indices), 'ports': len(self.target_port_indices)}
        if any(length != sample_count for length in aligned_lengths.values()):
            raise ValueError(f'Single/multi alignment length mismatch: ids={sample_count}, {aligned_lengths}')
        if len(np.unique(ids)) != sample_count:
            raise ValueError(f'id_{split}.npy contains duplicate IDs')

        group_count, port_count, source_time_steps, _ = self.features.shape
        if sample_count and (int(self.group_indices.min()) < 0 or int(self.group_indices.max()) >= group_count):
            raise IndexError('single_to_group contains an invalid group index')
        if sample_count and (int(self.target_port_indices.min()) < 0 or int(self.target_port_indices.max()) >= port_count):
            raise IndexError('single_port_index contains an invalid port index')
        if np.any(self.pad_len < 0) or np.any(self.pad_len >= source_time_steps):
            raise ValueError('Every multi-port pad length must satisfy 0 <= pad_len < T')

        selected_time_steps = source_time_steps if time_steps is None else int(time_steps)
        if not 0 < selected_time_steps <= source_time_steps:
            raise ValueError(f'Multi-port time_steps must satisfy 0 < time_steps <= {source_time_steps}, got {selected_time_steps}')

        aligned_port_labels = self.port_labels[self.group_indices, self.target_port_indices]
        mismatch_count = int(np.count_nonzero(aligned_port_labels != labels))
        if mismatch_count:
            raise ValueError(f'Port-level label alignment failed: {mismatch_count}/{sample_count} samples disagree')

        self.id_to_row = {int(sample_id): row for row, sample_id in enumerate(ids.tolist())}
        self.ports = int(port_count)
        self.source_time_steps = int(source_time_steps)
        self.time_steps = selected_time_steps
        self.time_start = self.source_time_steps - self.time_steps
        self.input_channels = int(expected_channels)
        self.split = split

    def gather(self, sample_ids, device: str) -> dict[str, torch.Tensor]:
        rows = [self.id_to_row[int(sample_id)] for sample_id in sample_ids]
        group_indices = np.asarray(self.group_indices[rows], dtype=np.int64)
        target_port_idx = np.asarray(self.target_port_indices[rows], dtype=np.int64)
        features = np.array(self.features[group_indices, :, self.time_start:, :], dtype=np.float32, copy=True)
        # Use every value in the selected tail; pad_len is retained only as metadata.
        time_mask = np.ones((len(group_indices), self.ports, self.time_steps), dtype=np.bool_)
        port_mask = np.ones((len(group_indices), self.ports), dtype=np.bool_)
        return {'x_multi': torch.from_numpy(features).to(device), 'multi_time_mask': torch.from_numpy(time_mask).to(device), 'target_port_idx': torch.from_numpy(target_port_idx).to(device), 'multi_port_mask': torch.from_numpy(port_mask).to(device)}


__all__ = ['MultiPortFeatureStore']
