import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split


def _read_samples(paths: list[Path], min_real_steps: int):
    records = []
    for path in paths:
        for sample in joblib.load(path):
            values = np.asarray(sample["data"].values,dtype=np.float32,)
            real_steps = values.shape[0] - int(sample.get("pad_len", 0))
            if np.isfinite(values).all() and real_steps >= min_real_steps:
                records.append({"values": values,"label": int(sample["label"] != 0),"sn": str(sample.get("sn", "")),"slot": str(sample.get("slot",sample.get("error_slot", ""),)),})
    return records


def _limit_health(records: list[dict], health_per_fault: int):
    health = [record for record in records if record["label"] == 0]
    fault = [record for record in records if record["label"] == 1]
    if fault:
        health = health[: len(fault) * health_per_fault]
    return health + fault


def _records_to_arrays(records: list[dict]):
    x = np.stack([record["values"] for record in records]).astype(np.float32)
    y = np.asarray([record["label"] for record in records],dtype=np.int64,)
    metadata = {"source_sn": np.asarray([record["sn"] for record in records]),"source_slot": np.asarray([record["slot"] for record in records]),"source_label": y.copy(),}
    return x, y, metadata


def _save_split(output_dir: Path, split: str, x: np.ndarray, y: np.ndarray, ids: np.ndarray, metadata: dict) -> None:
    np.save(output_dir / f"X_{split}.npy", x, allow_pickle=False)
    np.save(output_dir / f"y_{split}.npy", y, allow_pickle=False)
    np.save(output_dir / f"id_{split}.npy", ids, allow_pickle=False)
    joblib.dump({key: values[ids] for key, values in metadata.items()},output_dir / f"metadata_{split}.joblib",compress=3,)


def build_window_dataset(input_files: list[str | Path], output_dir: str | Path, *, test_files: list[str | Path] | None = None, test_size: float = 0.3, seed: int = 42, health_per_fault: int = 200, min_real_steps: int = 6, stats_source: str | Path | None = None, all_as_test: bool = False):
    """Build full sequences from joblib samples; models select tail windows at runtime."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = _read_samples([Path(path) for path in input_files],min_real_steps,)

    if all_as_test:
        if test_files:
            raise ValueError("all_as_test cannot be combined with test_files")
        all_records = train_records
        x_all, y_all, metadata = _records_to_arrays(all_records)
        train_ids = np.empty(0, dtype=np.int64)
        test_ids = np.arange(len(x_all), dtype=np.int64)
    elif test_files:
        test_records = _read_samples([Path(path) for path in test_files],min_real_steps,)
        all_records = train_records + test_records
        x_all, y_all, metadata = _records_to_arrays(all_records)
        train_ids = np.arange(len(train_records), dtype=np.int64)
        test_ids = np.arange(len(train_records),len(all_records),dtype=np.int64,)
    else:
        all_records = _limit_health(train_records,health_per_fault,)
        x_all, y_all, metadata = _records_to_arrays(all_records)
        all_ids = np.arange(len(x_all), dtype=np.int64)
        if test_size:
            train_ids, test_ids = train_test_split(all_ids,test_size=test_size,random_state=seed,shuffle=True,stratify=y_all,)
        else:
            train_ids = all_ids
            test_ids = None

    if stats_source is None:
        if len(train_ids) == 0:
            raise ValueError("all_as_test requires stats_source")
        max_feature = x_all[train_ids].max(axis=(0, 1))
        max_feature = np.where(max_feature == 0,1.0,max_feature,)
    else:
        stats_source = Path(stats_source)
        max_feature = np.load(stats_source / "max_feature.npy")

    np.save(output_dir / "max_feature.npy",max_feature,allow_pickle=False,)
    if len(train_ids):
        _save_split(output_dir,"train",x_all[train_ids],y_all[train_ids],train_ids,metadata,)
    if test_ids is not None:
        _save_split(output_dir,"test",x_all[test_ids],y_all[test_ids],test_ids,metadata,)

    manifest = {"source_files": [str(Path(path)) for path in input_files],"test_files": [str(Path(path)) for path in test_files] if test_files else [],"all_shape": list(x_all.shape),"train_count": int(len(train_ids)),"test_count": int(len(test_ids)) if test_ids is not None else 0,"all_as_test": bool(all_as_test),"label_counts": np.bincount(y_all,minlength=2,).tolist(),}
    with (output_dir / "manifest.json").open("w",encoding="utf-8",) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def build_cv_dataset(input_files: list[str | Path], output_dir: str | Path, *, min_real_steps: int = 6):

    records = _read_samples([Path(path) for path in input_files],min_real_steps,)
    if not records:
        raise ValueError("No valid samples were found in input_files")

    x_all, y_all, _ = _records_to_arrays(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_all.npy", x_all, allow_pickle=False)
    np.save(output_dir / "y_all.npy", y_all, allow_pickle=False)

    return {"source_files": [str(Path(path)) for path in input_files],"output_files": ["X_all.npy", "y_all.npy"],"all_shape": list(x_all.shape),"sample_count": int(len(y_all)),"label_counts": np.bincount(y_all, minlength=2).tolist(),"data_dtype": str(x_all.dtype),"label_dtype": str(y_all.dtype),}
