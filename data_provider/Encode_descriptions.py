from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch


MODEL_SPECS = {
    "gpt5": {"description_tag": "gpt_5", "feature_tag": "longclip_b_ctx248_trunc_model_gpt_5"},
    "qwen3_7_plus": {"description_tag": "qwen3_7_plus", "feature_tag": "longclip_b_ctx248_trunc_model_qwen3_7_plus"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--data_root", type=lambda value: Path(value).expanduser().resolve(), default="../data/sta_data/single_multi")
    parser.add_argument("--encoder_root", type=lambda value: Path(value).expanduser().resolve(), default=".")
    parser.add_argument("--checkpoint", type=lambda value: Path(value).expanduser().resolve(), default="../checkpoints/longclip/longclip-B.pt")
    parser.add_argument("--longclip_root", type=lambda value: Path(value).expanduser().resolve(), default="../Long-CLIP")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch", type=int, nargs="+", choices=(1, 2, 3), default=(1, 2, 3))
    parser.add_argument("--model", nargs="+", choices=tuple(MODEL_SPECS), default=tuple(MODEL_SPECS))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def cache_paths(model_root: Path, split: str, feature_tag: str, time_steps: int = 200) -> dict[str, Path]:
    suffix = "" if time_steps == 500 else f"_len{time_steps}"
    stem = f"{split}{suffix}_{feature_tag}"
    return {
        "features": model_root / f"text_features_{stem}.npy",
        "mask": model_root / f"text_mask_{stem}.npy",
        "ids": model_root / f"text_ids_{stem}.json",
        "meta": model_root / f"text_meta_{stem}.json",
    }

class TextFeatureStore:
    """Read encoded tokens and gather each batch in sample-ID order."""

    def __init__(self, root: str | Path, split: str, time_steps: int, feature_tag: str):
        root = Path(root)
        paths = cache_paths(root, split, feature_tag, time_steps)
        required = ("features", "mask", "ids")
        # Train/test can share the all-ID cache used by Batch1 cross-validation.
        if split in {"train", "test"} and not all(paths[key].is_file() for key in required):
            all_paths = cache_paths(root, "all", feature_tag, time_steps)
            if all(all_paths[key].is_file() for key in required):
                paths = all_paths
        self.features = np.load(paths["features"], mmap_mode="r")
        self.mask = np.load(paths["mask"], mmap_mode="r")
        with paths["ids"].open("r", encoding="utf-8") as handle:
            ids = json.load(handle)
        if len(ids) != len(self.features) or len(ids) != len(self.mask):
            raise ValueError(f"LongCLIP cache length mismatch: features={len(self.features)}, mask={len(self.mask)}, ids={len(ids)}, root={root}")
        self.id_to_row = {str(sample_id): row for row, sample_id in enumerate(ids)}

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[-1])

    def gather(self, sample_ids, device: str):
        rows = [self.id_to_row[str(int(sample_id))] for sample_id in sample_ids]
        features = np.asarray(self.features[rows], dtype=np.float32)
        mask = np.asarray(self.mask[rows], dtype=np.bool_)
        return torch.from_numpy(features).to(device), torch.from_numpy(mask).to(device)


def validate_existing(paths: dict[str, Path], expected_count: int) -> bool:
    existing = {key: path.is_file() for key, path in paths.items()}
    if not any(existing.values()):
        return False
    if not all(existing.values()):
        raise FileExistsError(f"Partial cache exists; refusing to overwrite: {existing}")
    features = np.load(paths["features"], mmap_mode="r")
    mask = np.load(paths["mask"], mmap_mode="r")
    with paths["ids"].open("r", encoding="utf-8") as handle:
        ids = json.load(handle)
    if len(features) != expected_count or len(mask) != expected_count or len(ids) != expected_count:
        raise ValueError(f"Existing cache has the wrong length: expected={expected_count}, features={len(features)}, mask={len(mask)}, ids={len(ids)}")
    print(f"SKIP complete existing cache: {paths['features']}", flush=True)
    return True


def encode_one(*, encoder, checkpoint_hash: str, data_root: Path, batch: int, model_name: str, batch_size: int) -> dict[str, Any]:
    split = "all" if batch == 1 else "train"
    dataset_root = data_root / f"PCIe_new_batch{batch}_500_ch15"
    model_root = dataset_root / model_name
    spec = MODEL_SPECS[model_name]
    description_path = model_root / f"desc_{split}_len200_{spec['description_tag']}.json"
    id_path = dataset_root / f"id_{split}.npy"
    label_path = dataset_root / f"y_{split}.npy"

    with description_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    ids = np.load(id_path, allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False)
    expected_keys = [str(int(sample_id)) for sample_id in ids]
    if set(records) != set(expected_keys):
        raise ValueError(f"Description IDs do not match {id_path}: {description_path}")
    if len(labels) != len(expected_keys):
        raise ValueError(f"Label length mismatch: {label_path}")

    descriptions: list[str] = []
    generator_counts: Counter[str] = Counter()
    for row, key in enumerate(expected_keys):
        record = records[key]
        if not isinstance(record, dict):
            raise TypeError(f"Description record {key} is not an object")
        description = str(record.get("description", "")).strip()
        if not description:
            raise ValueError(f"Empty description: ID={key}, file={description_path}")
        if record.get("label") != int(labels[row]):
            raise ValueError(f"Description label mismatch: ID={key}")
        descriptions.append(description)
        generator_counts[str(record.get("generator_model", "<MISSING>"))] += 1

    paths = cache_paths(model_root, split, str(spec["feature_tag"]))
    if validate_existing(paths, len(descriptions)):
        return {"status": "skipped", "records": len(descriptions)}

    pid = os.getpid()
    temporary = {key: path.with_name(f".{path.name}.{pid}.tmp") for key, path in paths.items()}
    if any(path.exists() for path in temporary.values()):
        raise FileExistsError(f"Temporary output already exists: {temporary}")

    context_length = int(encoder.context_length)
    embed_dim = int(encoder.embed_dim)
    features = mask = None
    truncated_count = 0
    token_sum = 0
    token_max = 0
    try:
        features = np.lib.format.open_memmap(temporary["features"], mode="w+", dtype=np.float16, shape=(len(descriptions), context_length, embed_dim))
        mask = np.lib.format.open_memmap(temporary["mask"], mode="w+", dtype=np.bool_, shape=(len(descriptions), context_length))
        total_batches = (len(descriptions) + batch_size - 1) // batch_size
        print(f"ENCODE batch{batch}/{model_name}/{split}: records={len(descriptions)}, batches={total_batches}, shape=[N,{context_length},{embed_dim}]", flush=True)
        for batch_index, start in enumerate(range(0, len(descriptions), batch_size), start=1):
            stop = min(start + batch_size, len(descriptions))
            hidden, valid_mask, lengths = encoder.encode_batch(descriptions[start:stop])
            features[start:stop] = hidden.numpy().astype(np.float16, copy=False)
            mask[start:stop] = valid_mask.numpy()
            truncated_count += sum(length > context_length for length in lengths)
            token_sum += sum(lengths)
            token_max = max(token_max, max(lengths, default=0))
            if batch_index % 100 == 0 or batch_index == total_batches:
                print(f"  {batch_index}/{total_batches} ({100.0 * stop / len(descriptions):.1f}%)", flush=True)
        features.flush()
        mask.flush()
        del features, mask
        features = mask = None
        gc.collect()

        metadata = {
            "version": 1, "created_at": datetime.now().isoformat(timespec="seconds"), "encoder": "LongCLIP",
            "checkpoint": str(encoder.checkpoint_path), "checkpoint_sha256": checkpoint_hash,
            "description_path": str(description_path.resolve()), "description_sha256": sha256(description_path),
            "id_path": str(id_path.resolve()), "id_sha256": sha256(id_path),
            "batch": batch, "split": split, "model_directory": model_name, "feature_tag": spec["feature_tag"],
            "feature_shape": [len(descriptions), context_length, embed_dim], "feature_dtype": "float16",
            "mask_shape": [len(descriptions), context_length], "mask_semantics": "true_is_valid_through_eot",
            "normalization": "per_token_l2", "record_order": "exact id_<split>.npy order",
            "num_samples": len(descriptions), "num_truncated": int(truncated_count),
            "max_original_tokens": int(token_max), "mean_original_tokens": token_sum / len(descriptions),
            "generator_model_counts": dict(generator_counts), "overwrite_policy": "never",
        }
        write_json_exclusive(temporary["ids"], expected_keys)
        write_json_exclusive(temporary["meta"], metadata)

        # The completion marker (meta) is moved last. No destination is ever
        # replaced: every final path was checked absent above.
        for key in ("features", "mask", "ids", "meta"):
            if paths[key].exists():
                raise FileExistsError(f"Refusing to overwrite {paths[key]}")
            temporary[key].rename(paths[key])
        print(f"DONE batch{batch}/{model_name}/{split}: truncated={truncated_count}, max_tokens={token_max}", flush=True)
        return {"status": "encoded", "records": len(descriptions), "truncated": truncated_count, "max_tokens": token_max}
    finally:
        if features is not None:
            del features
        if mask is not None:
            del mask
        gc.collect()
        for path in temporary.values():
            if path.exists():
                path.unlink()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    sys.path.insert(0, str(args.encoder_root))
    from src.LongClipTextEncoder import LongClipTextEncoder
    encoder = LongClipTextEncoder(checkpoint_path=args.checkpoint, repo_path=args.longclip_root, device=args.device)
    checkpoint_hash = sha256(Path(encoder.checkpoint_path))
    results: dict[str, Any] = {}
    for batch in args.batch:
        for model_name in args.model:
            key = f"batch{batch}/{model_name}"
            results[key] = encode_one(encoder=encoder, checkpoint_hash=checkpoint_hash, data_root=args.data_root, batch=batch, model_name=model_name, batch_size=args.batch_size)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
