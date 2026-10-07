from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import matplotlib.ticker as mticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

if __package__:
    from .config import TCCMModelConfig, DataConfig, SSEModelConfig, sse_sequence_length
    from .data_provider.data import Ali_Dataset
    from .src.engine import load_tccm, load_sse
else:
    from config import TCCMModelConfig, DataConfig, SSEModelConfig, sse_sequence_length
    from data_provider.data import Ali_Dataset
    from src.engine import load_tccm, load_sse


DEFAULT_FEATURE_NAMES = ('baddllperrors', 'badtlperrors', 'portreceivererrors', 'recoverydiagnosticserrors', 'pcie_fatal_error', 'retire_page_dbe', 'ecc_v6_sram_uce', 'pcie_l0_recovery_count', 'temperature', 'retire_page_total', 'current_sm_utilization', 'pcie_ce_count', 'ecc_v6_dram_uce', 'retire_page_sbe', 'pcie_non_fatal_error')

APP_ROOT = Path(__file__).resolve().parent
# Full telemetry images with selected intervals and healthy causality prompts.
SYSTEM_PROMPT = 'You are a senior GPU/PCIe telemetry diagnosis AIOps expert. You will receive a telemetry image with 15 time-series sub-plots, key_segments selected by a neural model, and healthy_baseline_causality learned only from healthy data. Treat healthy_baseline_causality as the normal-behavior baseline; it is not optional background. You must use it to identify causal deviations, including expected propagation disappearing or weakening, metrics changing without their normal causal triggers, abnormal co-movement inconsistent with healthy propagation, and persistence or escalation beyond normal self-resolving dynamics. If prompt hints and visible image evidence conflict, trust the image while still explaining the deviation relative to the healthy baseline.'
USER_PROMPT = 'You are given:\n1) one telemetry image with actual values and healthy-baseline predicted values for 15 features,\n2) key_segments proposed by a neural model,\n3) healthy_baseline_causality learned only from healthy data.\n\nVisible features: {feature_list}\n\nPink-highlighted regions are anomalous intervals selected by the small model.\n\n{segment_info}{health_lib_info}Return strict JSON only:\n{{\n  "description": "<One cohesive paragraph in English, <=75 words. Mention up to 4 key time fragments in chronological order using explicit spans like t=[4-61]. Describe the main phenomenon in each fragment. Explicitly state at least one causal deviation relative to healthy_baseline_causality, such as expected propagation weakening, a metric changing without its normal trigger, abnormal co-movement, or persistence beyond healthy self-recovery. Conclude with the overall cross-feature propagation pattern that best explains the sample. Use only visible metric names and only healthy_baseline_causality-supported relations. No bullets, no headings, no extra JSON fields.>"\n}}\n'


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is unavailable. Use --device cpu.")
    return requested


def validate_args(args: argparse.Namespace, data_config: DataConfig, tccm_model_config: TCCMModelConfig) -> None:
    if not args.splits or any(split not in {"train", "test", "all"} for split in args.splits):
        raise ValueError("splits must contain only 'train', 'test', and/or 'all'")
    if len(args.splits) > 1 and args.output_file is not None:
        raise ValueError("--output_file requires exactly one split")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.save_every < 1:
        raise ValueError("--save_every must be positive")
    if args.llm_retries < 1:
        raise ValueError("--llm_retries must be positive")
    if not np.isfinite(args.full_des_y_floor) or args.full_des_y_floor <= 0:
        raise ValueError("--full_des_y_floor must be a positive finite number")
    if data_config.time_steps <= tccm_model_config.lag:
        raise ValueError("--time_steps must exceed --win_size")
    if args.sse_checkpoint is None:
        raise ValueError("--sse_checkpoint is required")
    for path in (args.data_root, args.tccm_checkpoint, args.sse_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)


def load_feature_names(path: Path | None, count: int) -> list[str]:
    if path is None:
        if count == len(DEFAULT_FEATURE_NAMES):
            return list(DEFAULT_FEATURE_NAMES)
        return [f"feature_{index}" for index in range(count)]
    with path.open("r", encoding="utf-8") as handle:
        names = json.load(handle)
    if not isinstance(names, list) or len(names) != count:
        raise ValueError(f"{path} must contain a JSON list with exactly {count} names")
    return [str(name) for name in names]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_causality_summary(path: Path | None, feature_names: Sequence[str], max_self_nodes: int = 4, max_cross_edges: int = 6) -> str:
    """Use the parent's compact healthy baseline, without sending the entire library."""
    if path is None:
        return ""
    with path.open("r", encoding="utf-8") as handle:
        library = json.load(handle)
    if not isinstance(library, dict):
        raise ValueError(f"Causality library root must be a JSON object: {path}")

    if "cross_edges" in library:
        self_dynamics = library.get("self_dynamics", {}) or {}
        cross_edges = library.get("cross_edges", {}) or {}
    elif "healthy_causal_relations" in library:
        relations = library["healthy_causal_relations"]
        if not isinstance(relations, dict):
            raise ValueError(f"healthy_causal_relations must be a JSON object: {path}")
        self_dynamics, cross_edges = {}, {}
        pattern = re.compile(r"([A-Za-z0-9_]+)\s*\(t-(\d+)\)\s*->\s*([A-Za-z0-9_]+)\s*\(t\)")
        for item in relations.get("self_dynamics", []) or []:
            if isinstance(item, dict):
                for source, _, destination in pattern.findall(str(item.get("relation", ""))):
                    if source == destination:
                        self_dynamics[destination] = 1.0
        for item in relations.get("pairwise_relations", []) or []:
            if isinstance(item, dict):
                for source, lag, destination in pattern.findall(str(item.get("relation", ""))):
                    block = cross_edges.setdefault(destination, {"is_active": True, "top_causes": []})
                    if len(block["top_causes"]) < 12:
                        block["top_causes"].append({"src": source, "strength": 1.0, "lag": float(lag)})
    else:
        raise ValueError(f"Expected healthy_causal_relations or cross_edges in {path}")

    feature_set = set(map(str, feature_names))
    self_items = [(str(node), _safe_float(score)) for node, score in self_dynamics.items() if str(node) in feature_set]
    self_items.sort(key=lambda item: item[1], reverse=True)
    edge_items = []
    for destination, block in cross_edges.items():
        destination = str(destination)
        if destination not in feature_set or not bool(block.get("is_active", False)):
            continue
        for cause in block.get("top_causes", []):
            source = str(cause.get("src", ""))
            if source in feature_set:
                edge_items.append((source, destination, _safe_float(cause.get("strength")), _safe_float(cause.get("lag"))))
    edge_items.sort(key=lambda item: item[2], reverse=True)
    self_text = ", ".join(node for node, _ in self_items[:max_self_nodes])
    edge_text = ', '.join((f'{source}->{destination}(lag~{(int(round(lag)) if lag > 0 else 1)})' for source, destination, _, lag in edge_items[:max_cross_edges]))
    if not self_text and not edge_text:
        return ""
    parts = ["healthy_baseline_causality normal-behavior baseline reference: "]
    if self_text:
        parts.append(f"strong healthy self-dynamics often appear in {self_text}. ")
    if edge_text:
        parts.append(f"healthy lag-1 cross-feature influences include {edge_text}. ")
    parts.append('Use this baseline to judge whether observed propagation follows or departs from healthy causality. If any deviation is visible, it must be reflected in the final description.\n\n')
    return "".join(parts)


def flatten_patch_indices(values: Any) -> list[int]:
    flattened: list[int] = []

    def collect(value: Any) -> None:
        if value is None:
            return
        if torch.is_tensor(value):
            flattened.extend(value.detach().cpu().reshape(-1).tolist())
        elif isinstance(value, np.ndarray):
            flattened.extend(value.reshape(-1).tolist())
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        else:
            flattened.append(int(value))

    collect(values)
    return sorted({int(index) for index in flattened if int(index) >= 0})


def patch_indices_to_intervals(patch_indices: Any, *, shape_size: int, stride: int, visible_length: int, time_offset: int = 0) -> list[list[int]]:
    """Map SSE patch indices onto the TCCM-visible target time axis."""

    intervals: list[list[int]] = []
    for patch_index in flatten_patch_indices(patch_indices):
        start = patch_index * stride - time_offset
        end = start + shape_size - 1
        start = max(0, start)
        end = min(visible_length - 1, end)
        if start <= end:
            intervals.append([start, end])

    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def compress_intervals(intervals: Iterable[Sequence[int]], *, merge_gap: int = 3, maximum: int = 4) -> list[list[int]]:
    """Compress prompt spans as in the parent; never use this to expand image highlights."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    normalized = sorted((min(int(item[0]), int(item[1])), max(int(item[0]), int(item[1]))) for item in intervals if len(item) >= 2)
    merged: list[list[int]] = []
    for start, end in normalized:
        if merged and start - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    while len(merged) > 1:
        for index, (start, end) in enumerate(merged):
            if end - start + 1 > 5:
                continue
            candidates = []
            for neighbor in (index - 1, index + 1):
                if not 0 <= neighbor < len(merged):
                    continue
                left, right = sorted((index, neighbor))
                gap = max(0, merged[right][0] - merged[left][1] - 1)
                if gap <= 12:
                    neighbor_length = merged[neighbor][1] - merged[neighbor][0] + 1
                    candidates.append((gap, -neighbor_length, neighbor))
            if not candidates:
                continue
            neighbor = min(candidates)[2]
            left, right = sorted((index, neighbor))
            merged[left] = [merged[left][0], merged[right][1]]
            del merged[right]
            break
        else:
            break
    while len(merged) > maximum:
        index = min(range(len(merged) - 1), key=lambda i: (max(0, merged[i + 1][0] - merged[i][1] - 1), merged[i][1] - merged[i][0] + merged[i + 1][1] - merged[i + 1][0] + 2))
        merged[index][1] = merged[index + 1][1]
        del merged[index + 1]
    return merged


def constrained_y_limits(true_values: np.ndarray, predicted_values: np.ndarray, y_floor: float) -> tuple[float, float] | None:
    """Return a widened Y range when a nearly flat trace would be exaggerated."""
    values = np.concatenate([np.asarray(true_values).reshape(-1), np.asarray(predicted_values).reshape(-1)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-float(y_floor), float(y_floor))
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max - value_min >= float(y_floor):
        return None
    center = (value_min + value_max) / 2.0
    return center - float(y_floor), center + float(y_floor)


def render_telemetry_image(true_values: np.ndarray, predicted_values: np.ndarray, feature_names: Sequence[str], intervals: Sequence[Sequence[int]], y_floor: float=5.0) -> bytes:
    """Render aligned [T,F] arrays as a PNG without requiring a GUI backend."""

    true_values = np.asarray(true_values)
    predicted_values = np.asarray(predicted_values)
    if true_values.ndim != 2 or predicted_values.ndim != 2:
        raise ValueError("true_values and predicted_values must have shape [T,F]")
    if true_values.shape[1] != predicted_values.shape[1]:
        raise ValueError("true/predicted feature dimensions differ")
    if len(feature_names) != true_values.shape[1]:
        raise ValueError("feature name count does not match data")

    length = min(true_values.shape[0], predicted_values.shape[0])
    if length == 0 or not feature_names:
        raise ValueError("Telemetry arrays must contain time steps and features")
    true_values = true_values[-length:]
    predicted_values = predicted_values[-length:]
    # Union overlapping/adjacent spans without filling unselected gaps.
    mask = np.zeros(length, dtype=bool)
    for start, end in intervals:
        left, right = max(0, int(start)), min(length - 1, int(end))
        if left <= right:
            mask[left:right + 1] = True
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)

    dpi = 150
    figure = Figure(figsize=(3300 / dpi, 10000 / (len(feature_names) / 3) / dpi), dpi=dpi)
    FigureCanvasAgg(figure)
    axes = np.asarray(figure.subplots(math.ceil(len(feature_names) / 3), 3), dtype=object).reshape(-1)
    x_axis = np.arange(length)

    for index, axis in enumerate(axes):
        if index >= len(feature_names):
            axis.axis("off")
            continue
        for spine in axis.spines.values():
            spine.set_zorder(0)
        for left, stop in boundaries:
            axis.axvspan(left - 0.5, stop - 0.5, color="#f4b6c2", alpha=0.18, zorder=1)
        axis.plot(x_axis, true_values[:, index], color="black", linewidth=1.2, linestyle="-", label="True", zorder=3)
        axis.plot(x_axis, predicted_values[:, index], color="red", linewidth=1.2, linestyle="--", label="Pred", zorder=3)
        axis.tick_params(axis="both", labelsize=9)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        limits = constrained_y_limits(true_values[:, index], predicted_values[:, index], y_floor)
        if limits is not None:
            center = (limits[0] + limits[1]) / 2.0
            axis.set_ylim(*limits)
            axis.yaxis.set_major_locator(mticker.FixedLocator([limits[0], center, limits[1]]))
        original_ticks = axis.get_xticks()
        if len(original_ticks) > 1:
            xmin, xmax = axis.get_xlim()
            original_ticks = original_ticks[(original_ticks >= xmin) & (original_ticks <= xmax)]
            midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
            midpoints = midpoints[(midpoints > xmin) & (midpoints < xmax)]
            axis.set_xticks(np.sort(np.concatenate([original_ticks, midpoints])))
            axis.set_xlim(xmin, xmax)
        axis.set_xlabel("Time", fontsize=11)
        axis.set_ylabel("Value", fontsize=11)
        axis.set_title(str(feature_names[index]), fontsize=15)
        axis.margins(x=0.02)
        axis.legend(fontsize=11, loc="upper right", frameon=False)

    figure.tight_layout(h_pad=1.5, w_pad=1.0)
    with BytesIO() as buffer:
        figure.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi)
        image = buffer.getvalue()
    figure.clear()
    return image


def extract_message_text(message: Any) -> str:
    def collect(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple)):
            return [piece for item in value for piece in collect(item)]
        if isinstance(value, dict):
            pieces: list[str] = []
            for key in ("text", "value", "content", "output_text"):
                pieces.extend(collect(value.get(key)))
            return pieces
        pieces = []
        for name in ("text", "value", "content", "output_text"):
            if hasattr(value, name):
                pieces.extend(collect(getattr(value, name)))
        return pieces

    pieces = collect(getattr(message, "content", None))
    if not pieces:
        pieces = collect(getattr(message, "reasoning_content", None))
    return "\n".join(dict.fromkeys(pieces)).strip()


def parse_description(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict) and set(parsed) == {"description"}:
            description = parsed["description"]
            if isinstance(description, str) and description.strip():
                return description.strip()
    return ""


def build_user_prompt(feature_names: Sequence[str], intervals: Sequence[Sequence[int]], causality_summary: str) -> str:
    compact = compress_intervals(intervals)
    segment_info = ""
    if compact:
        spans = ", ".join(f"[{start}-{end}]" for start, end in compact)
        segment_info = f'Reference key anomaly fragments from the small model: {spans}. Use these explicit time spans in the paragraph when they are visually supported.\n\n'
    return USER_PROMPT.format(feature_list=", ".join(feature_names), segment_info=segment_info, health_lib_info=causality_summary)


def create_llm_client(args: argparse.Namespace):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("The openai package is required. Run: pip install openai") from error

    api_key = args.llm_api_key or os.environ.get("PALM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set PALM_LLM_API_KEY or OPENAI_API_KEY, or pass --llm_api_key.")
    # Retries are controlled below; disable the SDK's additional retry layer.
    return OpenAI(api_key=api_key, base_url=args.llm_base_url, max_retries=0)


def request_description(client, *, model: str, image_png: bytes, user_prompt: str, temperature: float, max_tokens: int, retries: int) -> tuple[str, str | None]:
    image_b64 = base64.b64encode(image_png).decode("ascii")
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': [{'type': 'text', 'text': user_prompt}, {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_b64}'}}]}]
    attempts = max(1, retries)
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            raw_text = extract_message_text(response.choices[0].message)
            description = parse_description(raw_text)
            if description:
                return description, None
            last_error = "InvalidDescriptionJSON"
        except Exception as error:  # API SDKs expose provider-specific errors.
            last_error = f"{type(error).__name__}: {error}"
        if attempt < attempts:
            time.sleep(min(1.2 * attempt, 3.0))
    return "", last_error or "EmptyDescription"


def atomic_write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def load_json_dict(path: Path, overwrite: bool) -> dict[str, Any]:
    if overwrite or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return content


def normalize_description_model_tag(value: str | None) -> str:
    raw = "" if value is None else str(value).strip()
    if raw.lower() in {"", "none", "untagged"}:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if not normalized:
        raise ValueError(f"Invalid description model tag: {value!r}")
    return normalized


def resolve_description_model_tag(configured: str | None, model: str) -> str:
    configured = "" if configured is None else str(configured).strip()
    if not configured or configured.lower() == "auto":
        configured = str(model).strip()
    return normalize_description_model_tag(configured)


def validate_description_model_identity(args: argparse.Namespace, records: dict[str, Any], path: Path) -> None:
    model_tag = normalize_description_model_tag(args.description_model_tag)
    if not model_tag:
        return
    recorded_models = {str(record.get("generator_model")).strip() for record in records.values() if isinstance(record, dict) and record.get("generator_model")}
    if recorded_models and recorded_models != {str(args.llm_model).strip()}:
        raise ValueError(f'Description model mismatch for {path}: tag={model_tag!r}, requested_model={args.llm_model!r}, recorded_models={sorted(recorded_models)}')


def output_path_for_split(args: argparse.Namespace, split: str) -> Path:
    if args.output_file is not None:
        return args.output_file
    suffix = "" if args.time_steps == 500 else f"_len{args.time_steps}"
    shard_suffix = f"_part{args.shard_id + 1}of{args.num_shards}" if args.num_shards > 1 else ""
    model_tag = normalize_description_model_tag(args.description_model_tag)
    model_suffix = f"_{model_tag}" if model_tag else ""
    output_root = args.output_root or args.data_root
    if model_tag:
        output_root = output_root / model_tag
    return output_root / f"desc_{split}{suffix}{model_suffix}{shard_suffix}.json"


def image_output_directory(args: argparse.Namespace, split: str) -> Path:
    base = args.save_images or APP_ROOT / "outputs" / "description_full_images"
    dataset_name = args.data_root.resolve().name
    model_tag = normalize_description_model_tag(args.description_model_tag)
    return base / dataset_name / (model_tag or "untagged") / split


def shard_indices(length: int, count: int, shard_id: int) -> np.ndarray:
    return np.array_split(np.arange(length, dtype=np.int64), count)[shard_id]


def infer_description_evidence(batch_x: torch.Tensor, *, tccm, sse, data_config: DataConfig, tccm_model_config: TCCMModelConfig, sse_model_config: SSEModelConfig) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    _, raw_prediction, _, residual, normalized_target = tccm(batch_x)
    aligned_raw = batch_x[:, tccm_model_config.lag:]

    outputs = sse(residual, normalized_target, sparse_rate_override=sse_model_config.sparse_rate)
    selected_indices = outputs[2][0]
    intervals = patch_indices_to_intervals(selected_indices, shape_size=sse_model_config.shape_size, stride=sse_model_config.shape_stride, visible_length=sse_sequence_length(data_config, tccm_model_config))
    return aligned_raw[0].detach().cpu().numpy(), raw_prediction[0].detach().cpu().numpy(), intervals


def generate_split(args: argparse.Namespace, data_config: DataConfig, tccm_model_config: TCCMModelConfig, sse_model_config: SSEModelConfig, split: str, *, tccm, sse, feature_names: Sequence[str], causality_summary: str, client, device: str) -> None:
    dataset = Ali_Dataset(args.data_root, split, data_config.time_steps)
    indices = shard_indices(len(dataset), args.num_shards, args.shard_id)
    if args.limit is not None:
        indices = indices[:args.limit]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=1, shuffle=False)

    output_path = output_path_for_split(args, split)
    failure_path = output_path.with_name(f"{output_path.stem}_failures.json")
    results = load_json_dict(output_path, args.overwrite)
    failures = load_json_dict(failure_path, args.overwrite)
    validate_description_model_identity(args, results, output_path)
    save_images = bool(args.save_full_des_images)
    image_directory = image_output_directory(args, split)
    if save_images:
        image_directory.mkdir(parents=True, exist_ok=True)
    added = skipped = failed = rendered = 0

    for batch_x, batch_y, batch_id in tqdm(loader, desc=f"describe {split}"):
        sample_id = str(int(batch_id.item()))
        existing = results.get(sample_id)
        existing_description = existing.get("description") if isinstance(existing, dict) else None
        has_valid_description = isinstance(existing_description, str) and bool(existing_description.strip())
        image_path = image_directory / f"sample_{sample_id}.png"
        if has_valid_description and (not save_images or image_path.is_file()):
            skipped += 1
            continue

        batch_x = batch_x.to(device)
        with torch.inference_mode():
            true_values, predicted_values, intervals = infer_description_evidence(batch_x, tccm=tccm, sse=sse, data_config=data_config, tccm_model_config=tccm_model_config, sse_model_config=sse_model_config)
        image_png = render_telemetry_image(true_values, predicted_values, feature_names, intervals, args.full_des_y_floor)
        if save_images:
            image_path.write_bytes(image_png)

        if has_valid_description:
            skipped += 1
            continue

        if args.dry_run:
            rendered += 1
            continue

        prompt = build_user_prompt(feature_names, intervals, causality_summary)
        description, error = request_description(client, model=args.llm_model, image_png=image_png, user_prompt=prompt, temperature=args.llm_temperature, max_tokens=args.llm_max_tokens, retries=args.llm_retries)
        if not description:
            failures[sample_id] = {"label": int(batch_y.item()), "error": error}
            failed += 1
        else:
            results[sample_id] = {'description': description, 'generator_model': str(args.llm_model), 'description_model_tag': normalize_description_model_tag(args.description_model_tag) or 'untagged', 'label': int(batch_y.item())}
            failures.pop(sample_id, None)
            added += 1

        if (added + failed) % args.save_every == 0:
            atomic_write_json(output_path, results)
            atomic_write_json(failure_path, failures)

    if not args.dry_run:
        atomic_write_json(output_path, results)
        atomic_write_json(failure_path, failures)
    print(f'{split}: added={added}, existing={skipped}, failed={failed}, dry_run_rendered={rendered}, valid_total={len(results)}, output={output_path}')


def main(args: argparse.Namespace) -> None:
    data_config = DataConfig(time_steps=args.time_steps, num_channels=args.num_channels, num_classes=args.num_classes)
    tccm_model_config = TCCMModelConfig(lag=args.win_size)
    sse_model_config = SSEModelConfig(embedding_dim=args.embedding_dim, shape_size=args.shape_size, shape_stride=args.shape_stride, sparse_rate=args.sparse_rate, num_experts=args.num_experts, alpha=args.alpha, attention_hidden_dim=args.attention_hidden_dim, dropout=args.dropout, selector_gate_strength=args.selector_gate_strength, use_revin=bool(args.use_revin), affine=bool(args.affine), subtract_last=bool(args.subtract_last))
    args.description_model_tag = resolve_description_model_tag(args.description_model_tag, args.llm_model)
    validate_args(args, data_config, tccm_model_config)
    device = resolve_device(args.device)
    feature_names = load_feature_names(args.feature_names_json, data_config.num_channels)
    causality_summary = load_causality_summary(args.health_prior_path, feature_names)
    print(f"Description run: tail={data_config.time_steps} | model_tag={args.description_model_tag or 'untagged'} | device={device}")
    print(f"TCCM checkpoint: {args.tccm_checkpoint}")
    print(f"SSE checkpoint: {args.sse_checkpoint}")
    print(f"Healthy causality prompt: {args.health_prior_path} | chars={len(causality_summary)}")

    tccm = load_tccm(tccm_model_config, args.tccm_checkpoint, device, num_channels=data_config.num_channels)
    sse = load_sse(sse_model_config, args.sse_checkpoint, device, sequence_length=sse_sequence_length(data_config, tccm_model_config), num_channels=data_config.num_channels, num_classes=data_config.num_classes)
    client = None if args.dry_run else create_llm_client(args)

    for split in args.splits:
        generate_split(args, data_config, tccm_model_config, sse_model_config, split, tccm=tccm, sse=sse, feature_names=feature_names, causality_summary=causality_summary, client=client, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate telemetry descriptions", formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)

    parser.add_argument('--device', type=str, default='cuda', help='Compute device')
    parser.add_argument('--data_root', type=Path, default='', help='Dataset root directory')
    parser.add_argument('--splits', nargs='+', choices=['train', 'test', 'all'], default=['train', 'test'], help='Dataset splits to describe')
    parser.add_argument('--feature_names_json', type=Path, default=None, help='Optional JSON list of feature names')

    parser.add_argument('--output_root', type=Path, default=None, help='Description JSON output root; defaults to the dataset root')
    parser.add_argument('--output_file', type=Path, default=None, help='Explicit output file when generating one split')
    parser.add_argument('--health_prior_path', type=Path, default=APP_ROOT / 'healthy_causal_relations.json', help='Healthy causality prior JSON path')
    parser.add_argument('--save_images', type=Path, default='outputs/pictures', help='Root directory for full telemetry images')
    parser.add_argument('--save_full_des_images', type=int, choices=[0, 1], default=1, help='Save full telemetry images')
    parser.add_argument('--full_des_y_floor', type=float, default=5.0, help='Minimum half-range of each feature Y axis')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing description JSON files')
    parser.add_argument('--save_every', type=int, default=10, help='Atomically save after this many generated records')
    parser.add_argument('--limit', type=int, default=None, help='Maximum records per split; unset processes all records')
    parser.add_argument('--num_shards', type=int, default=4, help='Total number of description shards')
    parser.add_argument('--shard_id', type=int, default=0, help='Zero-based shard index')
    parser.add_argument('--dry_run', type=int, default=0, help='Run inference and plotting without calling the VLM')

    # VLM API
    parser.add_argument('--llm_model', type=str, default='gpt-5', help='Multimodal model name')
    parser.add_argument('--description_model_tag', type=str, default='auto', help='Description model tag; auto derives it from llm_model')
    parser.add_argument('--llm_max_tokens', type=int, default=300, help='Maximum VLM output tokens')
    parser.add_argument('--llm_temperature', type=float, default=0.2, help='VLM sampling temperature')
    parser.add_argument('--llm_api_key', default=None, help='API key; defaults to PALM_LLM_API_KEY or OPENAI_API_KEY')
    parser.add_argument('--llm_base_url', type=str, default='https://xxx', help='OpenAI-compatible API base URL')
    parser.add_argument('--llm_retries', type=int, default=3, help='Maximum API request attempts')

    # Checkpoint
    parser.add_argument('--tccm_checkpoint', type=Path, default=APP_ROOT / 'checkpoints/tccm_top_tail200_seed42.pth', help='TCCM checkpoint path')
    parser.add_argument('--sse_checkpoint', type=Path, default=None, help='SSE checkpoint path (required)')

    # Model configuration
    parser.add_argument('--win_size', type=int, default=6, help='TCCM lag window size')
    parser.add_argument('--time_steps', type=int, default=200, help='Number of time steps selected from each sample tail')
    parser.add_argument('--num_channels', type=int, default=15, help='Number of input features')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--embedding_dim', type=int, default=128, help='SSE embedding dimension')
    parser.add_argument('--sparse_rate', type=float, default=0.5, help='SSE patch pruning rate')
    parser.add_argument('--shape_size', type=int, default=4, help='Patch size')
    parser.add_argument('--shape_stride', type=int, default=2, help='Patch stride')
    parser.add_argument('--selector_gate_strength', type=float, default=0.0, help='Selector feature enhancement strength')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of MoE experts')
    parser.add_argument('--use_revin', type=int, choices=[0, 1], default=0, help='Enable RevIN inside SSE')
    parser.add_argument('--alpha', type=float, default=0.7, help='SSE dual-input fusion weight')
    parser.add_argument('--attention_hidden_dim', type=int, default=8, help='Attention hidden dimension')
    parser.add_argument('--affine', type=int, choices=[0, 1], default=0, help='Enable RevIN affine parameters')
    parser.add_argument('--subtract_last', type=int, choices=[0, 1], default=0, help='Enable RevIN subtract-last mode')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout probability')

    args = parser.parse_args()
    main(args)
