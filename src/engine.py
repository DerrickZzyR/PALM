from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import CRPModelConfig, CRPTrainConfig, TCCMModelConfig, TCCMTrainConfig, DataConfig, SSEModelConfig, SSETrainConfig, sse_sequence_length
from data_provider.Encode_descriptions import TextFeatureStore
from data_provider.data import Ali_Dataset
from data_provider.multi_port_processors import MultiPortFeatureStore
from src import CRP, SSE, TCCM, selector_distillation_loss


# Shared utilities: checkpoints, data loaders, and window evaluation.
def _checkpoint_state(path: str | Path, device: str):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _save_checkpoint(model: nn.Module, path: str | Path, **metadata) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **metadata}, output_path)
    print(f"Saved checkpoint: {output_path}")


def make_loader(dataset, batch_size: int, shuffle: bool = False, seed: int = 42) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def print_classification_result(labels: np.ndarray, predictions: np.ndarray) -> None:
    print("\nClassification Report:")
    print(classification_report(labels, predictions, labels=[0, 1], digits=6, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))


@torch.no_grad()
def predict_windows(loader: DataLoader, tccm: TCCM, sse: SSE, device: str, threshold: float, *, crp: CRP | None = None, text_store: TextFeatureStore | None = None, multi_port_store: MultiPortFeatureStore | None = None) -> tuple[np.ndarray, np.ndarray]:
    if crp is not None and (text_store is None or multi_port_store is None):
        raise ValueError("CRP requires text features and aligned multi-port data")
    tccm.eval()
    sse.eval()
    if crp is not None:
        crp.eval()
    labels, predictions = [], []
    for batch_x, batch_y, batch_ids in tqdm(loader, desc="Predicting windows"):
        outputs = _forward_sse(tccm, sse, batch_x.to(device))
        logits = outputs.logits if crp is None else _forward_crp(crp, outputs, text_store, multi_port_store, batch_ids, device)
        fault_probability = torch.softmax(logits, dim=1)[:, 1]
        labels.append(batch_y.numpy())
        predictions.append((fault_probability >= threshold).long().cpu().numpy())
    return np.concatenate(labels), np.concatenate(predictions)


# TCCM: healthy-sequence prediction, causal regularization, and training.
def load_tccm(config: TCCMModelConfig, checkpoint_path: str | Path, device: str, *, num_channels: int):
    model = TCCM(num_series=num_channels, lag=config.lag, affine=config.affine, subtract_last=config.subtract_last).to(device)
    model.load_state_dict(_checkpoint_state(checkpoint_path, device), strict=True)
    model.eval()
    # Keep TCCM frozen when supplying residuals to SSE and CRP.
    model.requires_grad_(False)
    return model


def _ridge_regularize(network, strength: float):
    return strength * sum(layer.weight.square().sum() for layer in network.layers[1:])


def _group_regularize(network, strength: float, penalty: str):
    weights = network.layers[0].weight
    if penalty == "GL":
        value = torch.norm(weights, dim=(0, 2)).sum()
    elif penalty == "GSGL":
        value = torch.norm(weights, dim=(0, 2)).sum() + torch.norm(weights, dim=0).sum()
    else:
        value = sum(torch.norm(weights[:, :, :index + 1], dim=(0, 2)).sum() for index in range(weights.shape[-1]))
    return strength * value


def _load_topology_mask(path: str | Path, device: str, num_channels: int) -> torch.Tensor:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    mask = torch.tensor(payload["mask"], dtype=torch.float32, device=device)
    expected_shape = (num_channels, num_channels)
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"Topology mask shape must be {expected_shape}, got {tuple(mask.shape)}")
    if not torch.isfinite(mask).all():
        raise ValueError("Topology mask contains NaN or Inf")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Topology mask must contain only 0 and 1")
    feature_names = payload.get("feature_names")
    if feature_names is not None and len(feature_names) != num_channels:
        raise ValueError("Topology feature_names length must match num_channels")
    return mask


def _topology_regularize(tccm, topology_mask: torch.Tensor | None, strength: float):
    penalty = tccm.networks[0].layers[0].weight.new_zeros(())
    if topology_mask is None or strength <= 0:
        return penalty
    for target_index, network in enumerate(tccm.networks):
        edge_norm = torch.norm(network.layers[0].weight, dim=(0, 2))
        penalty = penalty + (edge_norm * (1.0 - topology_mask[target_index])).sum()
    return strength * penalty


def _topology_violation_stats(tccm, topology_mask: torch.Tensor | None) -> tuple[int, int, float, float]:
    if topology_mask is None:
        return 0, 0, 0.0, 0.0
    edge_norms = torch.stack([torch.norm(network.layers[0].weight, dim=(0, 2)) for network in tccm.networks])
    forbidden_norms = edge_norms[topology_mask == 0]
    maximum = forbidden_norms.max().item() if forbidden_norms.numel() else 0.0
    return int((forbidden_norms > 1e-8).sum().item()), forbidden_norms.numel(), forbidden_norms.sum().item(), maximum


def _proximal_update(network, strength: float, learning_rate: float, penalty: str) -> None:
    weights = network.layers[0].weight
    threshold = learning_rate * strength
    if penalty == "H":
        for index in range(weights.shape[-1]):
            section = weights[:, :, :index + 1]
            norm = torch.norm(section, dim=(0, 2), keepdim=True)
            section.copy_(section / norm.clamp_min(threshold) * (norm - threshold).clamp_min(0.0))
        return
    if penalty == "GSGL":
        norm = torch.norm(weights, dim=0, keepdim=True)
        weights.copy_(weights / norm.clamp_min(threshold) * (norm - threshold).clamp_min(0.0))
    norm = torch.norm(weights, dim=(0, 2), keepdim=True)
    weights.copy_(weights / norm.clamp_min(threshold) * (norm - threshold).clamp_min(0.0))


def train_tccm(*, data_config: DataConfig, tccm_model_config: TCCMModelConfig, tccm_train_config: TCCMTrainConfig, data_root: str | Path, output_checkpoint: str | Path, batch_size: int, device: str, seed: int = 42) -> None:
    dataset = Ali_Dataset(data_root, "train", data_config.time_steps)
    loader = make_loader(Subset(dataset, np.flatnonzero(dataset.y == 0)), batch_size, shuffle=True, seed=seed)
    model = TCCM(num_series=data_config.num_channels, lag=tccm_model_config.lag, affine=tccm_model_config.affine, subtract_last=tccm_model_config.subtract_last).to(device)
    batch_ridge = tccm_train_config.ridge_penalty / len(loader)
    batch_group = tccm_train_config.group_penalty / len(loader)
    topology_mask = _load_topology_mask(tccm_train_config.topology_mask_path, device, data_config.num_channels) if tccm_train_config.use_topology_mask else None
    topology_strength = tccm_train_config.topology_lam if tccm_train_config.use_topology_mask else 0.0
    train_metadata = asdict(tccm_train_config)
    train_metadata["topology_mask_path"] = str(train_metadata["topology_mask_path"])
    best_loss = float("inf")

    for epoch in range(tccm_train_config.epochs):
        model.train()
        mse_values = []
        for batch_x, _, _ in tqdm(loader, desc=f"TCCM {epoch + 1}/{tccm_train_config.epochs}"):
            model.zero_grad(set_to_none=True)
            prediction, _, _, _, target = model(batch_x.to(device))
            mse = nn.functional.mse_loss(prediction, target) * target.shape[-1]
            ridge = sum(_ridge_regularize(network, batch_ridge) for network in model.networks)
            topology_loss = _topology_regularize(model, topology_mask, topology_strength)
            loss = mse + ridge + topology_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tccm_train_config.gradient_clip_norm)

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= tccm_train_config.learning_rate * parameter.grad
                if tccm_train_config.group_penalty > 0:
                    for network in model.networks:
                        _proximal_update(network, batch_group, tccm_train_config.learning_rate, tccm_train_config.penalty)
            mse_values.append(mse.item())

        with torch.no_grad():
            average_mse = sum(mse_values) / len(mse_values)
            ridge_loss = sum(_ridge_regularize(network, tccm_train_config.ridge_penalty) for network in model.networks).item()
            group_loss = sum(_group_regularize(network, tccm_train_config.group_penalty, tccm_train_config.penalty) for network in model.networks).item()
            topology_loss = _topology_regularize(model, topology_mask, topology_strength).item()
            forbidden_nonzero, forbidden_total, forbidden_sum, forbidden_max = _topology_violation_stats(model, topology_mask)
            channels = data_config.num_channels
            mean_loss = (average_mse + ridge_loss + group_loss + topology_loss) / channels
            print(f"Epoch {epoch + 1} | Loss={mean_loss:.6f} | MSE={average_mse / channels:.6f} | Ridge={ridge_loss / channels:.6f} | Group={group_loss / channels:.6f} | Topology={topology_loss / channels:.6f} | Variable usage={100 * model.GC().float().mean():.2f}% | Forbidden edges={forbidden_nonzero}/{forbidden_total} | Forbidden norm sum/max={forbidden_sum:.6f}/{forbidden_max:.6f}")
            if mean_loss < best_loss:
                best_loss = mean_loss
                _save_checkpoint(model, output_checkpoint, data_config=asdict(data_config), tccm_model_config=asdict(tccm_model_config), tccm_train_config=train_metadata, seed=seed, best_epoch=epoch + 1, best_loss=best_loss)


# SSE: residual feature extraction, selector training, and evaluation.
def load_sse(config: SSEModelConfig, checkpoint_path: str | Path, device: str, *, sequence_length: int, num_channels: int, num_classes: int):
    model = SSE(seq_len=sequence_length, shape_size=config.shape_size, num_channels=num_channels, embedding_dim=config.embedding_dim, sparse_rate=config.sparse_rate, depth=2, num_classes=num_classes, affine=config.affine, subtract_last=config.subtract_last, alpha=config.alpha, attention_hidden_dim=config.attention_hidden_dim, num_experts=config.num_experts, stride=config.shape_stride, use_extra_patch=False, use_revin=config.use_revin, dropout=config.dropout, selector_gate_strength=config.selector_gate_strength).to(device)
    model.load_state_dict(_checkpoint_state(checkpoint_path, device), strict=True)
    model.eval()
    return model


def _forward_sse(tccm: TCCM, sse: SSE, batch_x: torch.Tensor, *, sparse_rate_override: float | None = None, return_debug: bool = False):
    # Extract residuals without backpropagating through the frozen TCCM.
    with torch.no_grad():
        _, _, _, residual, normalized_target = tccm(batch_x)
    return sse(residual, normalized_target, sparse_rate_override=sparse_rate_override, return_debug=return_debug)


def sse_loss(outputs, labels: torch.Tensor, train_config: SSETrainConfig) -> torch.Tensor:
    # Combine classification, MoE balance, and selector distillation losses.
    debug = outputs[-1]
    classification_loss = nn.functional.cross_entropy(outputs[0], labels)
    selector_loss = selector_distillation_loss(selector_logits=debug["selector_logits"], class_attention=debug["class_attention"], valid_patch_mask=debug["valid_patch_mask"], labels=labels)
    return classification_loss + train_config.moe_loss_weight * outputs[1] + train_config.selector_loss_weight * selector_loss


def train_sse(*, data_config: DataConfig, tccm_model_config: TCCMModelConfig, sse_model_config: SSEModelConfig, sse_train_config: SSETrainConfig, data_root: str | Path, tccm_checkpoint: str | Path, output_checkpoint: str | Path, batch_size: int, device: str, seed: int = 42) -> None:
    loader = make_loader(Ali_Dataset(data_root, "train", data_config.time_steps), batch_size, shuffle=True, seed=seed)
    eval_loader = DataLoader(loader.dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(seed))
    tccm = load_tccm(tccm_model_config, tccm_checkpoint, device, num_channels=data_config.num_channels)
    sse = SSE(seq_len=sse_sequence_length(data_config, tccm_model_config), shape_size=sse_model_config.shape_size, num_channels=data_config.num_channels, embedding_dim=sse_model_config.embedding_dim, sparse_rate=sse_model_config.sparse_rate, depth=2, num_classes=data_config.num_classes, affine=sse_model_config.affine, subtract_last=sse_model_config.subtract_last, alpha=sse_model_config.alpha, attention_hidden_dim=sse_model_config.attention_hidden_dim, num_experts=sse_model_config.num_experts, stride=sse_model_config.shape_stride, use_extra_patch=False, use_revin=sse_model_config.use_revin, dropout=sse_model_config.dropout, selector_gate_strength=sse_model_config.selector_gate_strength).to(device)
    optimizer = torch.optim.Adam(sse.parameters(), lr=sse_train_config.learning_rate, weight_decay=sse_train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=sse_train_config.plateau_factor, patience=sse_train_config.plateau_patience, min_lr=sse_train_config.minimum_learning_rate)
    best_score = (-float("inf"),) * 4

    for epoch in range(sse_train_config.epochs):
        warmup = epoch < sse_train_config.warmup_epochs
        # Ramp up the learning rate while disabling patch pruning.
        if warmup:
            progress = epoch / max(1, sse_train_config.warmup_epochs - 1)
            ratio = sse_train_config.warmup_start_ratio + (1.0 - sse_train_config.warmup_start_ratio) * progress
            for group in optimizer.param_groups:
                group["lr"] = sse_train_config.learning_rate * ratio
        sse.train()
        loss_sum, sample_count = 0.0, 0
        for batch_x, batch_y, _ in tqdm(loader, desc=f"SSE {epoch + 1}/{sse_train_config.epochs}"):
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = _forward_sse(tccm, sse, batch_x.to(device), sparse_rate_override=0.0 if warmup else None, return_debug=True)
            loss = sse_loss(outputs, batch_y, sse_train_config)
            loss.backward()
            nn.utils.clip_grad_norm_(sse.parameters(), sse_train_config.gradient_clip_norm)
            optimizer.step()
            loss_sum += loss.item() * batch_y.size(0)
            sample_count += batch_y.size(0)

        epoch_loss = loss_sum / sample_count
        if not warmup:
            scheduler.step(epoch_loss)
        print(f"\nEpoch {epoch + 1} | Loss={epoch_loss:.6f}")
        # Select the best checkpoint only after warmup.
        if warmup:
            print("  Warm-up: excluded from best checkpoint selection.")
            continue
        sse.eval()
        labels, predictions = [], []
        eval_loss, eval_count = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, _ in eval_loader:
                batch_y = batch_y.to(device)
                outputs = _forward_sse(tccm, sse, batch_x.to(device), return_debug=True)
                eval_loss += sse_loss(outputs, batch_y, sse_train_config).item() * batch_y.size(0)
                eval_count += batch_y.size(0)
                labels.append(batch_y.cpu().numpy())
                predictions.append(outputs[0].argmax(dim=1).cpu().numpy())
        sse.train()
        precision, recall, f1, _ = precision_recall_fscore_support(np.concatenate(labels), np.concatenate(predictions), labels=[1], zero_division=0)
        score = (float(precision[0]), float(recall[0]), float(f1[0]), -eval_loss / eval_count)
        if score > best_score:
            best_score = score
            _save_checkpoint(sse, output_checkpoint, data_config=asdict(data_config), tccm_model_config=asdict(tccm_model_config), sse_model_config=asdict(sse_model_config), sse_train_config=asdict(sse_train_config), seed=seed, best_epoch=epoch + 1, best_loss=-best_score[3], best_score=best_score)


def test_sse(*, data_config: DataConfig, tccm_model_config: TCCMModelConfig, sse_model_config: SSEModelConfig, data_root: str | Path, tccm_checkpoint: str | Path, sse_checkpoint: str | Path, batch_size: int, device: str, threshold: float) -> None:
    loader = make_loader(Ali_Dataset(data_root, "test", data_config.time_steps), batch_size)
    tccm = load_tccm(tccm_model_config, tccm_checkpoint, device, num_channels=data_config.num_channels)
    sse = load_sse(sse_model_config, sse_checkpoint, device, sequence_length=sse_sequence_length(data_config, tccm_model_config), num_channels=data_config.num_channels, num_classes=data_config.num_classes)
    labels, predictions = predict_windows(loader, tccm, sse, device, threshold)
    print_classification_result(labels, predictions)


# CRP: patch/text fusion with aligned multi-port features.
def load_crp(config: CRPModelConfig, checkpoint_path: str | Path, device: str, *, text_input_dim: int, patch_input_dim: int, num_classes: int):
    model = CRP(embed_dim=text_input_dim, text_input_dim=text_input_dim, patch_input_dim=patch_input_dim, num_classes=num_classes, fusion_hidden_dim=config.hidden_dim, num_heads=config.num_heads, dropout=config.dropout, multi_ports=config.multi_ports, multi_input_channels=config.multi_input_channels, multi_hidden_dim=config.multi_hidden_dim, multi_context_dim=config.multi_context_dim, multi_dilations=config.multi_dilations, multi_dropout=config.multi_dropout, multi_revin_affine=config.multi_revin_affine, multi_revin_eps=config.multi_revin_eps).to(device)
    model.load_state_dict(_checkpoint_state(checkpoint_path, device), strict=True)
    model.eval()
    return model


def _forward_crp(crp: CRP, sse_outputs, text_store: TextFeatureStore, multi_port_store: MultiPortFeatureStore, batch_ids: torch.Tensor, device: str):
    # Align text and multi-port features with the target window IDs.
    text_tokens, text_mask = text_store.gather(batch_ids, device)
    return crp(sse_outputs.patch_tokens, text_tokens, patch_mask=sse_outputs.patch_mask, txt_mask=text_mask, **multi_port_store.gather(batch_ids, device))


def train_crp(*, data_config: DataConfig, tccm_model_config: TCCMModelConfig, sse_model_config: SSEModelConfig, crp_model_config: CRPModelConfig, crp_train_config: CRPTrainConfig, data_root: str | Path, tccm_checkpoint: str | Path, sse_checkpoint: str | Path, crp_checkpoint: str | Path, text_root: str | Path, text_feature_tag: str, batch_size: int, device: str, multi_port_root: str | Path | None = None) -> None:
    loader = make_loader(Ali_Dataset(data_root, "train", data_config.time_steps), batch_size, shuffle=True)
    text_store = TextFeatureStore(text_root, "all", data_config.time_steps, text_feature_tag)
    multi_port_store = MultiPortFeatureStore(multi_port_root or data_root, "train", expected_channels=crp_model_config.multi_input_channels, time_steps=data_config.time_steps)
    tccm = load_tccm(tccm_model_config, tccm_checkpoint, device, num_channels=data_config.num_channels)
    sse = load_sse(sse_model_config, sse_checkpoint, device, sequence_length=sse_sequence_length(data_config, tccm_model_config), num_channels=data_config.num_channels, num_classes=data_config.num_classes)
    crp = CRP(embed_dim=text_store.input_dim, text_input_dim=text_store.input_dim, patch_input_dim=sse.output_dim, num_classes=data_config.num_classes, fusion_hidden_dim=crp_model_config.hidden_dim, num_heads=crp_model_config.num_heads, dropout=crp_model_config.dropout, multi_ports=crp_model_config.multi_ports, multi_input_channels=crp_model_config.multi_input_channels, multi_hidden_dim=crp_model_config.multi_hidden_dim, multi_context_dim=crp_model_config.multi_context_dim, multi_dilations=crp_model_config.multi_dilations, multi_dropout=crp_model_config.multi_dropout, multi_revin_affine=crp_model_config.multi_revin_affine, multi_revin_eps=crp_model_config.multi_revin_eps).to(device)
    optimizer = torch.optim.AdamW(crp.parameters(), lr=crp_train_config.learning_rate, weight_decay=crp_train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=crp_train_config.restart_period, T_mult=crp_train_config.restart_multiplier, eta_min=crp_train_config.minimum_learning_rate)
    best_loss = float("inf")

    for epoch in range(crp_train_config.epochs):
        crp.train()
        loss_sum, batch_count = 0.0, 0
        for batch_x, batch_y, batch_ids in tqdm(loader, desc=f"CRP {epoch + 1}/{crp_train_config.epochs}"):
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            # Freeze both feature encoders and train only the crp.
            with torch.no_grad():
                outputs = _forward_sse(tccm, sse, batch_x.to(device))
            logits = _forward_crp(crp, outputs, text_store, multi_port_store, batch_ids, device)
            loss = nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(crp.parameters(), crp_train_config.gradient_clip_norm)
            optimizer.step()
            loss_sum += loss.item()
            batch_count += 1

        scheduler.step()
        average_loss = loss_sum / batch_count
        print(f"Epoch {epoch + 1} | Loss={average_loss:.6f}")
        if average_loss < best_loss:
            best_loss = average_loss
            _save_checkpoint(crp, crp_checkpoint, data_config=asdict(data_config), tccm_model_config=asdict(tccm_model_config), sse_model_config=asdict(sse_model_config), crp_model_config=asdict(crp_model_config), crp_train_config=asdict(crp_train_config), best_loss=best_loss)


def test_crp(*, data_config: DataConfig, tccm_model_config: TCCMModelConfig, sse_model_config: SSEModelConfig, crp_model_config: CRPModelConfig, data_root: str | Path, tccm_checkpoint: str | Path, sse_checkpoint: str | Path, crp_checkpoint: str | Path, text_root: str | Path, text_feature_tag: str, batch_size: int, device: str, threshold: float, multi_port_root: str | Path | None = None) -> None:
    loader = make_loader(Ali_Dataset(data_root, "test", data_config.time_steps), batch_size)
    text_store = TextFeatureStore(text_root, "all", data_config.time_steps, text_feature_tag)
    multi_port_store = MultiPortFeatureStore(multi_port_root or data_root, "test", expected_channels=crp_model_config.multi_input_channels, time_steps=data_config.time_steps)
    tccm = load_tccm(tccm_model_config, tccm_checkpoint, device, num_channels=data_config.num_channels)
    sse = load_sse(sse_model_config, sse_checkpoint, device, sequence_length=sse_sequence_length(data_config, tccm_model_config), num_channels=data_config.num_channels, num_classes=data_config.num_classes)
    crp = load_crp(crp_model_config, crp_checkpoint, device, text_input_dim=text_store.input_dim, patch_input_dim=sse.output_dim, num_classes=data_config.num_classes)
    labels, predictions = predict_windows(loader, tccm, sse, device, threshold, crp=crp, text_store=text_store, multi_port_store=multi_port_store)
    print_classification_result(labels, predictions)
