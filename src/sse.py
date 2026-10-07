"""SSE residual feature encoder and classifier."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tccm import RevIN

class RMSNorm(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.gamma * self.scale

class SSEMLP(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gelu = nn.GELU()
        self.out_drop = nn.Dropout(0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.out_drop(self.gelu(self.fc1(x))))

class SparseDispatcher:

    def __init__(self, num_experts: int, gates: torch.Tensor):
        self.gates = gates
        nonzero = torch.nonzero(gates)
        sorted_experts, sort_indices = nonzero.sort(0)
        _, self.expert_index = sorted_experts.split(1, dim=1)
        self.batch_index = nonzero[sort_indices[:, 1], 0]
        self.part_sizes = (gates > 0).sum(0).tolist()
        expanded_gates = gates[self.batch_index.flatten()]
        self.nonzero_gates = torch.gather(expanded_gates, 1, self.expert_index)

    def dispatch(self, x: torch.Tensor):
        expanded = x[self.batch_index].squeeze(1)
        return torch.split(expanded, self.part_sizes, dim=0)

    def combine(self, expert_outputs: list[torch.Tensor]) -> torch.Tensor:
        stitched = torch.cat(expert_outputs, dim=0) * self.nonzero_gates
        combined = torch.zeros(self.gates.size(0), expert_outputs[-1].size(1), device=stitched.device, requires_grad=True)
        return combined.index_add(0, self.batch_index, stitched.float())

class MoE_Block(nn.Module):

    def __init__(self, input_size: int, output_size: int, num_experts: int, hidden_size: int, k: int=1):
        super().__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.experts = nn.ModuleList(SSEMLP(input_size, output_size, hidden_size) for _ in range(num_experts))
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts))
        nn.init.normal_(self.w_gate, std=0.02)
        self.rmsnorm = RMSNorm(output_size)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer('mean', torch.tensor([0.0]))
        self.register_buffer('std', torch.tensor([1.0]))

    @staticmethod
    def _cv_squared(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return torch.zeros(1, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean().square() + 1e-10)

    def _top_k_gating(self, x: torch.Tensor):
        probabilities = self.softmax(x @ self.w_gate)
        top_values, top_indices = probabilities.topk(min(self.k + 1, self.num_experts), dim=1)
        top_values = top_values[:, :self.k]
        top_indices = top_indices[:, :self.k]
        top_gates = top_values / (top_values.sum(dim=1, keepdim=True) + 1e-06)
        gates = torch.zeros_like(probabilities, requires_grad=True).scatter(1, top_indices, top_gates)
        load = (gates > 0).sum(dim=0)
        return gates, load

    def forward(self, x: torch.Tensor):
        batch_size, num_patches, feature_size = x.shape
        flattened = x.reshape(batch_size * num_patches, feature_size)
        gates, load = self._top_k_gating(flattened)
        balance_loss = self._cv_squared(gates.sum(0))
        balance_loss = balance_loss + self._cv_squared(load)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(flattened)
        expert_outputs = [expert(expert_input) for expert, expert_input in zip(self.experts, expert_inputs)]
        combined = flattened + dispatcher.combine(expert_outputs)
        combined = combined.view(batch_size, num_patches, feature_size)
        return self.act(self.rmsnorm(combined)), balance_loss

class ShapeEmbedLayer(nn.Module):

    def __init__(self, seq_len: int, shape_size: int, in_chans: int, embed_dim: int, stride: int):
        super().__init__()
        self.num_patches = (seq_len - shape_size) // stride + 1
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=shape_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)

class InceptionModule(nn.Module):

    def __init__(self, ni: int, nf: int, kernel_size: int=16, bottleneck: bool=True):
        super().__init__()
        kernels = [value if value % 2 else value - 1 for value in (kernel_size, kernel_size // 2, kernel_size // 4)]
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False) if bottleneck and ni > 1 else nn.Identity()
        conv_input = nf if bottleneck and ni > 1 else ni
        self.convs = nn.ModuleList(nn.Conv1d(conv_input, nf, kernel, padding=kernel // 2, bias=False) for kernel in kernels)
        self.maxconvpool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(ni, nf, 1, bias=False))
        self.bn = nn.InstanceNorm1d(nf * 4)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_x = self.bottleneck(x)
        branches = [conv(bottleneck_x) for conv in self.convs]
        branches.append(self.maxconvpool(x))
        return self.act(self.bn(torch.cat(branches, dim=1)))

@dataclass(frozen=True)
class _SSEConfig:
    residual_length: int
    shape_size: int
    num_channels: int
    embedding_dim: int
    sparse_rate: float
    num_classes: int
    use_revin: bool
    alpha: float
    shape_stride: int
    dropout: float
    selector_gate_strength: float
    affine: bool
    subtract_last: bool
    attention_hidden_dim: int
    num_experts: int
    inception_kernel_size: int = 16

    @property
    def output_dim(self) -> int:
        return self.embedding_dim * 3

class SSEOutputs(NamedTuple):
    logits: torch.Tensor
    moe_loss: torch.Tensor
    selected_indices: list[torch.Tensor]
    normalized_input: torch.Tensor
    patch_tokens: torch.Tensor
    patch_mask: torch.Tensor

class SSESelectorHead(nn.Sequential):

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__(nn.Linear(embedding_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor):
        logits = super().forward(x)
        return (logits, torch.sigmoid(logits))

class SSEWarmupLayer(nn.Module):
    """First SSE block: Inception feature extraction without pruning."""

    def __init__(self, dim: int, moe: MoE_Block, selector: nn.Module, dropout: float, selector_gate_strength: float, inception_kernel_size: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attention_head = selector
        self.out_drop = nn.Dropout(dropout)
        self.moe = moe
        self.inception = InceptionModule(dim, dim // 4, kernel_size=inception_kernel_size)
        self.sparse_concat_projection = None
        self.act = nn.GELU()
        self.include_extra_patch = False
        self.selector_gate_strength = selector_gate_strength
        self.block_mode = 'inception_only'

    def forward(self, x: torch.Tensor, valid_patch_mask: torch.Tensor, target_keep_count: int, end_depth: bool):
        if end_depth:
            raise ValueError('The fixed first block cannot be the final block')
        num_valid = int(valid_patch_mask[0].sum().item())
        if target_keep_count < num_valid:
            raise ValueError('The fixed first block does not perform pruning')
        normalized = self.norm1(x)
        _, selector_scores = self.attention_head(normalized)
        direct_x = normalized * (1.0 + self.selector_gate_strength * selector_scores.detach())
        inception_x = self.inception(self.norm2(direct_x).transpose(1, 2)).transpose(1, 2)
        output = self.act(inception_x)
        return (output, output.new_zeros(()), None, None, None, False)

class SSEFusionLayer(nn.Module):

    def __init__(self, dim: int, moe: MoE_Block, selector: nn.Module, dropout: float, selector_gate_strength: float, inception_kernel_size: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attention_head = selector
        self.out_drop = nn.Dropout(dropout)
        self.moe = moe
        self.inception = InceptionModule(dim, dim // 4, kernel_size=inception_kernel_size)
        self.act = nn.GELU()
        self.include_extra_patch = False
        self.selector_gate_strength = selector_gate_strength
        self.block_mode = 'fusion'
        self.output_dim = dim * 3

    @staticmethod
    def _gather_scores(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        return torch.gather(values, 1, selected.unsqueeze(-1))

    def forward(self, x: torch.Tensor, valid_patch_mask: torch.Tensor, target_keep_count: int, end_depth: bool):
        if not end_depth:
            raise ValueError('The SSE fusion block must be the final block')
        normalized_input = self.norm1(x)
        selector_logits, selector_scores = self.attention_head(normalized_input)
        candidate_x = normalized_input * (1.0 + self.selector_gate_strength * selector_scores.detach())
        num_valid = int(valid_patch_mask[0].sum().item())
        selected_indices = None
        if target_keep_count < num_valid:
            ranking_scores = selector_scores.squeeze(-1).masked_fill(~valid_patch_mask, torch.finfo(selector_scores.dtype).min)
            selected = torch.topk(ranking_scores, k=target_keep_count, dim=1).indices
            selected = torch.sort(selected, dim=1).values
            selected_indices = selected.unsqueeze(-1)
            direct_x = torch.gather(candidate_x, 1, selected_indices.expand(-1, -1, candidate_x.size(-1)))
            end_selector_logits = self._gather_scores(selector_logits, selected)
            end_selector_scores = self._gather_scores(selector_scores, selected)
            normalized = self.norm2(direct_x)
            inception_x = self.inception(normalized.transpose(1, 2)).transpose(1, 2)
            moe_x, moe_loss = self.moe(normalized)
        else:
            direct_x = candidate_x
            normalized = self.norm2(direct_x)
            inception_x = self.inception(normalized.transpose(1, 2)).transpose(1, 2)
            moe_x = torch.zeros_like(direct_x)
            moe_loss = direct_x.new_zeros(())
            end_selector_logits = selector_logits
            end_selector_scores = selector_scores
        output = self.act(self.out_drop(torch.cat([direct_x, inception_x, moe_x], dim=-1)))
        return (output, moe_loss, end_selector_logits, end_selector_scores, selected_indices, False)

class SSEClassPooling(nn.Module):

    def __init__(self, dim: int, num_classes: int, hidden_dim: int, dropout: float=0.1) -> None:
        super().__init__()
        self.attention_temperature = 1.0
        self.use_local_context = True
        self.use_instance_margin = False
        self.instance_margin_temperature = 1.0
        self.final_norm = RMSNorm(dim)
        self.context_encoder = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False), nn.GELU(), nn.Conv1d(dim, dim, kernel_size=1), nn.GELU())
        self.global_projection = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout))
        self.class_attention_head = nn.Sequential(nn.Linear(dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.global_cls_head = nn.Linear(dim, num_classes)
        initial_global_mix = 0.2
        self.global_mix_logit = nn.Parameter(torch.tensor(math.log(initial_global_mix / (1.0 - initial_global_mix)), dtype=torch.float32))

    def forward(self, x: torch.Tensor, instance_head: nn.Module):
        normalized = self.final_norm(x)
        global_mean = normalized.mean(dim=1)
        global_max = normalized.max(dim=1).values
        global_context = self.global_projection(torch.cat([global_mean, global_max], dim=-1))
        expanded_global = global_context.unsqueeze(1).expand_as(normalized)
        local_context = self.context_encoder(normalized.transpose(1, 2)).transpose(1, 2)
        contextual_features = torch.cat([normalized, local_context, expanded_global, torch.abs(normalized - expanded_global)], dim=-1)
        attention_logits = self.class_attention_head(contextual_features)
        class_attention = torch.softmax(attention_logits.float() / self.attention_temperature, dim=1).to(normalized.dtype)
        instance_logits = instance_head(normalized)
        patch_logits = (class_attention * instance_logits).sum(dim=1)
        global_logits = self.global_cls_head(global_context)
        global_mix = torch.sigmoid(self.global_mix_logit)
        logits = (1.0 - global_mix) * patch_logits + global_mix * global_logits
        return (logits, {'instance_logits': instance_logits, 'class_attention': class_attention, 'class_attention_logits': attention_logits, 'patch_cls_logits': patch_logits, 'global_cls_logits': global_logits, 'global_mix': global_mix, 'normalized_patch_tokens': normalized})

class SSE(nn.Module):
    """Encode residual patches and aggregate class evidence."""

    def __init__(self, seq_len: int, shape_size: int, num_channels: int, embedding_dim: int, sparse_rate: float, depth: int, num_classes: int, affine: bool, subtract_last: bool, use_revin: bool, alpha: float, attention_hidden_dim: int, num_experts: int, stride: int, use_extra_patch: bool, dropout: float, selector_gate_strength: float) -> None:
        super().__init__()
        if depth != 2:
            raise ValueError(f'SSE requires depth=2, got {depth}')
        if use_extra_patch:
            raise ValueError('SSE requires use_extra_patch=False')
        config = _SSEConfig(residual_length=seq_len, shape_size=shape_size, num_channels=num_channels, embedding_dim=embedding_dim, sparse_rate=sparse_rate, num_classes=num_classes, use_revin=use_revin, alpha=alpha, shape_stride=stride, dropout=dropout, selector_gate_strength=selector_gate_strength, affine=affine, subtract_last=subtract_last, attention_hidden_dim=attention_hidden_dim, num_experts=num_experts)
        self.seq_len = config.residual_length
        self.shape_size = config.shape_size
        self.num_channels = config.num_channels
        self.emb_dim = config.embedding_dim
        self.sparse_rate = config.sparse_rate
        self.depth = 2
        self.num_classes = config.num_classes
        self.RevIN = int(config.use_revin)
        self.raw = 1
        self.alpha = config.alpha
        self.shape_stride = config.shape_stride
        self.use_extra_patch = False
        self.use_local_context = True
        self.use_instance_margin = False
        self.dropout = config.dropout
        self.selector_gate_strength = config.selector_gate_strength
        self.selector_type = 'simple'
        self.revin_d_layer = RevIN(config.num_channels, affine=config.affine, subtract_last=config.subtract_last)
        self.revin_r_layer = RevIN(config.num_channels, affine=config.affine, subtract_last=config.subtract_last)
        self.main_projection = nn.Linear(config.num_channels, config.embedding_dim)
        self.stats_projection = nn.Linear(config.num_channels, config.embedding_dim)
        self.shape_embed = ShapeEmbedLayer(self.seq_len, config.shape_size, config.embedding_dim, config.embedding_dim, config.shape_stride)
        self.attention_head = SSESelectorHead(config.embedding_dim, config.attention_hidden_dim)
        self.moe = MoE_Block(config.embedding_dim, config.embedding_dim, config.num_experts, config.embedding_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.shape_embed.num_patches, config.embedding_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        self.shape_blocks = nn.ModuleList([SSEWarmupLayer(config.embedding_dim, self.moe, self.attention_head, config.dropout, config.selector_gate_strength, config.inception_kernel_size), SSEFusionLayer(config.embedding_dim, self.moe, self.attention_head, config.dropout, config.selector_gate_strength, config.inception_kernel_size)])
        self.concat_dim = config.output_dim
        self.output_dim = self.concat_dim
        self.projection_removed = True
        self.head = nn.Linear(self.concat_dim, config.num_classes)
        self.contextual_pool = SSEClassPooling(self.concat_dim, config.num_classes, config.attention_hidden_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _validate_inputs(self, residual: torch.Tensor, normalized_target: torch.Tensor) -> None:
        expected = (self.seq_len, self.num_channels)
        if residual.ndim != 3 or tuple(residual.shape[1:]) != expected:
            raise ValueError(f'residual must have shape [B, {expected[0]}, {expected[1]}], got {tuple(residual.shape)}')
        if normalized_target.shape != residual.shape:
            raise ValueError('normalized_target must have the same shape as residual')

    def forward(self, residual: torch.Tensor, normalized_target: torch.Tensor, *, return_debug: bool=False, sparse_rate_override: float | None=None):
        self._validate_inputs(residual, normalized_target)
        sparse_rate = self.sparse_rate if sparse_rate_override is None else sparse_rate_override
        normalized_x = self.revin_r_layer(residual, 'norm')[0] if self.RevIN else residual
        tokens = self.alpha * self.main_projection(normalized_x) + (1.0 - self.alpha) * self.stats_projection(normalized_target)
        tokens = self.shape_embed(tokens.transpose(1, 2))
        tokens = self.pos_drop(tokens + self.pos_embed)
        batch_size, initial_patch_count, _ = tokens.shape
        global_indices = torch.arange(initial_patch_count, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        moe_loss = tokens.new_zeros(())
        end_selector_logits = None
        end_selector_scores = None
        for depth_index, block in enumerate(self.shape_blocks):
            depth_progress = depth_index / (self.depth - 1)
            keep_count = max(2, math.ceil((1.0 - sparse_rate * depth_progress) * initial_patch_count))
            valid_patch_mask = global_indices != -1
            tokens, block_moe_loss, end_selector_logits, end_selector_scores, local_indices, extra_added = block(tokens, valid_patch_mask, keep_count, depth_index + 1 == self.depth)
            moe_loss = moe_loss + block_moe_loss
            if local_indices is not None:
                selected = local_indices.squeeze(-1)
                global_indices = torch.gather(global_indices, 1, selected)
                if extra_added:
                    raise RuntimeError('SSE cannot add an extra patch')
        valid_patch_mask = global_indices != -1
        logits, pooling_debug = self.contextual_pool(tokens, self.head)
        valid_count = valid_patch_mask.sum(dim=1)
        max_patches = int(valid_count.max().item())
        patch_tokens = tokens.new_zeros(batch_size, max_patches, tokens.size(-1))
        patch_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool, device=tokens.device)
        for batch_index in range(batch_size):
            count = int(valid_count[batch_index].item())
            patch_tokens[batch_index, :count] = tokens[batch_index, valid_patch_mask[batch_index]]
            patch_mask[batch_index, :count] = True
        outputs = SSEOutputs(logits=logits, moe_loss=moe_loss, selected_indices=[indices[indices != -1] for indices in global_indices], normalized_input=normalized_x, patch_tokens=patch_tokens, patch_mask=patch_mask)
        if not return_debug:
            return outputs
        if end_selector_logits is None or end_selector_scores is None:
            raise RuntimeError('Final selector outputs were not produced')
        selector_distribution = torch.softmax(end_selector_logits.squeeze(-1).float().masked_fill(~valid_patch_mask, float('-inf')), dim=1).to(end_selector_logits.dtype)
        debug = dict(pooling_debug)
        debug.update({'selector_logits': end_selector_logits, 'selector_scores': end_selector_scores, 'selector_distribution': selector_distribution, 'valid_patch_mask': valid_patch_mask, 'global_indices_with_extra': global_indices})
        return tuple(outputs) + (debug,)

def selector_distillation_loss(selector_logits: torch.Tensor, class_attention: torch.Tensor, valid_patch_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _, _, num_classes = class_attention.shape
    labels = labels.long().reshape(-1)
    class_weights = F.one_hot(labels, num_classes=num_classes).float()
    teacher = (class_attention.detach().float() * class_weights.unsqueeze(1)).sum(dim=-1)
    teacher = teacher.masked_fill(~valid_patch_mask, 0.0)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-08)
    student_logits = selector_logits.squeeze(-1).float().masked_fill(~valid_patch_mask, -1000000000.0)
    per_sample_kl = F.kl_div(F.log_softmax(student_logits, dim=1), teacher, reduction='none').sum(dim=1)
    class_losses = [per_sample_kl[labels == class_index].mean() for class_index in range(num_classes) if (labels == class_index).any()]
    return torch.stack(class_losses).mean()

__all__ = ['SSE', 'selector_distillation_loss']

