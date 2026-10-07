from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn as nn

class MaskedPortRevIN(nn.Module):

    def __init__(self, channels: int, eps: float=1e-05, affine: bool=True) -> None:
        super().__init__()
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, 1, channels))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        valid = time_mask.to(dtype=x.dtype).unsqueeze(-1)
        count = valid.sum(dim=2, keepdim=True)
        center = ((x * valid).sum(dim=2, keepdim=True) / count).detach()
        variance = (((x - center).square() * valid).sum(dim=2, keepdim=True) / count).detach()
        normalized = (x - center) * torch.rsqrt(variance + self.eps)
        if self.affine:
            normalized = normalized * self.affine_weight + self.affine_bias
        return normalized * valid

class ResidualTemporalBlock(nn.Module):

    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return self.activation(x + residual)

class SharedPortTemporalEncoder(nn.Module):

    def __init__(self, channels: int, hidden_dim: int, dilations: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.output_dim = hidden_dim * 3
        self.input_projection = nn.Sequential(nn.Linear(channels, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.temporal_encoder = nn.Sequential(*(ResidualTemporalBlock(hidden_dim, int(dilation), dropout) for dilation in dilations))
        self.temporal_attention = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x * time_mask.to(x.dtype).unsqueeze(-1)
        x = x.transpose(1, 2)
        temporal_mask = time_mask.to(x.dtype).unsqueeze(1)
        for block in self.temporal_encoder:
            x = block(x)
            x = x * temporal_mask
        valid = time_mask.bool().unsqueeze(1)
        count = valid.sum(dim=-1).clamp_min(1).to(x.dtype)
        temporal_mean = (x * valid.to(x.dtype)).sum(dim=-1) / count
        temporal_max = x.masked_fill(~valid, -torch.inf).amax(dim=-1)
        attention_logits = self.temporal_attention(x).masked_fill(~valid, -torch.inf)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        temporal_attention = (x * attention_weights).sum(dim=-1)
        representation = torch.cat([temporal_mean, temporal_max, temporal_attention], dim=-1)
        return self.output_norm(representation)

class MultiPortEncoder(nn.Module):
    output_dim = 512

    def __init__(self, ports: int=8, input_channels: int=15, hidden_dim: int=64, context_dim: int=128, dilations: Sequence[int]=(1, 2, 4, 8, 16, 32), dropout: float=0.2, revin_affine: bool=True, revin_eps: float=1e-05) -> None:
        super().__init__()
        self.revin = MaskedPortRevIN(channels=input_channels, eps=revin_eps, affine=revin_affine)
        self.port_encoder = SharedPortTemporalEncoder(channels=input_channels, hidden_dim=hidden_dim, dilations=dilations, dropout=dropout)
        relation_dim = self.port_encoder.output_dim * 5
        self.relation_adapter = nn.Sequential(nn.LayerNorm(relation_dim), nn.Linear(relation_dim, context_dim), nn.LayerNorm(context_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(context_dim, self.output_dim), nn.LayerNorm(self.output_dim), nn.GELU(), nn.Dropout(dropout))

    @staticmethod
    def _local_global_relation(port_repr: torch.Tensor, port_mask: torch.Tensor) -> torch.Tensor:
        valid = port_mask.to(port_repr.dtype).unsqueeze(-1)
        total = (port_repr * valid).sum(dim=1, keepdim=True)
        count = valid.sum(dim=1, keepdim=True)
        other_sum = total - port_repr * valid
        other_count = (count - valid).clamp_min(1.0)
        other_mean = other_sum / other_count * valid
        signed_difference = port_repr - other_mean
        return torch.cat([port_repr, other_mean, signed_difference, signed_difference.abs(), port_repr * other_mean], dim=-1)

    def forward(self, x_multi: torch.Tensor, time_mask: torch.Tensor, target_port_idx: torch.Tensor, port_mask: torch.Tensor | None=None) -> torch.Tensor:
        batch_size, ports, time_steps, channels = x_multi.shape
        port_mask = time_mask.bool().any(dim=-1) if port_mask is None else port_mask.bool()
        time_mask = time_mask.bool() & port_mask.unsqueeze(-1)
        safe_time_mask = time_mask.clone()
        safe_time_mask[:, :, 0] |= ~port_mask
        x_multi = torch.nan_to_num(x_multi, nan=0.0, posinf=0.0, neginf=0.0)
        x_multi = x_multi.masked_fill(~port_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        normalized = self.revin(x_multi, safe_time_mask)
        port_repr = self.port_encoder(normalized.reshape(batch_size * ports, time_steps, channels), safe_time_mask.reshape(batch_size * ports, time_steps)).reshape(batch_size, ports, -1)
        port_repr = port_repr * port_mask.to(port_repr.dtype).unsqueeze(-1)
        relation = self._local_global_relation(port_repr, port_mask)
        target_relation = relation[torch.arange(batch_size, device=x_multi.device), target_port_idx]
        return self.relation_adapter(target_relation)
