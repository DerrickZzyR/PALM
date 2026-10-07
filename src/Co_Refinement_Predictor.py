"""Co-Refinement Predictor: multimodal fusion of patch, text, and multi-port features."""
import torch
import torch.nn as nn
from .multi_port_encoder import MultiPortEncoder

class _MaskedAttentionPooling(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.score = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        if mask is not None:
            weights = weights * mask.unsqueeze(-1).float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return (weights * x).sum(dim=1)

class MultimodalFusionHead(nn.Module):
    """Fuse patch, text, and multi-port features for classification."""

    def __init__(self, embedding_dim: int, num_classes: int=2, hidden_dim: int=512, num_heads: int=4, dropout: float=0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, embedding_dim))
        self.ffn_ln = nn.LayerNorm(embedding_dim)
        self.raw_pool = _MaskedAttentionPooling(embedding_dim)
        self.fused_pool = _MaskedAttentionPooling(embedding_dim)
        self.txt_pool = _MaskedAttentionPooling(embedding_dim)
        self.fusion_mlp = nn.Sequential(nn.Linear(embedding_dim * 6 + 512, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.LayerNorm(hidden_dim // 2), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        with torch.no_grad():
            self.fusion_mlp[0].weight[:, embedding_dim * 6:].zero_()

    def forward(self, patch_tokens: torch.Tensor, txt_tokens: torch.Tensor, h_multi: torch.Tensor, patch_mask: torch.Tensor | None=None, txt_mask: torch.Tensor | None=None):
        if h_multi.ndim != 2 or h_multi.shape[-1] != 512:
            raise ValueError('h_multi must have shape [B,512]')
        if h_multi.shape[0] != patch_tokens.shape[0]:
            raise ValueError('h_multi and patch batch sizes must match')
        key_padding_mask = ~txt_mask.bool() if txt_mask is not None else None
        attention_output, _ = self.cross_attn(patch_tokens, txt_tokens, txt_tokens, key_padding_mask=key_padding_mask, need_weights=False)
        fused_tokens = self.attn_ln(patch_tokens + attention_output)
        fused_tokens = self.ffn_ln(fused_tokens + self.ffn(fused_tokens))
        raw_vector = self.raw_pool(patch_tokens, patch_mask)
        fused_vector = self.fused_pool(fused_tokens, patch_mask)
        text_vector = self.txt_pool(txt_tokens, txt_mask)
        six_route_features = torch.cat([raw_vector, fused_vector, torch.abs(fused_vector - text_vector), fused_vector * text_vector, torch.abs(fused_vector - raw_vector), fused_vector * raw_vector], dim=-1)
        classification_features = torch.cat([six_route_features, h_multi], dim=-1)
        logits = self.classifier(self.fusion_mlp(classification_features))
        return logits

class CRP(nn.Module):
    """Complete Co-Refinement Predictor for SSE patches, text tokens, and multi-port data."""

    def __init__(self, embed_dim: int, text_input_dim: int | None=None, num_classes: int=2, fusion_hidden_dim: int=512, num_heads: int=4, dropout: float=0.3, patch_input_dim: int | None=None, multi_ports: int=8, multi_input_channels: int=15, multi_hidden_dim: int=64, multi_context_dim: int=128, multi_dilations: tuple[int, ...]=(1, 2, 4, 8, 16, 32), multi_dropout: float=0.2, multi_revin_affine: bool=True, multi_revin_eps: float=1e-05) -> None:
        super().__init__()
        text_input_dim = embed_dim if text_input_dim is None else int(text_input_dim)
        patch_input_dim = embed_dim if patch_input_dim is None else int(patch_input_dim)
        self.patch_input_dim = patch_input_dim
        self.patch_proj = nn.LayerNorm(embed_dim) if patch_input_dim == embed_dim else nn.Sequential(nn.Linear(patch_input_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.text_proj = nn.Identity() if text_input_dim == embed_dim else nn.Sequential(nn.Linear(text_input_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.fusion_head = MultimodalFusionHead(embedding_dim=embed_dim, num_classes=num_classes, hidden_dim=fusion_hidden_dim, num_heads=num_heads, dropout=dropout)
        self.multi_port_plugin = MultiPortEncoder(ports=multi_ports, input_channels=multi_input_channels, hidden_dim=multi_hidden_dim, context_dim=multi_context_dim, dilations=multi_dilations, dropout=multi_dropout, revin_affine=multi_revin_affine, revin_eps=multi_revin_eps)

    def forward(self, patch_tokens: torch.Tensor, txt_tokens: torch.Tensor, x_multi: torch.Tensor, multi_time_mask: torch.Tensor, target_port_idx: torch.Tensor, multi_port_mask: torch.Tensor | None=None, patch_mask: torch.Tensor | None=None, txt_mask: torch.Tensor | None=None):
        h_multi = self.multi_port_plugin(x_multi=x_multi,time_mask=multi_time_mask,target_port_idx=target_port_idx,port_mask=multi_port_mask)
        return self.fusion_head(self.patch_proj(patch_tokens),self.text_proj(txt_tokens),h_multi,patch_mask=patch_mask,txt_mask=txt_mask)
    
__all__ = ['CRP']

