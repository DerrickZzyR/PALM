from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Dataset-level contract shared by TCCM, SSE, and CRP."""

    time_steps: int = 100
    num_channels: int = 15
    num_classes: int = 2

    def __post_init__(self) -> None:
        if self.time_steps <= 0:
            raise ValueError("time_steps must be positive")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")


@dataclass(frozen=True)
class TCCMModelConfig:
    """Architecture owned exclusively by the TCCM stage."""

    lag: int = 6
    affine: bool = False
    subtract_last: bool = False

    def __post_init__(self) -> None:
        if self.lag <= 0:
            raise ValueError("TCCM lag must be positive")


@dataclass(frozen=True)
class TCCMTrainConfig:
    """Optimization settings owned exclusively by the TCCM stage."""

    epochs: int = 500
    learning_rate: float = 0.05
    group_penalty: float = 0.05
    ridge_penalty: float = 1e-2
    penalty: str = "GSGL"
    use_topology_mask: bool = True
    topology_mask_path: str | Path = Path("topology_mask_ali.json")
    topology_lam: float = 1e-4
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class SSEModelConfig:
    """Architecture owned exclusively by the SSE stage."""

    embedding_dim: int = 64
    shape_size: int = 2
    shape_stride: int = 2
    sparse_rate: float = 0.5
    num_experts: int = 4
    alpha: float = 0.7
    attention_hidden_dim: int = 16
    dropout: float = 0.2
    selector_gate_strength: float = 0.0
    use_revin: bool = True
    affine: bool = False
    subtract_last: bool = False

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.shape_size <= 0 or self.shape_stride <= 0:
            raise ValueError("shape_size and shape_stride must be positive")
        if not 0.0 <= self.sparse_rate < 1.0:
            raise ValueError("sparse_rate must be in [0, 1)")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")


@dataclass(frozen=True)
class SSETrainConfig:
    """Optimization settings owned exclusively by the SSE stage."""

    epochs: int = 100
    warmup_epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    moe_loss_weight: float = 1e-3
    selector_loss_weight: float = 0.1
    warmup_start_ratio: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 10
    minimum_learning_rate: float = 1e-6
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class CRPModelConfig:
    """Architecture owned exclusively by the CRP fusion stage."""

    hidden_dim: int = 512
    num_heads: int = 4
    dropout: float = 0.3
    multi_ports: int = 8
    multi_input_channels: int = 15
    multi_hidden_dim: int = 64
    multi_context_dim: int = 128
    multi_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    multi_dropout: float = 0.2
    multi_revin_affine: bool = True
    multi_revin_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("CRP hidden_dim must be positive")
        if self.num_heads <= 0:
            raise ValueError("CRP num_heads must be positive")
        if self.multi_ports < 2:
            raise ValueError("CRP multi_ports must be at least 2")
        if self.multi_input_channels <= 0:
            raise ValueError("CRP multi_input_channels must be positive")
        if not self.multi_dilations or any(value <= 0 for value in self.multi_dilations):
            raise ValueError("CRP multi_dilations must contain positive values")


@dataclass(frozen=True)
class CRPTrainConfig:
    """Optimization settings owned exclusively by the CRP stage."""

    epochs: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    restart_period: int = 10
    restart_multiplier: int = 2
    minimum_learning_rate: float = 1e-6
    gradient_clip_norm: float = 1.0


def sse_sequence_length(data_config: DataConfig,tccm_config: TCCMModelConfig) -> int:
    """Return the explicit TCCM-to-SSE sequence-length contract."""

    length = data_config.time_steps - tccm_config.lag
    if length <= 0:
        raise ValueError(
            "time_steps must exceed the TCCM lag: "
            f"time_steps={data_config.time_steps}, lag={tccm_config.lag}"
        )
    return length
