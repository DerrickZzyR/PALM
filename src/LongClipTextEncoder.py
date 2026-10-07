from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LONGCLIP_CONTEXT_LENGTH = 248


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def load_official_longclip_module(repo_path: str | Path) -> ModuleType:
    """Load the official ``Long-CLIP/model`` package under an isolated name."""
    repo_path = _resolve_project_path(repo_path)
    model_dir = repo_path / "model"
    init_path = model_dir / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(
            f"Official Long-CLIP package not found: {init_path}. "
            "Clone https://github.com/beichenzbc/Long-CLIP into the configured directory."
        )

    path_digest = hashlib.sha1(str(model_dir).encode("utf-8")).hexdigest()[:12]
    package_name = f"_palm_longclip_{path_digest}"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(model_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create an import spec for {init_path}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(package)
        except Exception:
            sys.modules.pop(package_name, None)
            raise
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode

    return importlib.import_module(f"{package_name}.longclip")


def build_eot_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Return ``True`` through EOT (inclusive), and ``False`` for right padding."""
    if tokens.ndim != 2:
        raise ValueError(f"Expected token ids shaped [B,T], got {tuple(tokens.shape)}")
    eot_positions = tokens.argmax(dim=-1)
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    return positions <= eot_positions.unsqueeze(1)


class LongClipTextEncoder(nn.Module):
    """Frozen official LongCLIP text tower returning token-level features and masks."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        repo_path: str | Path = "Long-CLIP",
        device: str | torch.device = "cuda",
        normalize_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.checkpoint_path = _resolve_project_path(checkpoint_path)
        self.repo_path = _resolve_project_path(repo_path)
        self.normalize_tokens = bool(normalize_tokens)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"LongCLIP checkpoint not found: {self.checkpoint_path}")

        self.longclip = load_official_longclip_module(self.repo_path)
        self.model, _ = self.longclip.load(
            str(self.checkpoint_path),
            device=self.device,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.context_length = int(self.model.context_length)
        self.embed_dim = int(self.model.token_embedding.embedding_dim)
        if self.context_length != LONGCLIP_CONTEXT_LENGTH:
            raise ValueError(
                f"PALM expects the official LongCLIP context length "
                f"{LONGCLIP_CONTEXT_LENGTH}, but the checkpoint reports {self.context_length}."
            )
        if not hasattr(self.model, "encode_text_full"):
            raise AttributeError("The configured LongCLIP model has no encode_text_full method.")

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        tokenizer = getattr(self.longclip, "_tokenizer", None)
        if tokenizer is None:
            raise AttributeError("The official LongCLIP tokenizer instance is unavailable.")
        return [len(tokenizer.encode(str(text))) + 2 for text in texts]

    def encode_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        texts = [str(text) for text in texts]
        if not texts:
            empty_features = torch.empty(
                (0, self.context_length, self.embed_dim),
                dtype=torch.float32,
            )
            empty_mask = torch.empty((0, self.context_length), dtype=torch.bool)
            return empty_features, empty_mask, []

        original_lengths = self.token_lengths(texts)
        tokens = self.longclip.tokenize(
            texts,
            context_length=self.context_length,
            truncate=True,
        ).to(self.device)
        valid_mask = build_eot_mask(tokens)

        with torch.inference_mode():
            hidden = self.model.encode_text_full(tokens).float()
            if self.normalize_tokens:
                hidden = F.normalize(hidden, dim=-1)
            hidden = hidden.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        return hidden.cpu(), valid_mask.cpu(), original_lengths

    def forward(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, valid_mask, _ = self.encode_batch(texts)
        return hidden, valid_mask
