from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class LoomType(ABC):
    """Abstract base for all Loom type decoders.

    Every type decoder maps a logit slice to a typed value, provides a
    matching loss function, and can encode a value back into an embedding.
    """

    @abstractmethod
    def logit_size(self) -> int:
        """Number of logits this type consumes."""
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> Any:
        """Decode raw logits *z* into a typed value."""
        ...

    @abstractmethod
    def loss(self, z: torch.Tensor, target: Any) -> torch.Tensor:
        """Type-aware loss between logits *z* and ground-truth *target*."""
        ...

    @abstractmethod
    def encode(self, value: Any, device: torch.device | None = None) -> torch.Tensor:
        """Encode a typed value back into a logit-shaped or embedding tensor."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
