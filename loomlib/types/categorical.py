from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import LoomType


class Categorical(LoomType):
    """Categorical type: softmax activation, cross-entropy loss.

    Parameterised via ``Categorical[n]`` where *n* is the number of classes.
    """

    _n: int

    def __init__(self, n: int):
        self._n = n

    def __class_getitem__(cls, n: int) -> Categorical:
        return cls(n)

    # -- LoomType interface --------------------------------------------------

    def logit_size(self) -> int:
        return self._n

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Return the class index (greedy argmax)."""
        return z.argmax(dim=-1)

    def decode_probs(self, z: torch.Tensor) -> torch.Tensor:
        """Return the full probability vector."""
        return F.softmax(z, dim=-1)

    def loss(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss.

        *z*: raw logits ``(..., n)``.
        *target*: class indices ``(...)`` (long).
        """
        target = target.long()
        if z.dim() == 1:
            z = z.unsqueeze(0)
            target = target.unsqueeze(0)
        return F.cross_entropy(z, target)

    def encode(self, value: Any, device: torch.device | None = None) -> torch.Tensor:
        """One-hot encode a class index."""
        device = device or torch.device("cpu")
        if isinstance(value, torch.Tensor):
            idx = value.long()
        else:
            idx = torch.tensor(value, dtype=torch.long, device=device)
        return F.one_hot(idx, num_classes=self._n).float().to(device)

    def log_prob(self, z: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Log-probability of *value* under the softmax distribution."""
        log_probs = F.log_softmax(z, dim=-1)
        return log_probs.gather(-1, value.unsqueeze(-1).long()).squeeze(-1)

    def __repr__(self) -> str:
        return f"Categorical({self._n})"
