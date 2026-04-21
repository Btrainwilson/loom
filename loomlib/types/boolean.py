from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import LoomType


class Boolean(LoomType):
    """Boolean type: sigmoid activation, BCE loss.

    Consumes a single logit. Decodes to True/False via thresholding at 0.5.
    """

    def logit_size(self) -> int:
        return 1

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Threshold at 0.5 and return a bool tensor."""
        return (torch.sigmoid(z.squeeze(-1)) > 0.5)

    def decode_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Return P(True) as a float tensor."""
        return torch.sigmoid(z.squeeze(-1))

    def loss(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy with logits."""
        return F.binary_cross_entropy_with_logits(
            z.squeeze(-1).float(), target.float()
        )

    def encode(self, value: Any, device: torch.device | None = None) -> torch.Tensor:
        """Encode boolean as a single logit (+6 for True, -6 for False)."""
        device = device or torch.device("cpu")
        if isinstance(value, torch.Tensor):
            val = value.float()
        else:
            val = torch.tensor(float(value), device=device)
        return (val * 12.0 - 6.0).unsqueeze(-1).to(device)

    def log_prob(self, z: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Log Bernoulli probability."""
        return -F.binary_cross_entropy_with_logits(
            z.squeeze(-1).float(), value.float(), reduction="none"
        )

    def __repr__(self) -> str:
        return "Boolean()"
