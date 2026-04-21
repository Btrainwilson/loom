from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import LoomType


class Scalar(LoomType):
    """Unbounded continuous scalar: identity activation, MSE loss.

    Consumes a single logit and passes it through directly (no activation).
    This is the default mapping for bare ``float`` annotations in a
    ``LoomModel``.  Use ``ContinuousScalar[lo, hi]`` instead when the
    value must be clamped to a known range.
    """

    def logit_size(self) -> int:
        return 1

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Identity -- return the raw logit as a scalar."""
        return z.squeeze(-1)

    def loss(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE loss between the raw logit and the target."""
        return F.mse_loss(self.decode(z), target.float())

    def encode(self, value: Any, device: torch.device | None = None) -> torch.Tensor:
        """Wrap a scalar value into a single-element logit tensor."""
        device = device or torch.device("cpu")
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=device)
        return value.float().to(device).unsqueeze(-1)

    def log_prob(self, z: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Negative MSE as a surrogate log-probability (unnormalised)."""
        return -0.5 * (self.decode(z) - value.float()) ** 2

    def __repr__(self) -> str:
        return "Scalar()"
