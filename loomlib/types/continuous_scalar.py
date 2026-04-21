from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import LoomType


class ContinuousScalar(LoomType):
    """Bounded continuous scalar: tanh + affine activation, MSE loss.

    Parameterised via ``ContinuousScalar[lo, hi]``.
    Maps a single logit to the range ``[lo, hi]`` using
    ``value = midpoint + halfrange * tanh(z)``.
    """

    _lo: float
    _hi: float

    def __init__(self, lo: float = 0.0, hi: float = 1.0):
        if hi <= lo:
            raise ValueError(f"hi ({hi}) must be greater than lo ({lo})")
        self._lo = float(lo)
        self._hi = float(hi)

    def __class_getitem__(cls, params: tuple[float, float]) -> ContinuousScalar:
        lo, hi = params
        return cls(lo, hi)

    @property
    def midpoint(self) -> float:
        return (self._lo + self._hi) / 2.0

    @property
    def halfrange(self) -> float:
        return (self._hi - self._lo) / 2.0

    # -- LoomType interface --------------------------------------------------

    def logit_size(self) -> int:
        return 1

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map logit(s) to ``[lo, hi]``."""
        return self.midpoint + self.halfrange * torch.tanh(z.squeeze(-1))

    def loss(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE loss between decoded value and target."""
        predicted = self.decode(z)
        return F.mse_loss(predicted, target.float())

    def encode(self, value: Any, device: torch.device | None = None) -> torch.Tensor:
        """Inverse-tanh to recover a logit from a scalar value."""
        device = device or torch.device("cpu")
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=device)
        value = value.float().to(device)
        normed = (value - self.midpoint) / self.halfrange
        normed = normed.clamp(-0.999999, 0.999999)
        return torch.atanh(normed).unsqueeze(-1)

    def log_prob(self, z: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Negative MSE as a surrogate log-probability (unnormalised)."""
        predicted = self.decode(z)
        return -0.5 * (predicted - value.float()) ** 2

    def __repr__(self) -> str:
        return f"ContinuousScalar({self._lo}, {self._hi})"
