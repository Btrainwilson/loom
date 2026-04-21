from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import LoomType

class WeightedSumOfBits(LoomType):
    """Weighted sum of bits: per-bit sigmoid, dual-objective BCE + gamma*MSE loss."""

    def __init__(self, weights: torch.Tensor):
        self._weights = weights

    def __class_getitem__(cls, weights: torch.Tensor) -> WeightedSumOfBits:
        return cls(weights)

    def logit_size(self) -> int:
        return self._weights.shape[-1]

    def _bit_probs(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode logits to weighted sum of bits."""
        return (self._bit_probs(z) > 0.5).float() * self._weights.to(z.device)

    def loss(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dual-objective loss: bitwise BCE 

        *z*: raw logits ``(..., B)``.
        *target*: target logits ``(..., B)``.
        """
        return F.binary_cross_entropy_with_logits(z, target.float())

    def encode(self, value: Any, device: torch.device | None = None) -> torch.Tensor:
        """Encode weighted sum of bits as logits."""
        device = device or torch.device("cpu")
        if isinstance(value, torch.Tensor):
            value = value.float().to(device)
        else:
            value = torch.tensor(value, dtype=torch.float, device=device)
        return value.unsqueeze(-1)

    def log_prob(self, z: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Sum of per-bit log Bernoulli probabilities."""
        target_bits = self._int_to_bits(value.long(), z.device)
        return -F.binary_cross_entropy_with_logits(
            z, target_bits, reduction="none"
        ).sum(dim=-1)

    def __repr__(self) -> str:
        return f"WeightedSumOfBits({self._weights.shape[-1]})"


class BitInteger(LoomType):
    """Bitwise integer type: per-bit sigmoid, dual-objective BCE + gamma*MSE loss.

    Parameterised via ``BitInteger[B]`` where *B* is the number of bits.
    Decodes to an integer in ``[0, 2^B - 1]``.
    """

    _bits: int

    def __init__(self, bits: int, weight_dtype: torch.dtype = torch.float32):
        self._bits = bits
        self._weights = torch.tensor(
            [2**i for i in range(bits)], dtype=weight_dtype
        )

    def __class_getitem__(cls, bits: int, weight_dtype: torch.dtype = torch.float32) -> BitInteger:
        return cls(bits, weight_dtype=weight_dtype)

    # -- LoomType interface --------------------------------------------------

    def logit_size(self) -> int:
        return self._bits

    def _bit_probs(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode logits to integer(s) via thresholded bit probabilities."""
        bits = (self._bit_probs(z) > 0.5).float()
        weights = self._weights.to(z.device)
        return (bits * weights).sum(dim=-1).long()

    def decode_soft(self, z: torch.Tensor) -> torch.Tensor:
        """Differentiable soft-decode (uses sigmoid probabilities directly)."""
        probs = self._bit_probs(z)
        weights = self._weights.to(z.device)
        return (probs * weights).sum(dim=-1)

    def loss(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dual-objective loss: bitwise BCE + gamma * magnitude MSE.

        *z*: raw logits ``(..., B)``.
        *target*: integer targets ``(...)``.
        """
        target = target.long()
        target_bits = self._int_to_bits(target, z.device)
        bce = F.binary_cross_entropy_with_logits(z, target_bits.float())
        return bce

    def encode(self, value: Any, device: torch.device | None = None) -> torch.Tensor:
        """Encode integer(s) as bit vectors."""
        device = device or torch.device("cpu")
        if isinstance(value, torch.Tensor):
            value = value.long().to(device)
        else:
            value = torch.tensor(value, dtype=torch.long, device=device)
        return self._int_to_bits(value, device).float()

    def _int_to_bits(self, value: torch.Tensor, device: torch.device) -> torch.Tensor:
        shifts = torch.arange(self._bits, device=device)
        return ((value.unsqueeze(-1) >> shifts) & 1).to(self._weights.dtype)

    def log_prob(self, z: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Sum of per-bit log Bernoulli probabilities."""
        target_bits = self._int_to_bits(value.long(), z.device)
        return -F.binary_cross_entropy_with_logits(
            z, target_bits, reduction="none"
        ).sum(dim=-1)

    def __repr__(self) -> str:
        return f"BitInteger({self._bits})"