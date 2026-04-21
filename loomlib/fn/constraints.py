from __future__ import annotations

import torch
import torch.nn.functional as F


def less_than_penalty(z_x: torch.Tensor, z_y: torch.Tensor) -> torch.Tensor:
    """Differentiable penalty for constraint ``decode(z_x) < decode(z_y)``.

    Returns ``ReLU(decode(z_x) - decode(z_y))``, which is zero when the
    constraint is satisfied and positive otherwise.
    """
    return F.relu(z_x - z_y)


def range_penalty(
    value: torch.Tensor, lo: float, hi: float
) -> torch.Tensor:
    """Penalty for value outside ``[lo, hi]``."""
    return F.relu(lo - value) + F.relu(value - hi)


def equality_penalty(
    z_a: torch.Tensor, z_b: torch.Tensor
) -> torch.Tensor:
    """Soft penalty for ``decode(z_a) == decode(z_b)``."""
    return (z_a - z_b).pow(2)
