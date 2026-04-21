"""Branch embedding modules for the Loom encoder.

All modules follow the branch embedding contract::

    forward(field_ids: Tensor[N], values: Tensor[N]) -> Tensor[N, d_model]

where ``field_ids`` are **0-based within the branch**.

Modules
-------
- :class:`XValEmbedding` -- Multi-scale tanh encoding (Golkar et al.).
- :class:`ThermometerEmbedding` -- Binary threshold vector projected to d_model.
- :class:`FourierValueEmbedding` -- Deterministic sinusoidal value features.
- :class:`GaussianBasisEmbedding` -- Radial basis function encoding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class XValEmbedding(nn.Module):
    """xVal multi-scale tanh encoding (Golkar et al.).

    For each field token with scalar value *x*, the embedding is::

        field_emb(f) + sum_{i in [-k, k]}  tanh(x * 10^i) * NUM_i

    where each ``NUM_i`` is a **learned** d_model vector per (field, scale)
    pair.  ``tanh`` squashes the scaled value to ``[-1, 1]``, giving
    fine-grained discrimination near zero at each scale and graceful
    saturation for outliers.

    With ``k=0`` this reduces to ``field_emb(f) + tanh(x) * NUM_0``.

    Args:
        num_fields: Number of distinct fields in this branch.
        d_model: Embedding dimension.
        k: Number of decades above and below unity.  Total scales = ``2k + 1``.
    """

    def __init__(self, num_fields: int, d_model: int, k: int = 2) -> None:
        super().__init__()
        self.k = k
        self.num_scales = 2 * k + 1
        self.num_fields = num_fields
        self.d_model = d_model

        self.field_embedding = nn.Embedding(num_fields, d_model)
        self.scale_embs = nn.Embedding(num_fields * self.num_scales, d_model)

        exponents = torch.arange(-k, k + 1, dtype=torch.float32)
        self.register_buffer("scale_factors", 10.0 ** exponents)  # [num_scales]

    def forward(self, field_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        scaled = torch.tanh(values.unsqueeze(-1) * self.scale_factors)  # [N, num_scales]

        base = field_ids.unsqueeze(-1) * self.num_scales  # [N, 1]
        offsets = torch.arange(self.num_scales, device=field_ids.device)
        emb_ids = base + offsets  # [N, num_scales]

        embs = self.scale_embs(emb_ids)  # [N, num_scales, d_model]
        xval = (scaled.unsqueeze(-1) * embs).sum(dim=1)  # [N, d_model]
        return self.field_embedding(field_ids) + xval


class ThermometerEmbedding(nn.Module):
    """Thermometer encoding: binary threshold vector projected to d_model.

    Given a scalar value, produces a binary vector where position *k* is
    ``1`` iff ``value >= threshold_k``.  The thresholds are linearly spaced
    over ``[val_min, val_max]``.  This vector is projected through a learned
    linear layer and added to a field identity embedding.

    Args:
        num_fields: Number of distinct fields in this branch.
        d_model: Embedding dimension.
        num_buckets: Number of threshold levels.
        val_min: Lower bound of the threshold range.
        val_max: Upper bound of the threshold range.
    """

    def __init__(
        self,
        num_fields: int,
        d_model: int,
        num_buckets: int = 64,
        val_min: float = 0.0,
        val_max: float = 65535.0,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.field_embedding = nn.Embedding(num_fields, d_model)
        self.thermo_proj = nn.Linear(num_buckets, d_model)
        self.register_buffer(
            "thresholds",
            torch.linspace(val_min, val_max, num_buckets),
        )

    def forward(self, field_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        thermo = (values.unsqueeze(-1) >= self.thresholds).float()  # [N, num_buckets]
        return self.field_embedding(field_ids) + self.thermo_proj(thermo)


class FourierValueEmbedding(nn.Module):
    """Deterministic sinusoidal features for scalar values.

    The value-domain analog of positional encoding.  For each token with
    scalar value *v*::

        raw = [sin(v * f_1), cos(v * f_1), ..., sin(v * f_F), cos(v * f_F)]
        embedding = field_emb(field_id) + proj(raw)

    Frequencies ``f_i`` are log-spaced using the standard transformer PE
    divisor formula.  Only the projection and field embeddings are learned;
    the frequencies are fixed buffers.

    Args:
        num_fields: Number of distinct fields in this branch.
        d_model: Embedding dimension.
        num_frequencies: Number of sin/cos pairs (raw feature dimension
            is ``2 * num_frequencies``).
    """

    def __init__(
        self,
        num_fields: int,
        d_model: int,
        num_frequencies: int = 32,
    ) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.field_embedding = nn.Embedding(num_fields, d_model)
        self.proj = nn.Linear(2 * num_frequencies, d_model)

        freqs = torch.exp(
            torch.arange(0, num_frequencies, dtype=torch.float32)
            * (-math.log(10_000.0) / num_frequencies)
        )
        self.register_buffer("freqs", freqs)  # [num_frequencies]

    def forward(self, field_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        angles = values.unsqueeze(-1) * self.freqs  # [N, num_frequencies]
        raw = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [N, 2*F]
        return self.field_embedding(field_ids) + self.proj(raw)


class GaussianBasisEmbedding(nn.Module):
    """Radial basis function encoding for scalar values.

    Widely used in scientific ML (SchNet, DimeNet).  For each value *v*::

        phi_k(v) = exp(-(v - mu_k)^2 / (2 * sigma^2))
        embedding = field_emb(field_id) + proj(phi)

    Centers ``mu_k`` are evenly spaced over ``[val_min, val_max]`` and
    ``sigma`` is the spacing between adjacent centers (so the bases
    overlap).  Gives a soft histogram-like representation.

    Args:
        num_fields: Number of distinct fields in this branch.
        d_model: Embedding dimension.
        num_centers: Number of Gaussian basis functions.
        val_min: Lower bound of the center range.
        val_max: Upper bound of the center range.
        learnable_centers: If ``True``, centers and sigma become learnable
            parameters instead of fixed buffers.
    """

    def __init__(
        self,
        num_fields: int,
        d_model: int,
        num_centers: int = 64,
        val_min: float = 0.0,
        val_max: float = 1.0,
        learnable_centers: bool = False,
    ) -> None:
        super().__init__()
        self.num_centers = num_centers
        self.field_embedding = nn.Embedding(num_fields, d_model)
        self.proj = nn.Linear(num_centers, d_model)

        centers = torch.linspace(val_min, val_max, num_centers)
        spacing = (val_max - val_min) / max(num_centers - 1, 1)
        sigma = torch.tensor(spacing)

        if learnable_centers:
            self.centers = nn.Parameter(centers)
            self.sigma = nn.Parameter(sigma)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("sigma", sigma)

    def forward(self, field_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        diff = values.unsqueeze(-1) - self.centers  # [N, num_centers]
        phi = torch.exp(-0.5 * (diff / self.sigma) ** 2)  # [N, num_centers]
        return self.field_embedding(field_ids) + self.proj(phi)
