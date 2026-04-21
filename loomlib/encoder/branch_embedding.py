from __future__ import annotations

import torch
import torch.nn as nn


class DefaultBranchEmbedding(nn.Module):
    """Default per-branch embedding: additive field identity + scaled value.

    Contract: every branch embedding module must accept::

        forward(field_ids: Tensor[N], values: Tensor[N]) -> Tensor[N, d_model]

    where ``field_ids`` are **0-based within the branch**.
    """

    def __init__(self, num_fields: int, d_model: int) -> None:
        super().__init__()
        self.field_embedding = nn.Embedding(num_fields, d_model)
        self.value_scale = nn.Embedding(num_fields, d_model)

    def forward(self, field_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return (
            self.field_embedding(field_ids)
            + values.unsqueeze(-1) * self.value_scale(field_ids)
        )
