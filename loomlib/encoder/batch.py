from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LoomBatch:
    """Columnar tensors for a batch of field-tokenized structured instances.

    ``padding_mask`` follows PyTorch convention: True = padding, False = real.
    """

    type_ids: torch.Tensor      # [B, N_max] long
    inst_ids: torch.Tensor      # [B, N_max] long
    field_ids: torch.Tensor     # [B, N_max] long  (global field index)
    values: torch.Tensor        # [B, N_max] float (one scalar per token)
    padding_mask: torch.Tensor  # [B, N_max] bool  (True = padding)

    def to(self, device: torch.device | str) -> LoomBatch:
        return LoomBatch(**{k: v.to(device) for k, v in self.__dict__.items()})

    @property
    def batch_size(self) -> int:
        return self.type_ids.shape[0]

    @property
    def seq_len(self) -> int:
        return self.type_ids.shape[1]
