from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from ..types.base import LoomType
from .batch import LoomBatch
from .branch_embedding import DefaultBranchEmbedding


def _sinusoidal_embeddings(max_len: int, d_model: int) -> torch.Tensor:
    """Return a ``[max_len, d_model]`` tensor of fixed sinusoidal position embeddings."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class LoomEncoder(nn.Module):
    """Field-level tokenizer and embedding module for structured instances.

    Each field of each instance becomes a single token.  The encoder
    collates raw Python data into a :class:`LoomBatch` of columnar
    tensors, then embeds them via per-branch modules (customisable).

    Instance-level positional information uses **fixed sinusoidal
    embeddings** (not learned), so all fields belonging to the same
    instance share the same positional signal.

    Created by :meth:`LoomCompiler.build_encoder`.
    """

    def __init__(
        self,
        branch_names: list[str],
        branch_idx: dict[str, int],
        branch_fields: dict[str, OrderedDict[str, LoomType]],
        d_model: int,
        max_instances: int = 512,
        branch_embeddings: dict[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.branch_names = branch_names
        self.branch_idx = branch_idx
        self.branch_fields = branch_fields
        self.d_model = d_model
        self.max_instances = max_instances

        num_branches = len(branch_names)
        self.type_embedding = nn.Embedding(num_branches, d_model)
        self.register_buffer(
            "inst_pos_embedding",
            _sinusoidal_embeddings(max_instances, d_model),
        )

        # Build global field id map and per-branch field offsets
        self._global_field_id: dict[str, dict[str, int]] = {}
        self._field_offsets: dict[str, int] = {}
        gid = 0
        for bname in branch_names:
            self._field_offsets[bname] = gid
            self._global_field_id[bname] = {}
            for fname in branch_fields[bname]:
                self._global_field_id[bname][fname] = gid
                gid += 1
        self._total_fields = gid

        # Per-branch embedding modules
        user_modules = branch_embeddings or {}
        encoders: dict[str, nn.Module] = {}
        for bname in branch_names:
            if bname in user_modules:
                encoders[bname] = user_modules[bname]
            else:
                num_fields = len(branch_fields[bname])
                encoders[bname] = DefaultBranchEmbedding(num_fields, d_model)
        self.branch_encoders = nn.ModuleDict(encoders)

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def collate(self, instances: list[list[tuple[str, dict[str, Any]]]]) -> LoomBatch:
        """Convert raw instance sequences into a :class:`LoomBatch`.

        Args:
            instances: ``batch[seq_idx] = [(branch_name, {field: value}), ...]``

        Returns:
            A :class:`LoomBatch` with all five columnar tensors.
        """
        B = len(instances)
        token_counts = [
            sum(len(self.branch_fields[b]) for b, _ in seq)
            for seq in instances
        ]
        N_max = max(token_counts) if token_counts else 0

        type_ids = torch.zeros(B, N_max, dtype=torch.long)
        inst_ids = torch.zeros(B, N_max, dtype=torch.long)
        field_ids = torch.zeros(B, N_max, dtype=torch.long)
        values = torch.zeros(B, N_max)
        padding_mask = torch.ones(B, N_max, dtype=torch.bool)

        for b, seq in enumerate(instances):
            tok = 0
            for inst_idx, (bname, field_vals) in enumerate(seq):
                bidx = self.branch_idx[bname]
                for fname, thunk in self.branch_fields[bname].items():
                    type_ids[b, tok] = bidx
                    inst_ids[b, tok] = inst_idx
                    field_ids[b, tok] = self._global_field_id[bname][fname]
                    values[b, tok] = thunk.encode(field_vals[fname]).item()
                    padding_mask[b, tok] = False
                    tok += 1

        return LoomBatch(type_ids, inst_ids, field_ids, values, padding_mask)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: LoomBatch) -> torch.Tensor:
        """Embed a collated batch into ``[B, N_max, d_model]``.

        Dispatches each branch's tokens to its own embedding module,
        then adds shared type and instance-positional embeddings.
        Padding positions remain zero.
        """
        B, N = batch.type_ids.shape
        real_mask = ~batch.padding_mask
        device = batch.values.device
        out = torch.zeros(B, N, self.d_model, device=device)

        pos = (
            self.type_embedding(batch.type_ids)
            + self.inst_pos_embedding[batch.inst_ids]
        )

        for bidx, bname in enumerate(self.branch_names):
            mask = (batch.type_ids == bidx) & real_mask
            if not mask.any():
                continue
            local_field_ids = batch.field_ids[mask] - self._field_offsets[bname]
            local_values = batch.values[mask]
            branch_emb = self.branch_encoders[bname](local_field_ids, local_values)
            out[mask] = pos[mask] + branch_emb

        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, branches={self.branch_names}, "
            f"total_fields={self._total_fields}, "
            f"max_instances={self.max_instances}"
        )
