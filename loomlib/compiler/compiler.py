from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from ..schema.model import LoomModel
from ..schema.union import LoomUnion
from ..slice.allocation import SliceAllocation
from ..types.base import LoomType
from ..types.categorical import Categorical
from ..head.loom_head import LoomHead
from ..encoder.loom_encoder import LoomEncoder


class LoomCompiler:
    """Compiles a LoomModel or LoomUnion schema into a LoomHead.

    Three-phase pipeline:
      1. **Allocation** -- walk the schema tree, assign contiguous logit
         index ranges to every field.
      2. **Decoder graph** -- instantiate the LoomType decoder for each
         allocated slice.
      3. **Mask generation** -- pre-compute per-branch binary gradient
         masks for unions.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @classmethod
    def build_head(
        cls,
        schema: type[LoomModel] | type[LoomUnion],
        d_model: int,
    ) -> LoomHead:
        """Compile *schema* into a ready-to-use ``LoomHead`` module.

        Args:
            schema: A ``LoomModel`` or ``LoomUnion`` subclass.
            d_model: Hidden dimension of the transformer backbone.
        """
        allocation = cls._allocate(schema)
        masks = cls._generate_masks(allocation)
        return LoomHead(
            d_model=d_model,
            allocation=allocation,
            gradient_masks=masks,
        )

    @classmethod
    def build_encoder(
        cls,
        schema: type[LoomModel] | type[LoomUnion],
        d_model: int,
        max_instances: int = 512,
        branch_embeddings: dict[str, nn.Module] | None = None,
    ) -> LoomEncoder:
        """Compile *schema* into a ready-to-use ``LoomEncoder`` module.

        Args:
            schema: A ``LoomModel`` or ``LoomUnion`` subclass.
            d_model: Hidden dimension of the embedding output.
            max_instances: Maximum number of instances per sequence.
            branch_embeddings: Optional per-branch embedding overrides.
                Keys are branch names, values are ``nn.Module`` instances
                matching the :class:`DefaultBranchEmbedding` contract.
        """
        allocation = cls._allocate(schema)
        branch_names = list(allocation.branches.keys())
        branch_idx = {name: i for i, name in enumerate(branch_names)}
        branch_fields: dict[str, OrderedDict[str, LoomType]] = {}
        for bname, branch in allocation.branches.items():
            fields: OrderedDict[str, LoomType] = OrderedDict()
            for entry in branch.entries:
                fname = entry.name.removeprefix(f"{bname}.")
                fields[fname] = entry.thunk_type
            branch_fields[bname] = fields
        return LoomEncoder(
            branch_names, branch_idx, branch_fields,
            d_model, max_instances, branch_embeddings,
        )

    # ------------------------------------------------------------------
    # Phase 1: Allocation
    # ------------------------------------------------------------------

    @classmethod
    def _allocate(cls, schema: type) -> SliceAllocation:
        if isinstance(schema, type) and issubclass(schema, LoomUnion):
            return cls._allocate_union(schema)
        if isinstance(schema, type) and issubclass(schema, LoomModel):
            return cls._allocate_model(schema)
        raise TypeError(f"Expected LoomModel or LoomUnion subclass, got {schema!r}")

    @classmethod
    def _allocate_model(cls, schema: type[LoomModel]) -> SliceAllocation:
        """Allocate a flat model (no opcode, single anonymous branch)."""
        alloc = SliceAllocation()
        branch = alloc.add_branch("__root__")
        offset = 0
        for field_name, thunk_type in schema.loom_fields().items():
            size = thunk_type.logit_size()
            alloc.add_entry("__root__", field_name, offset, offset + size, thunk_type)
            offset += size
        alloc.total_logits = offset
        return alloc

    @classmethod
    def _allocate_union(cls, schema: type[LoomUnion]) -> SliceAllocation:
        """Allocate a tagged union: opcode slice first, then each branch."""
        alloc = SliceAllocation()
        branches = schema.loom_branches()
        num_branches = len(branches)

        offset = 0

        # Opcode slice
        opcode_type = Categorical(num_branches)
        alloc.add_opcode("__opcode__", offset, offset + num_branches, opcode_type)
        offset += num_branches

        # Each branch
        for branch_name, model_cls in branches.items():
            alloc.add_branch(branch_name)
            for field_name, thunk_type in model_cls.loom_fields().items():
                size = thunk_type.logit_size()
                alloc.add_entry(branch_name, field_name, offset, offset + size, thunk_type)
                offset += size

        alloc.total_logits = offset
        return alloc

    # ------------------------------------------------------------------
    # Phase 2: Decoder graph
    # ------------------------------------------------------------------
    # Decoders are already instantiated as LoomType objects stored in the
    # SliceEntry nodes during allocation.  No separate phase needed -- the
    # types carry their own decode/loss/encode logic.

    # ------------------------------------------------------------------
    # Phase 3: Mask generation
    # ------------------------------------------------------------------

    @classmethod
    def _generate_masks(
        cls,
        allocation: SliceAllocation,
    ) -> dict[str, torch.Tensor]:
        """Pre-compute a gradient mask for each branch."""
        masks: dict[str, torch.Tensor] = {}
        for branch_name in allocation.branches:
            masks[branch_name] = allocation.build_gradient_mask(branch_name)
        return masks
