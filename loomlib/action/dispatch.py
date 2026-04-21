from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..slice.allocation import SliceAllocation


class ActionDispatch:
    """Action Layer: selects an action via opcode softmax, decodes only the
    active branch's parameters.

    Works with a compiled ``SliceAllocation`` to route logits to the
    correct branch and apply gradient masks.
    """

    def __init__(self, allocation: SliceAllocation) -> None:
        self.allocation = allocation

    def select_action(self, z: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        """Determine the selected branch for each sample in the batch.

        Returns:
            indices: int tensor of branch indices ``(batch,)``
            names: list of branch name strings
        """
        if self.allocation.opcode is None:
            branch_names = list(self.allocation.branches.keys())
            assert len(branch_names) == 1, "No opcode but multiple branches"
            idx = torch.zeros(z.shape[0] if z.dim() > 1 else 1, dtype=torch.long)
            return idx, branch_names * (idx.numel())

        opcode_entry = self.allocation.opcode
        opcode_logits = opcode_entry.extract(z)
        indices = opcode_logits.argmax(dim=-1)

        branch_names = list(self.allocation.branches.keys())
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        names = [branch_names[i.item()] for i in indices]
        return indices, names

    def decode_branch(
        self,
        z: torch.Tensor,
        branch_name: str,
    ) -> dict[str, Any]:
        """Decode all fields for a given branch."""
        branch = self.allocation.branches[branch_name]
        result: dict[str, Any] = {}
        for entry in branch.entries:
            field_name = entry.name.split(".")[-1]
            raw = entry.extract(z)
            result[field_name] = entry.thunk_type.decode(raw)
        return result

    def action_probs(self, z: torch.Tensor) -> torch.Tensor:
        """Return the softmax probabilities over action branches."""
        if self.allocation.opcode is None:
            return torch.ones(1)
        opcode_logits = self.allocation.opcode.extract(z)
        return F.softmax(opcode_logits, dim=-1)
