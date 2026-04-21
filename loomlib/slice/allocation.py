from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..types.base import LoomType


@dataclass
class SliceEntry:
    """A single allocated logit slice."""
    name: str
    start: int
    end: int
    thunk_type: LoomType

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def index_range(self) -> slice:
        return slice(self.start, self.end)

    def extract(self, z: torch.Tensor) -> torch.Tensor:
        """Slice logits from a full logit vector."""
        return z[..., self.start:self.end]

    def __repr__(self) -> str:
        return (
            f"SliceEntry({self.name!r}, z[{self.start}:{self.end}], "
            f"{self.thunk_type})"
        )


@dataclass
class BranchAllocation:
    """All slices belonging to one branch of a LoomUnion."""
    name: str
    entries: list[SliceEntry] = field(default_factory=list)

    @property
    def field_names(self) -> list[str]:
        return [e.name for e in self.entries]

    def total_logits(self) -> int:
        return sum(e.size for e in self.entries)


@dataclass
class SliceAllocation:
    """Complete logit allocation tree produced by the compiler.

    Holds an optional opcode slice (for unions), a list of branch
    allocations, and provides utilities for extracting, decoding, and
    masking logit vectors.
    """
    opcode: SliceEntry | None = None
    branches: dict[str, BranchAllocation] = field(default_factory=dict)
    total_logits: int = 0

    # flat lookup: fully-qualified name -> SliceEntry
    _flat: dict[str, SliceEntry] = field(default_factory=dict, repr=False)

    def add_opcode(self, name: str, start: int, end: int, thunk_type: LoomType) -> SliceEntry:
        entry = SliceEntry(name, start, end, thunk_type)
        self.opcode = entry
        self._flat[name] = entry
        return entry

    def add_branch(self, branch_name: str) -> BranchAllocation:
        branch = BranchAllocation(branch_name)
        self.branches[branch_name] = branch
        return branch

    def add_entry(
        self,
        branch_name: str,
        field_name: str,
        start: int,
        end: int,
        thunk_type: LoomType,
    ) -> SliceEntry:
        qualified = f"{branch_name}.{field_name}"
        entry = SliceEntry(qualified, start, end, thunk_type)
        self.branches[branch_name].entries.append(entry)
        self._flat[qualified] = entry
        return entry

    def get(self, qualified_name: str) -> SliceEntry:
        return self._flat[qualified_name]

    def all_entries(self) -> list[SliceEntry]:
        """All entries in allocation order."""
        result: list[SliceEntry] = []
        if self.opcode is not None:
            result.append(self.opcode)
        for branch in self.branches.values():
            result.extend(branch.entries)
        return result

    def build_gradient_mask(self, active_branch: str, device: torch.device | None = None) -> torch.Tensor:
        """Build a binary mask: 1 for opcode + active branch, 0 elsewhere."""
        device = device or torch.device("cpu")
        mask = torch.zeros(self.total_logits, device=device)
        if self.opcode is not None:
            mask[self.opcode.start:self.opcode.end] = 1.0
        if active_branch in self.branches:
            for entry in self.branches[active_branch].entries:
                mask[entry.start:entry.end] = 1.0
        return mask

    def pretty_print(self) -> str:
        lines = ["SliceAllocation:"]
        for entry in self.all_entries():
            lines.append(f"  z[{entry.start}:{entry.end}]  {entry.name:<30s}  {entry.thunk_type}")
        lines.append(f"  Total logits: {self.total_logits}")
        return "\n".join(lines)
