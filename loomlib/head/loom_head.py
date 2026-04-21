from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..slice.allocation import SliceAllocation


class _GradientMask(torch.autograd.Function):
    """Custom autograd function that applies a binary mask on the backward pass."""

    @staticmethod
    def forward(ctx: Any, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mask)
        return z

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None


class LoomHead(nn.Module):
    """Compiled output head that replaces the standard vocabulary projection.

    Created by ``LoomCompiler.build_head()``.  Provides:

    * ``forward(hidden)``  -- project hidden states to logit space.
    * ``decode(z)``        -- walk the allocation tree and return typed values.
    * ``loss(z, targets)`` -- composite loss with per-type objectives.
    * ``apply_gradient_mask(z, branch)`` -- mask unused logit gradients.
    * ``encode_action(decoded)`` -- map decoded values to a re-entrant embedding.
    """

    def __init__(
        self,
        d_model: int,
        allocation: SliceAllocation,
        gradient_masks: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.allocation = allocation
        self.d_model = d_model
        self.projection = nn.Linear(d_model, allocation.total_logits)

        self._gradient_masks = nn.ParameterDict()
        for name, mask in gradient_masks.items():
            self._gradient_masks[name] = nn.Parameter(mask, requires_grad=False)

        # Re-entrant encoder: maps decoded logit-sized vectors back to d_model
        self.encoder_projection = nn.Linear(allocation.total_logits, d_model)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project transformer hidden states to the logit space.

        Args:
            hidden: ``(batch, d_model)`` or ``(batch, seq, d_model)``.

        Returns:
            Logit tensor of shape ``(..., total_logits)``.
        """
        return self.projection(hidden)

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(self, z: torch.Tensor) -> dict[str, Any]:
        """Decode the full logit vector into a structured dict.

        For a union, the dict includes ``__opcode__`` (branch index) and
        each branch's fields under ``{branch}.{field}`` keys.  For a flat
        model, fields live under ``{field}`` directly.
        """
        result: dict[str, Any] = {}

        if self.allocation.opcode is not None:
            opcode_entry = self.allocation.opcode
            opcode_z = opcode_entry.extract(z)
            result["__opcode__"] = opcode_entry.thunk_type.decode(opcode_z)

        for branch_name, branch in self.allocation.branches.items():
            for entry in branch.entries:
                raw = entry.extract(z)
                field_key = entry.name.removeprefix(f"{branch_name}.") if branch_name == "__root__" else entry.name
                result[field_key] = entry.thunk_type.decode(raw)

        return result

    def decode_branch(self, z: torch.Tensor, branch_name: str) -> dict[str, Any]:
        """Decode only the fields belonging to *branch_name*."""
        branch = self.allocation.branches[branch_name]
        result: dict[str, Any] = {}
        for entry in branch.entries:
            raw = entry.extract(z)
            field_name = entry.name.removeprefix(f"{branch_name}.")
            result[field_name] = entry.thunk_type.decode(raw)
        return result

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------

    def loss(
        self,
        z: torch.Tensor,
        targets: dict[str, torch.Tensor],
        active_branch: str | None = None,
        loss_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute composite loss with per-type objectives.

        Args:
            z: logit tensor ``(..., total_logits)``.
            targets: maps field names (matching allocation entry names or
                short field names for flat models) to target tensors.
                For unions, ``__opcode__`` should map to the branch index.
            active_branch: if set, only compute param losses for this branch.
            loss_weights: optional per-field scalar multipliers.

        Returns:
            ``(total_loss, breakdown)`` where *breakdown* maps field names
            to their individual loss scalars.
        """
        breakdown: dict[str, torch.Tensor] = {}
        weights = loss_weights or {}

        # Opcode loss
        if self.allocation.opcode is not None and "__opcode__" in targets:
            opcode_entry = self.allocation.opcode
            opcode_z = opcode_entry.extract(z)
            opcode_loss = opcode_entry.thunk_type.loss(opcode_z, targets["__opcode__"])
            breakdown["__opcode__"] = opcode_loss

        # Determine which branches to compute loss for
        if active_branch is not None:
            branch_iter = [(active_branch, self.allocation.branches[active_branch])]
        else:
            branch_iter = list(self.allocation.branches.items())

        for branch_name, branch in branch_iter:
            for entry in branch.entries:
                field_name = entry.name.removeprefix(f"{branch_name}.")
                target_key = field_name if branch_name == "__root__" else entry.name
                if target_key not in targets:
                    # Also try the short field name for convenience
                    if field_name in targets:
                        target_key = field_name
                    else:
                        continue
                raw = entry.extract(z)
                field_loss = entry.thunk_type.loss(raw, targets[target_key])
                w = weights.get(target_key, 1.0)
                breakdown[target_key] = field_loss * w

        total = sum(breakdown.values(), torch.tensor(0.0, device=z.device))
        return total, breakdown

    # ------------------------------------------------------------------
    # gradient masking
    # ------------------------------------------------------------------

    def apply_gradient_mask(
        self,
        z: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        """Apply the pre-computed gradient mask for *branch*.

        The forward value is unchanged; on the backward pass, gradients
        for logit dimensions outside the opcode + active branch are zeroed.
        """
        mask = self._gradient_masks[branch]
        # Broadcast mask to match z shape
        shape = [1] * (z.dim() - 1) + [mask.shape[0]]
        mask = mask.view(*shape).to(z.device)
        return _GradientMask.apply(z, mask)

    # ------------------------------------------------------------------
    # re-entrant encoding
    # ------------------------------------------------------------------

    def encode_action(self, decoded: dict[str, Any]) -> torch.Tensor:
        """Map a decoded action back to a re-entrant embedding.

        Encodes each field back to its logit representation, concatenates
        them into a full logit-sized vector, then projects to ``d_model``.
        """
        device = self.projection.weight.device
        parts: list[torch.Tensor] = []

        for entry in self.allocation.all_entries():
            field_key = entry.name.removeprefix("__root__.") if entry.name.startswith("__root__.") else entry.name
            if entry.name == "__opcode__" and "__opcode__" in decoded:
                enc = entry.thunk_type.encode(decoded["__opcode__"], device=device)
            elif field_key in decoded:
                enc = entry.thunk_type.encode(decoded[field_key], device=device)
            else:
                enc = torch.zeros(entry.size, device=device)
            parts.append(enc.to(device))

        logit_vec = torch.cat(parts, dim=-1)
        return self.encoder_projection(logit_vec)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    @property
    def total_logits(self) -> int:
        return self.allocation.total_logits

    def extra_repr(self) -> str:
        branches = list(self.allocation.branches.keys())
        return (
            f"d_model={self.d_model}, total_logits={self.total_logits}, "
            f"branches={branches}"
        )
